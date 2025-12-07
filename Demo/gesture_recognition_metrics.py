import os
import sys
import platform
import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import joblib
import json
from pathlib import Path
from collections import deque, defaultdict, Counter
from tensorflow.lite.python.interpreter import Interpreter
import pandas as pd
from datetime import datetime
import threading
import queue

# --- 설정 및 상수 ---
ARTIFACTS_DIR = Path("./artifacts")
HMM_DIR = ARTIFACTS_DIR / "hmms"
ERGODIC_MODEL_PATH = HMM_DIR / "ergodic.pkl"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"
TFLITE_ENCODER_PATH = ARTIFACTS_DIR / "encoder_model_fp16_select.tflite"

WINDOW_SIZE = 82
THRESHOLD_DIFF = 0.0
PREDICTION_INTERVAL = 2 
P_ON = 3   # 손 등장 프레임 수 
P_OFF = 15  # 손 사라짐 프레임 수 

COMMAND_MAP = {
    "1": "Hidden",
    "2": "Start",
    "3": "Previous",
    "4": "Next",
    "5": "Stop",
    "6": "First",
    "7": "White",
    "8": "Last",
    "9": "Black",
    "10": "Bye",
}

# 전역 인터프리터
TFLITE_INTERP = None
TFLITE_IN = None
TFLITE_OUT = None


# --- 모델 로딩 함수들  ---
def load_hmms_and_labels():
    print("모델 로딩 중 (HMM/라벨)...")
    if not ARTIFACTS_DIR.exists():
        print(" - artifacts 폴더가 없습니다.")
        return None, None, None

    if not TFLITE_ENCODER_PATH.exists():
        print(f" - TFLite 인코더 파일을 찾을 수 없습니다: {TFLITE_ENCODER_PATH}")
        return None, None, None

    try:
        with open(ERGODIC_MODEL_PATH, "rb") as f:
            ergodic_model = joblib.load(f)
        print(f" - Ergodic 모델 로드 완료")

        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        gesture_names = list(label_map.keys())
        print(f" - 라벨 맵 로드 완료: {gesture_names}")

        gesture_hmms = {}
        for gesture_name in gesture_names:
            hmm_file_path = HMM_DIR / f"hmm_{gesture_name}.pkl"
            with open(hmm_file_path, "rb") as f:
                gesture_hmms[gesture_name] = joblib.load(f)
        print(f" - {len(gesture_hmms)}개의 개별 HMM 모델 로드 완료.")
        return gesture_hmms, ergodic_model, gesture_names
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None, None, None


def load_tflite_encoder(model_path: str):
    interp = Interpreter(model_path=model_path, num_threads=os.cpu_count())
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f" - TFLite encoder loaded")
    return interp, in_det, out_det


def encode_with_tflite(sequence_data: np.ndarray) -> np.ndarray:
    global TFLITE_INTERP, TFLITE_IN, TFLITE_OUT
    x = sequence_data[np.newaxis, ...]
    if x.dtype != TFLITE_IN["dtype"]:
        x = x.astype(TFLITE_IN["dtype"], copy=False)
    TFLITE_INTERP.set_tensor(TFLITE_IN["index"], x)
    TFLITE_INTERP.invoke()
    latent = TFLITE_INTERP.get_tensor(TFLITE_OUT["index"])[0]
    return latent


def recognize_gesture_from_window(sequence_data, hmms, ergodic, gesture_names):
    """제스처 인식 (동기식 - 레거시)"""
    latent_seq = encode_with_tflite(sequence_data)
    final_model_score = ergodic.score(latent_seq)
    best_gesture, max_diff = None, -np.inf
    for g_name in gesture_names:
        g_score = hmms[g_name].score(latent_seq)
        diff = g_score - final_model_score
        if diff > max_diff:
            max_diff, best_gesture = diff, g_name
    if max_diff >= THRESHOLD_DIFF:
        return best_gesture
    return None


# --- 비동기 추론 워커 ---
def infer_worker(
    infer_q: "queue.Queue[tuple]",
    result_q: "queue.Queue[tuple]",
    hmms,
    ergodic,
    gesture_names,
):
    """
    비동기 추론 워커 스레드
    infer_q item: (seq_id:int, enqueue_ts:float, sequence_data:np.ndarray)
    result_q item: (seq_id:int, gesture:str|None, enc_ms:float, hmm_ms:float, end2end_ms:float)
    """
    while True:
        item = infer_q.get()
        if item is None:
            break  # 종료 신호
        try:
            seq_id, t_enq, seq = item
            t0 = time.perf_counter()
            latent_seq = encode_with_tflite(seq)
            t1 = time.perf_counter()
            base = ergodic.score(latent_seq)
            best, diff_max = None, -np.inf
            for g in gesture_names:
                if g not in hmms:
                    continue
                d = hmms[g].score(latent_seq) - base
                if d > diff_max:
                    diff_max, best = d, g
            pred = best if diff_max >= THRESHOLD_DIFF else None
            t2 = time.perf_counter()
            
            enc_ms = (t1 - t0) * 1000.0
            hmm_ms = (t2 - t1) * 1000.0
            end2end_ms = (t2 - t_enq) * 1000.0  # 큐 입력 ~ 결과 도출
            
            result_q.put((seq_id, pred, enc_ms, hmm_ms, end2end_ms))
        except Exception as e:
            print(f"[infer_worker] 오류: {e}")


# --- 카메라 유틸  ---
def parse_resolution(resolution_str: str):
    if not resolution_str:
        return None, None
    try:
        parts = resolution_str.lower().split("x")
        if len(parts) != 2:
            return None, None
        w = int(parts[0].strip())
        h = int(parts[1].strip())
        if w <= 0 or h <= 0:
            return None, None
        return w, h
    except Exception:
        return None, None


def apply_stream_parameters(url: str, resolution: str | None, fps: float | None) -> str:
    params = []
    if resolution:
        params.append(f"res={resolution}")
    if fps:
        fps_int = int(fps)
        params.append(f"fps={fps_int}")
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{'&'.join(params)}"


def configure_capture(cap, width=None, height=None, fps=None):
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def init_camera(camera_source, width=None, height=None, fps=None):
    is_ipcam = isinstance(camera_source, str)
    if is_ipcam:
        print(f"안드로이드 카메라 연결 시도: {camera_source}")
    else:
        print(f"로컬 웹캠 연결 시도: 인덱스 {camera_source}")

    cap = cv2.VideoCapture(camera_source)

    if is_ipcam:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return None

    configure_capture(cap, width=width, height=height, fps=fps)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None

    return cap


# --- 제스처 테스트 클래스 ---
class GestureTestResult:
    """단일 제스처 테스트 결과"""
    def __init__(self, target_gesture: str):
        self.target_gesture = target_gesture
        self.total_samples = 0
        self.correct_count = 0
        self.e2e_times = []  # End-to-End 지연시간 (ms)
        self.response_times = []  # Response 시간 (ms)
        self.false_positives = 0  # 다른 제스처를 이 제스처로 잘못 인식한 횟수
        self.total_other_gestures = 0  # 다른 제스처 수행 횟수
        # 각 샘플별 상세 기록 (엑셀 저장용)
        self.sample_records = []  # 샘플별 기록 리스트
        self.fp_records = []  # False Positive 기록 리스트
        
    def add_result(self, recognized: str, e2e_ms: float, response_ms: float, is_target: bool,
                   fpr_percent: float = 0.0, fp_per_min: float = 0.0):
        """테스트 결과 추가"""
        if is_target:
            self.total_samples += 1
            is_correct = recognized == self.target_gesture if recognized else False
            if is_correct:
                self.correct_count += 1
            
            # 샘플 기록 저장
            error_type = "ok" if is_correct else (f"wrong_prediction_{recognized}" if recognized else "no_prediction")
            record = {
                "gesture": self.target_gesture,
                "predicted": recognized if recognized else "none",
                "correct": "Y" if is_correct else "N",
                "error_type": error_type,
                "E2E_ms": round(e2e_ms, 2),
                "Response_ms": round(response_ms, 2),
                "FPR_percent": round(fpr_percent, 2),
                "FP_per_min": round(fp_per_min, 2)
            }
            self.sample_records.append(record)
            
            self.e2e_times.append(e2e_ms)
            self.response_times.append(response_ms)
        else:
            # 다른 제스처를 수행했는데 이 제스처로 인식된 경우 (False Positive)
            self.total_other_gestures += 1
            if recognized == self.target_gesture:
                self.false_positives += 1
                # False Positive 기록 저장
                fp_record = {
                    "gesture": self.target_gesture,
                    "predicted": recognized,
                    "correct": "N",
                    "error_type": "false_positive",
                    "E2E_ms": round(e2e_ms, 2),
                    "Response_ms": round(response_ms, 2),
                    "FPR_percent": 0.0, 
                    "FP_per_min": 0.0  
                }
                self.fp_records.append(fp_record)
    
    def get_recognition_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.correct_count / self.total_samples) * 100.0
    
    def get_e2e_mean(self) -> float:
        if len(self.e2e_times) == 0:
            return 0.0
        return float(np.mean(self.e2e_times))
    
    def get_response_mean(self) -> float:
        if len(self.response_times) == 0:
            return 0.0
        return float(np.mean(self.response_times))
    
    def get_fpr_percent(self) -> float:
        """False Positive Rate (%)"""
        if self.total_other_gestures == 0:
            return 0.0
        return (self.false_positives / self.total_other_gestures) * 100.0
    
    def get_fp_per_min(self, total_test_duration_min: float) -> float:
        """분당 False Positive 수"""
        if total_test_duration_min <= 0:
            return 0.0
        return self.false_positives / total_test_duration_min


# --- 제스처 테스트 함수 (비동기 추론 + 히스테리시스 적용) ---
def test_gesture(
    cap,
    mp_hands,
    gesture_hmms,
    ergodic_model,
    gesture_names,
    target_gesture: str,
    samples_per_gesture: int = 100,
    gesture_name_display: str = None
):
    """
    특정 제스처를 테스트합니다 (비동기 추론 + 히스테리시스 기반 세그먼트 감지).
    손이 등장하고 사라질 때까지를 하나의 제스처로 간주합니다.
    
    Returns:
        GestureTestResult 객체
    """
    result = GestureTestResult(target_gesture)
    
    if gesture_name_display is None:
        gesture_name_display = COMMAND_MAP.get(target_gesture, f"Gesture {target_gesture}")
    
    window_name = f"Gesture Test - {gesture_name_display}({target_gesture})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # 비동기 추론 큐/워커 시작 (main_251021.py와 동일하게 maxsize=1)
    infer_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
    result_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=infer_worker,
        args=(infer_q, result_q, gesture_hmms, ergodic_model, gesture_names),
        daemon=True,
    )
    worker.start()
    
    landmark_buffer = deque(maxlen=WINDOW_SIZE)
    frame_counter = 0
    seq_id = 0  # 비동기 요청 ID
    test_start_time = time.time()
    
    # 히스테리시스 기반 세그먼트 감지 
    present_cnt, absent_cnt = 0, 0
    in_segment = False
    seg_t0 = None  # 세그먼트 시작 시간
    first_pred_time = None  # 첫 예측 시간 (ART 계산용)
    segment_predictions = []  # 세그먼트 내 예측 결과 (최빈값 계산용)
    segment_e2e_times = []  # 세그먼트 내 E2E 시간
    
    # FPR 계산용
    fp_count = 0
    non_gesture_opps = 0
    non_gesture_start = time.perf_counter()
    
    print(f"\n[{gesture_name_display}({target_gesture})] 테스트 시작 (비동기 모드)")
    print(f"목표: {samples_per_gesture}개 제스처 세션 수집")
    print("손을 카메라 앞에 보여주세요. 손이 등장하고 사라질 때마다 하나의 제스처로 기록됩니다.")
    print("'q'를 눌러 종료")
    
    try:
        while result.total_samples < samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_counter += 1
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = mp_hands.process(image_rgb)
            
            # 손 랜드마크 추출
            current_landmarks = np.zeros(2 * 21 * 3, dtype=np.float32)
            hand_detected = False
            
            if results_mp.multi_hand_landmarks:
                hand_detected = True
                for i, hand_landmarks in enumerate(results_mp.multi_hand_landmarks):
                    start_idx = i * 63
                    landmarks_flat = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()
                    current_landmarks[
                        start_idx : start_idx + len(landmarks_flat)
                    ] = landmarks_flat
            
            landmark_buffer.append(current_landmarks)
            
            # 히스테리시스
            if hand_detected:
                present_cnt += 1
                absent_cnt = 0
            else:
                absent_cnt += 1
                present_cnt = 0
            
            # 비동기 추론 요청 
            if (len(landmark_buffer) == WINDOW_SIZE) and (frame_counter % PREDICTION_INTERVAL == 0):
                if not infer_q.full():
                    seq = np.array(landmark_buffer, dtype=np.float32)
                    seq_id += 1
                    infer_q.put((seq_id, time.perf_counter(), seq))  # perf_counter로 타이머 일관 유지
                # 비제스처 구간 기회 카운트
                if not in_segment:
                    non_gesture_opps += 1
            
            # 비동기 결과 수거
            while True:
                try:
                    rid, pred, enc_ms, hmm_ms, end2end_ms = result_q.get_nowait()
                    
                    if pred is not None:
                        # 세그먼트가 활성화되어 있으면 수집
                        if in_segment:
                            if first_pred_time is None:
                                # 첫 예측 시간 기록 (ART 계산용)
                                first_pred_time = time.perf_counter()
                            segment_predictions.append(pred)
                            segment_e2e_times.append(end2end_ms)
                            # 디버깅: 인식 결과 수집 확인
                            if len(segment_predictions) % 5 == 0:  # 5개마다 로그
                                print(f"    [인식 수집] {len(segment_predictions)}개: {pred}")
                        else:
                            # 세그먼트가 없으면 FPR 계산
                            fp_count += 1
                except queue.Empty:
                    break
            
            # 세그먼트 시작 (히스테리시스 기반)
            if (not in_segment) and (present_cnt >= P_ON):
                in_segment = True
                seg_t0 = time.perf_counter()
                first_pred_time = None
                segment_predictions = []
                segment_e2e_times = []
                
                # 세그먼트 시작 시점에 이미 큐에 있는 결과도 수거
                while True:
                    try:
                        rid, pred, enc_ms, hmm_ms, end2end_ms = result_q.get_nowait()
                        if pred is not None:
                            if first_pred_time is None:
                                first_pred_time = time.perf_counter()
                            segment_predictions.append(pred)
                            segment_e2e_times.append(end2end_ms)
                    except queue.Empty:
                        break
                
                print(f"  [세그먼트 시작] 손이 검출되었습니다. (초기 예측: {len(segment_predictions)}개)")
            
            # 세그먼트 종료 → 기록 (히스테리시스 기반)
            if in_segment and (absent_cnt >= P_OFF):
                # 세그먼트 종료 전에 마지막으로 결과 수거 (대기 시간 추가)
                wait_start = time.perf_counter()
                while time.perf_counter() - wait_start < 0.1:  # 100ms 대기
                    try:
                        rid, pred, enc_ms, hmm_ms, end2end_ms = result_q.get_nowait()
                        if pred is not None:
                            if first_pred_time is None:
                                first_pred_time = time.perf_counter()
                            segment_predictions.append(pred)
                            segment_e2e_times.append(end2end_ms)
                    except queue.Empty:
                        break
                
                in_segment = False
                
                # ART 계산: 손 등장 ~ 첫 예측
                resp_ms = 0.0
                if first_pred_time is not None:
                    resp_ms = (first_pred_time - seg_t0) * 1000.0
                
                # 최빈값으로 최종 예측 결정
                if segment_predictions:
                    pred_counts = Counter(segment_predictions)
                    pred_label = pred_counts.most_common(1)[0][0]
                    correct = "Y" if pred_label == target_gesture else "N"
                    avg_e2e = np.mean(segment_e2e_times) if segment_e2e_times else 0.0
                    print(f"  [세그먼트 종료] 예측 수집: {len(segment_predictions)}개, 최빈값: {pred_label}")
                else:
                    pred_label = None
                    correct = "N"
                    avg_e2e = 0.0
                    print(f"  [세그먼트 종료] 예측 수집 실패 (버퍼: {len(landmark_buffer)}/{WINDOW_SIZE}, "
                          f"세그먼트 지속: {(time.perf_counter() - seg_t0)*1000:.1f}ms)")
                
                # FPR/FP_per_min 계산
                non_gesture_sec = time.perf_counter() - non_gesture_start
                FPR = (100.0 * fp_count / non_gesture_opps) if non_gesture_opps > 0 else 0.0
                FP_per_min = (fp_count / (non_gesture_sec / 60.0)) if non_gesture_sec > 1e-6 else 0.0
                
                # 샘플 수집
                if pred_label is not None:
                    is_target = True
                    result.add_result(pred_label, avg_e2e, resp_ms, is_target, FPR, FP_per_min)
                    print(f"  [세그먼트 종료] 샘플 {result.total_samples}: {COMMAND_MAP.get(pred_label, pred_label)} "
                          f"(E2E: {avg_e2e:.2f}ms, Response: {resp_ms:.2f}ms, correct={correct})")
                else:
                    print(f"  [세그먼트 종료] 제스처가 인식되지 않았습니다.")
            
            # 화면 표시 (성능 최적화)
            canvas = frame.copy()
            
            # 손 랜드마크 그리기
            if results_mp.multi_hand_landmarks:
                for hand_landmarks in results_mp.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2
                        ),
                    )
            
            # 정보 표시
            info_y = 30
            cv2.rectangle(canvas, (0, 0), (W, 200), (0, 0, 0), -1)
            
            cv2.putText(
                canvas,
                f"Target: {gesture_name_display}({target_gesture})",
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            
            cv2.putText(
                canvas,
                f"Collected: {result.total_samples}/{samples_per_gesture}",
                (10, info_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            
            # 세그먼트 상태 표시
            if in_segment:
                status_text = "Segment: ACTIVE (손 검출 중)"
                status_color = (0, 255, 0)
            else:
                status_text = "Segment: WAITING (손을 보여주세요)"
                status_color = (0, 165, 255)
            
            cv2.putText(
                canvas,
                status_text,
                (10, info_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            
            # 최근 인식 결과 표시
            if segment_predictions:
                latest_pred = segment_predictions[-1]
                color = (0, 255, 0) if latest_pred == target_gesture else (0, 0, 255)
                cv2.putText(
                    canvas,
                    f"Latest: {COMMAND_MAP.get(latest_pred, latest_pred)}",
                    (10, info_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            
            # 버퍼 상태 표시 (디버깅용)
            buffer_status = f"Buffer: {len(landmark_buffer)}/{WINDOW_SIZE}"
            cv2.putText(
                canvas,
                buffer_status,
                (10, info_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
            )
            
            cv2.putText(
                canvas,
                "Press 'q' to finish",
                (10, info_y + 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            
            cv2.imshow(window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        # 워커 스레드 종료
        try:
            infer_q.put_nowait(None)
        except Exception:
            pass
        try:
            worker.join(timeout=1.0)
        except Exception:
            pass
        
        cv2.destroyWindow(window_name)
        test_duration_min = (time.time() - test_start_time) / 60.0
        result.test_duration_min = test_duration_min
    
    return result


# --- 전체 제스처 테스트 및 False Positive 계산 ---
def test_all_gestures(
    cap,
    mp_hands,
    gesture_hmms,
    ergodic_model,
    gesture_names,
    samples_per_gesture: int = 100
):
    """모든 제스처를 테스트하고 False Positive를 계산"""
    
    # 각 제스처별 테스트 결과
    gesture_results = {}
    
    # 1단계: 각 제스처별로 테스트
    print("\n" + "=" * 80)
    print("1단계: 각 제스처별 테스트 시작")
    print("=" * 80)
    
    for gesture_id in gesture_names:
        gesture_display = COMMAND_MAP.get(gesture_id, f"Gesture {gesture_id}")
        result = test_gesture(
            cap, mp_hands, gesture_hmms, ergodic_model, gesture_names,
            gesture_id, samples_per_gesture, gesture_display
        )
        gesture_results[gesture_id] = result
    
    # 2단계: False Positive 계산을 위한 전체 제스처 테스트
    print("\n" + "=" * 80)
    print("2단계: False Positive 계산을 위한 전체 제스처 테스트")
    print("각 제스처를 순서대로 수행하세요. 다른 제스처를 수행했을 때")
    print("각 제스처로 잘못 인식되는 경우를 계산합니다.")
    print("=" * 80)
    
    # 각 제스처에 대해 다른 제스처를 수행했을 때의 False Positive 계산
    total_fp_test_time = time.time()
    
    window_name = "False Positive Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    landmark_buffer = deque(maxlen=WINDOW_SIZE)
    frame_counter = 0
    
    # 각 제스처별로 다른 제스처 수행 횟수 추적
    gesture_performed_count = defaultdict(int)  # 실제 수행한 제스처별 횟수
    
    print("\n각 제스처를 수행하세요. '1'-'0' 키를 눌러 현재 수행 중인 제스처를 표시하세요.")
    print("(1=Hidden, 2=Start, 3=Previous, 4=Next, 5=Stop, 6=First, 7=White, 8=Last, 9=Black, 0=Bye)")
    print("'q'를 눌러 종료")
    
    current_performed_gesture = None  # 현재 사용자가 수행 중인 제스처
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_counter += 1
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = mp_hands.process(image_rgb)
            
            current_landmarks = np.zeros(2 * 21 * 3, dtype=np.float32)
            hand_detected = False
            
            if results_mp.multi_hand_landmarks:
                hand_detected = True
                for i, hand_landmarks in enumerate(results_mp.multi_hand_landmarks):
                    start_idx = i * 63
                    landmarks_flat = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()
                    current_landmarks[
                        start_idx : start_idx + len(landmarks_flat)
                    ] = landmarks_flat
            
            landmark_buffer.append(current_landmarks)
            
            recognized_gesture = None
            e2e_ms = 0.0
            
            if len(landmark_buffer) == WINDOW_SIZE and frame_counter % PREDICTION_INTERVAL == 0:
                if hand_detected:
                    t0 = time.time()
                    sequence_data = np.array(landmark_buffer, dtype=np.float32)
                    recognized_gesture = recognize_gesture_from_window(
                        sequence_data, gesture_hmms, ergodic_model, gesture_names
                    )
                    t1 = time.time()
                    e2e_ms = (t1 - t0) * 1000.0
            
            # 키 입력 처리 (현재 수행 중인 제스처 표시)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key >= ord('1') and key <= ord('9'):
                current_performed_gesture = str(key - ord('0'))
            elif key == ord('0'):
                current_performed_gesture = "10"
            
            # 화면 표시
            canvas = frame.copy()
            
            if results_mp.multi_hand_landmarks:
                for hand_landmarks in results_mp.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2
                        ),
                    )
            
            cv2.rectangle(canvas, (0, 0), (W, 120), (0, 0, 0), -1)
            
            if current_performed_gesture:
                performed_name = COMMAND_MAP.get(current_performed_gesture, f"Gesture {current_performed_gesture}")
                cv2.putText(
                    canvas,
                    f"Performing: {performed_name}({current_performed_gesture})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
            
            if recognized_gesture:
                color = (0, 255, 0) if recognized_gesture == current_performed_gesture else (0, 0, 255)
                cv2.putText(
                    canvas,
                    f"Recognized: {COMMAND_MAP.get(recognized_gesture, recognized_gesture)}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
            
            cv2.putText(
                canvas,
                "Press 1-0 to mark current gesture, 'q' to finish",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            
            cv2.imshow(window_name, canvas)
            
            # False Positive 계산
            if recognized_gesture is not None and current_performed_gesture is not None:
                # 실제 수행한 제스처와 인식된 제스처가 다른 경우
                if recognized_gesture != current_performed_gesture:
                    # 다른 제스처를 수행했는데 이 제스처로 잘못 인식된 경우 (False Positive)
                    gesture_performed_count[current_performed_gesture] += 1
                    gesture_results[recognized_gesture].add_result(
                        recognized_gesture, e2e_ms, e2e_ms, is_target=False
                    )
                else:
                    # 올바르게 인식된 경우 (True Positive, 카운트만)
                    gesture_performed_count[current_performed_gesture] += 1
    
    finally:
        cv2.destroyWindow(window_name)
        total_fp_test_duration_min = (time.time() - total_fp_test_time) / 60.0
    
    # 각 결과에 전체 테스트 시간 설정 및 FPR, FP_per_min 업데이트
    for gesture_id, result in gesture_results.items():
        result.total_fp_test_duration_min = total_fp_test_duration_min
        fpr_percent = result.get_fpr_percent()
        fp_per_min = result.get_fp_per_min(total_fp_test_duration_min)
        
        # 샘플 기록에 FPR과 FP_per_min 업데이트
        for record in result.sample_records:
            record["FPR_percent"] = round(fpr_percent, 2)
            record["FP_per_min"] = round(fp_per_min, 2)
        
        # False Positive 기록에 FPR과 FP_per_min 업데이트
        for fp_record in result.fp_records:
            fp_record["FPR_percent"] = round(fpr_percent, 2)
            fp_record["FP_per_min"] = round(fp_per_min, 2)
    
    return gesture_results


# --- 엑셀 파일 저장 함수 ---
def save_gesture_results_to_excel(gesture_results: dict, output_dir: Path = None):
    """각 제스처별 테스트 결과를 엑셀 파일로 저장"""
    if output_dir is None:
        output_dir = Path("./gesture_test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 각 제스처별로 엑셀 파일 저장
    for gesture_id in sorted(gesture_results.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        result = gesture_results[gesture_id]
        gesture_name = COMMAND_MAP.get(gesture_id, f"Gesture {gesture_id}")
        
        # 샘플 기록과 False Positive 기록 합치기
        all_records = result.sample_records.copy()
        
        # False Positive 기록에 FPR과 FP_per_min 업데이트
        total_fp_test_duration = getattr(result, 'total_fp_test_duration_min', 1.0)
        fpr_percent = result.get_fpr_percent()
        fp_per_min = result.get_fp_per_min(total_fp_test_duration)
        
        # 샘플 기록에 FPR과 FP_per_min 업데이트
        for record in all_records:
            record["FPR_percent"] = round(fpr_percent, 2)
            record["FP_per_min"] = round(fp_per_min, 2)
        
        # False Positive 기록 추가
        for fp_record in result.fp_records:
            fp_record["FPR_percent"] = round(fpr_percent, 2)
            fp_record["FP_per_min"] = round(fp_per_min, 2)
            all_records.append(fp_record)
        
        if len(all_records) > 0:
            # DataFrame 생성
            df = pd.DataFrame(all_records)
            
            # 컬럼 순서 지정
            column_order = ["gesture", "predicted", "correct", "error_type", 
                          "E2E_ms", "Response_ms", "FPR_percent", "FP_per_min"]
            df = df[column_order]
            
            # 엑셀 파일 저장
            excel_filename = output_dir / f"gesture_{gesture_id}_{gesture_name.lower()}_{timestamp}.xlsx"
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"  제스처 {gesture_id} ({gesture_name}) 결과 저장: {excel_filename}")
    
    # 전체 결과를 하나의 엑셀 파일에 여러 시트로 저장
    all_records_combined = []
    for gesture_id in sorted(gesture_results.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        result = gesture_results[gesture_id]
        all_records_combined.extend(result.sample_records)
        all_records_combined.extend(result.fp_records)
    
    if len(all_records_combined) > 0:
        # 전체 결과 파일
        combined_filename = output_dir / f"all_gestures_{timestamp}.xlsx"
        with pd.ExcelWriter(combined_filename, engine='openpyxl') as writer:
            # 전체 데이터를 하나의 시트에
            df_all = pd.DataFrame(all_records_combined)
            if len(df_all) > 0:
                column_order = ["gesture", "predicted", "correct", "error_type", 
                              "E2E_ms", "Response_ms", "FPR_percent", "FP_per_min"]
                df_all = df_all[column_order]
                df_all.to_excel(writer, sheet_name='All Results', index=False)
            
            # 각 제스처별로 시트 생성
            for gesture_id in sorted(gesture_results.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                result = gesture_results[gesture_id]
                gesture_name = COMMAND_MAP.get(gesture_id, f"Gesture {gesture_id}")
                
                records = result.sample_records.copy()
                records.extend(result.fp_records)
                
                if len(records) > 0:
                    df_gesture = pd.DataFrame(records)
                    column_order = ["gesture", "predicted", "correct", "error_type", 
                                  "E2E_ms", "Response_ms", "FPR_percent", "FP_per_min"]
                    df_gesture = df_gesture[column_order]
                    sheet_name = f"Gesture_{gesture_id}"
                    if len(sheet_name) > 31:  # Excel 시트 이름 제한
                        sheet_name = sheet_name[:31]
                    df_gesture.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n전체 결과 저장: {combined_filename}")


# --- 결과 출력 함수 ---
def print_results_table(gesture_results: dict):
    """결과를 표 형식으로 출력"""
    print("\n" + "=" * 120)
    print("Gesture Recognition Metrics")
    print("=" * 120)
    print(f"{'Gesture':<20} {'TestData':<10} {'Correct':<10} {'Recognition Rate':<18} "
          f"{'E2E_ms':<12} {'Response_ms':<15} {'FPR_percent':<15} {'FP_per_min':<12}")
    print("-" * 120)
    
    for gesture_id in sorted(gesture_results.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        result = gesture_results[gesture_id]
        gesture_name = COMMAND_MAP.get(gesture_id, f"Gesture {gesture_id}")
        gesture_label = f"{gesture_name.lower()}({gesture_id})"
        
        test_data = result.total_samples
        correct = result.correct_count
        recognition_rate = result.get_recognition_rate()
        e2e_ms = result.get_e2e_mean()
        response_ms = result.get_response_mean()
        fpr_percent = result.get_fpr_percent()
        
        # FP_per_min 계산 (전체 테스트 시간 기준)
        total_duration = getattr(result, 'total_fp_test_duration_min', 1.0)
        fp_per_min = result.get_fp_per_min(total_duration)
        
        print(f"{gesture_label:<20} {test_data:<10} {correct:<10} {recognition_rate:<18.2f} "
              f"{e2e_ms:<12.4f} {response_ms:<15.4f} {fpr_percent:<15.8f} {fp_per_min:<12.8f}")
    
    print("=" * 120)


# --- 메인 함수 ---
def main():
    global TFLITE_INTERP, TFLITE_IN, TFLITE_OUT
    
    parser = argparse.ArgumentParser(
        description="제스처 인식 메트릭 테스트 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
카메라 소스 옵션:
  - 로컬 웹캠: --camera 0 (기본값)
  - 안드로이드 IP Webcam: --camera http://[스마트폰IP]:8080/video
        """
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="카메라 소스 (웹캠 인덱스 또는 IP Webcam URL, 기본값: 0)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="원하는 해상도 (예: 640x480)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="원하는 FPS (예: 30)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="제스처당 테스트 샘플 수 (기본값: 100)"
    )
    parser.add_argument(
        "--gesture",
        type=str,
        default=None,
        help="테스트할 제스처 번호 (1-10). 지정하지 않으면 모든 제스처 테스트"
    )
    args = parser.parse_args()
    
    # 모델 로딩
    gesture_hmms, ergodic_model, gesture_names = load_hmms_and_labels()
    if not all([gesture_hmms, ergodic_model, gesture_names]):
        print("오류: 모델 로딩 실패")
        return
    
    try:
        TFLITE_INTERP, TFLITE_IN, TFLITE_OUT = load_tflite_encoder(
            str(TFLITE_ENCODER_PATH)
        )
    except Exception as e:
        print(f"오류: TFLite 인코더 로딩 실패: {e}")
        return
    
    # MediaPipe 초기화
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    
    # 카메라 초기화
    target_width, target_height = parse_resolution(args.resolution) if args.resolution else (None, None)
    
    camera_arg = args.camera
    try:
        camera_source = int(camera_arg)
    except ValueError:
        camera_source = apply_stream_parameters(
            camera_arg,
            args.resolution if args.resolution else None,
            args.fps,
        )
    
    cap = init_camera(
        camera_source,
        width=target_width,
        height=target_height,
        fps=args.fps,
    )
    if cap is None:
        print("오류: 카메라를 열 수 없습니다.")
        return
    
    try:
        gesture_results = {}
        
        # 테스트할 제스처 선택
        if args.gesture:
            # 특정 제스처만 테스트
            target_gesture_id = args.gesture
            if target_gesture_id not in gesture_names:
                print(f"오류: 제스처 '{target_gesture_id}'를 찾을 수 없습니다.")
                print(f"사용 가능한 제스처: {gesture_names}")
                return
            
            gesture_display = COMMAND_MAP.get(target_gesture_id, f"Gesture {target_gesture_id}")
            print(f"\n제스처 {target_gesture_id} ({gesture_display}) 테스트를 시작합니다.")
            
            result = test_gesture(
                cap, mp_hands, gesture_hmms, ergodic_model, gesture_names,
                target_gesture_id, args.samples, gesture_display
            )
            gesture_results[target_gesture_id] = result
            
            # 결과 출력
            print(f"\n[{gesture_display}({target_gesture_id})] 테스트 완료")
            print(f"  총 세션 수: {result.total_samples}")
            print(f"  정확히 인식: {result.correct_count}")
            print(f"  인식률: {result.get_recognition_rate():.2f}%")
            print(f"  평균 E2E: {result.get_e2e_mean():.2f} ms")
            print(f"  평균 Response: {result.get_response_mean():.2f} ms")
            
        else:
            # 모든 제스처 테스트
            gesture_results = test_all_gestures(
                cap, mp_hands, gesture_hmms, ergodic_model, gesture_names,
                samples_per_gesture=args.samples
            )
            
            # 결과 출력
            print_results_table(gesture_results)
        
        # 엑셀 파일로 저장
        if gesture_results:
            print("\n" + "=" * 80)
            print("엑셀 파일로 결과 저장 중...")
            print("=" * 80)
            save_gesture_results_to_excel(gesture_results)
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mp_hands.close()
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()

