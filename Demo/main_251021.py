import cv2
import numpy as np
import mediapipe as mp
import os
import time
import joblib
import pyautogui
import platform
import subprocess
import json
from pathlib import Path
from collections import deque, defaultdict
import threading
import queue
import sys

# ------------------------------------------
# TFLite 인터프리터 임포트
# ------------------------------------------
try:
    from tensorflow.lite.python.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter
    except Exception as e:
        raise ImportError("TensorFlow Lite Interpreter Error" "" "") from e

# --- 1. 설정 및 상수 ---
ARTIFACTS_DIR = Path("./artifacts")
HMM_DIR = ARTIFACTS_DIR / "hmms"
ERGODIC_MODEL_PATH = HMM_DIR / "ergodic.pkl"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"

# TFLite 인코더 경로
TFLITE_ENCODER_PATH = ARTIFACTS_DIR / "encoder_model_fp16_select.tflite"

# PPT 파일명
PPT_FILENAME = "CM20315_12_Transformers.pptx"

# 파라미터
WINDOW_SIZE = 82
THRESHOLD_DIFF = 0.0

# 같은 제스처 재실행 쿨다운(초)
COMMAND_COOLDOWN = 3.0

# GUI 축소 표시 배율
DISPLAY_SCALE = 0.5

# 예측 간격 (N프레임마다 1회)
PREDICTION_INTERVAL = 2

# 시각화 스타일
COLOR_HAND1 = (0, 0, 255)  # 빨강
COLOR_HAND2 = (0, 0, 255)  # 빨강
RADIUS = 3
THICKNESS = 3
EMA_ALPHA = 0.6

# Response 표시 리셋 시간(초): 이 시간 동안 전환이 없으면 0으로 표시
RESPONSE_RESET_SEC = 3.0

# 명령어 매핑
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


# --- 2. 파일/모델 로딩 (HMM/라벨만) ---
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
        print(f" - Ergodic 모델 로드 완료: {ERGODIC_MODEL_PATH}")

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
        print("모든 모델 로딩 완료.")
        return gesture_hmms, ergodic_model, gesture_names
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None, None, None


# --- 2-1. TFLite 인코더 로딩/호출 ---
def load_tflite_encoder(model_path: str):
    interp = Interpreter(model_path=model_path, num_threads=os.cpu_count())
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f" - TFLite encoder loaded: {model_path}")
    print(f"   input: shape={in_det['shape']} dtype={in_det['dtype']}")
    print(f"   output: shape={out_det['shape']} dtype={out_det['dtype']}")
    return interp, in_det, out_det


# 전역 인터프리터 (한 번만 초기화)
TFLITE_INTERP = None
TFLITE_IN = None
TFLITE_OUT = None


def encode_with_tflite(sequence_data: np.ndarray) -> np.ndarray:
    """
    sequence_data: shape (WINDOW_SIZE, input_dim)
    return latent_seq: shape (T, latent_dim)
    """
    global TFLITE_INTERP, TFLITE_IN, TFLITE_OUT
    x = sequence_data[np.newaxis, ...]
    # 입력 dtype 맞추기
    if x.dtype != TFLITE_IN["dtype"]:
        x = x.astype(TFLITE_IN["dtype"], copy=False)
    TFLITE_INTERP.set_tensor(TFLITE_IN["index"], x)
    TFLITE_INTERP.invoke()
    latent = TFLITE_INTERP.get_tensor(TFLITE_OUT["index"])[0]
    return latent


# --- 3. PPT 관련 ---
def open_presentation(filepath):
    if not os.path.exists(filepath):
        print(f"오류: PPT 파일 없음: {filepath}")
        return False
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        else:
            subprocess.Popen(["open", filepath])
        return True
    except Exception as e:
        print(f"오류: PPT 열기 실패: {e}")
        return False


def get_ppt_window():
    try:
        import pygetwindow as gw

        ppt_windows = [
            win for win in gw.getAllWindows() if "powerpoint" in win.title.lower()
        ]
        if ppt_windows:
            return ppt_windows[0]
    except Exception:
        pass
    return None


def execute_ppt_command(gesture_label, window_name):
    """
    PPT 창을 활성화하고 해당 키 입력을 전송한다.
    반환값: exec_ms (창 활성화 + 키 입력 전송에 걸린 시간, ms)
    실패 시 None
    """
    command_name = COMMAND_MAP.get(gesture_label)
    if not command_name:
        return None
    print(f"명령 실행 시도: 제스처 '{gesture_label}' -> {command_name}")
    ppt_window = get_ppt_window()
    if not ppt_window:
        print(" -> 경고: PowerPoint 창을 찾을 수 없습니다.")
        return None
    try:
        t0 = time.time()
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
        time.sleep(0.1)
        if getattr(ppt_window, "isMinimized", False):
            ppt_window.restore()
        ppt_window.activate()
        time.sleep(0.2)
        print(f" -> '{command_name}' 키 입력 전송...")
        if gesture_label == "1":
            pyautogui.hotkey("win", "d")
        elif gesture_label == "2":
            pyautogui.press("f5")
        elif gesture_label == "3":
            pyautogui.press("left")
        elif gesture_label == "4":
            pyautogui.press("right")
        elif gesture_label == "5":
            pyautogui.press("esc")
        elif gesture_label == "6":
            pyautogui.press("home")
        elif gesture_label == "7":
            pyautogui.press("w")
        elif gesture_label == "8":
            pyautogui.press("end")
        elif gesture_label == "9":
            pyautogui.press("b")
        elif gesture_label == "10":
            pyautogui.hotkey("alt", "f4")
        t1 = time.time()
        exec_ms = (t1 - t0) * 1000.0
        print(" -> 키 입력 전송 완료.")
        return exec_ms
    except Exception as e:
        print(f" -> 명령 실행 중 오류 발생: {e}")
        return None
    finally:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


# --- 4. 제스처 인식 (HMM 스코어링) ---
def recognize_gesture_from_window(sequence_data, hmms, ergodic, gesture_names):
    """
    sequence_data: (WINDOW_SIZE, input_dim)
    return: best gesture name or None
    """
    latent_seq = encode_with_tflite(sequence_data)  # TFLite 인코더 호출
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


# --- 5. 스켈레톤 시각화/보조 ---
def draw_hand(canvas, pts, color, radius=3, thickness=3):
    HCON = list(mp.solutions.hands.HAND_CONNECTIONS)
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    for i, j in HCON:
        x1, y1 = int(pts[i, 0]), int(pts[i, 1])
        x2, y2 = int(pts[j, 0]), int(pts[j, 1])
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def ema_update(prev, cur, alpha=0.7):
    if prev is None:
        return cur.copy()
    return alpha * cur + (1 - alpha) * prev


# --- 6. 메트릭 수집 ---
class Metrics:
    def __init__(self, fps_alpha=0.2, time_alpha=0.2, spot_window=20):
        self.last_frame_ts = None
        self.fps_ema = 0.0
        self.end2end_ms_ema = 0.0
        self.enc_ms_ema = 0.0
        self.hmm_ms_ema = 0.0
        self.response_ms_ema = 0.0  # Response Latency EMA
        self._fps_alpha = fps_alpha
        self._time_alpha = time_alpha
        self.spot_hist = defaultdict(lambda: deque(maxlen=spot_window))
        self.queue_depth = 0

    def update_fps(self, now):
        if self.last_frame_ts is not None:
            dt = now - self.last_frame_ts
            if dt > 0:
                fps = 1.0 / dt
                self.fps_ema = (
                    self._fps_alpha * fps + (1 - self._fps_alpha) * self.fps_ema
                )
        self.last_frame_ts = now

    def update_times(self, end2end_ms=None, enc_ms=None, hmm_ms=None, response_ms=None):
        if end2end_ms is not None:
            self.end2end_ms_ema = (
                self._time_alpha * end2end_ms
                + (1 - self._time_alpha) * self.end2end_ms_ema
            )
        if enc_ms is not None:
            self.enc_ms_ema = (
                self._time_alpha * enc_ms + (1 - self._time_alpha) * self.enc_ms_ema
            )
        if hmm_ms is not None:
            self.hmm_ms_ema = (
                self._time_alpha * hmm_ms + (1 - self._time_alpha) * self.hmm_ms_ema
            )
        if response_ms is not None:
            self.response_ms_ema = (
                self._time_alpha * response_ms
                + (1 - self._time_alpha) * self.response_ms_ema
            )

    def add_spot(self, label, spot_ms):
        if label:
            self.spot_hist[label].append(spot_ms)

    def avg_spot_ms(self, label):
        q = self.spot_hist.get(label)
        if not q or len(q) == 0:
            return None
        return float(np.mean(q))


# --- 7. 비동기 추론 워커 스레드 ---
def infer_worker(
    infer_q: "queue.Queue[tuple]",
    result_q: "queue.Queue[tuple]",
    hmms,
    ergodic,
    gesture_names,
):
    """
    infer_q item: (seq_id:int, enqueue_ts:float, sequence_data:np.ndarray)
    result_q item: (seq_id:int, gesture:str|None, enc_ms:float, hmm_ms:float, end2end_ms:float)
    """
    while True:
        item = infer_q.get()
        if item is None:
            break  # 종료 신호
        try:
            seq_id, t_enq, seq = item
            t0 = time.time()
            latent_seq = encode_with_tflite(seq)
            t1 = time.time()
            final_model_score = ergodic.score(latent_seq)
            best_gesture, max_diff = None, -np.inf
            for g_name in gesture_names:
                g_score = hmms[g_name].score(latent_seq)
                diff = g_score - final_model_score
                if diff > max_diff:
                    max_diff, best_gesture = diff, g_name
            gesture = best_gesture if (max_diff >= THRESHOLD_DIFF) else None
            t2 = time.time()

            enc_ms = (t1 - t0) * 1000.0
            hmm_ms = (t2 - t1) * 1000.0
            end2end_ms = (t2 - t_enq) * 1000.0
            result_q.put((seq_id, gesture, enc_ms, hmm_ms, end2end_ms))
        except Exception as e:
            print(f"[infer_worker] 오류: {e}")


# --- 8. 스펙 출력 ---
def print_specs(cap, gesture_names):
    import tensorflow as tf
    import mediapipe as mp_local

    print("\n===== 장비/환경 스펙 =====")
    try:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS)
    except Exception:
        W = H = FPS = None
    print(f"OS           : {platform.platform()}")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"CPU Cores    : {os.cpu_count()}")
    print(f"OpenCV       : {cv2.__version__}")
    print(f"TensorFlow   : {tf.__version__}")
    print(
        f"Mediapipe    : {mp_local.__version__ if hasattr(mp_local,'__version__') else 'unknown'}"
    )
    print(f"TFLite Thrds : {os.cpu_count()} (Interpreter)")
    print(
        f"Camera Res   : {W}x{H} @ {FPS:.1f} fps"
        if W and H and FPS
        else "Camera Res   : (unknown)"
    )
    print(f"WINDOW_SIZE  : {WINDOW_SIZE}, PRED_INTERVAL: {PREDICTION_INTERVAL}")
    print(f"THRESH_DIFF  : {THRESHOLD_DIFF}, CMD_COOLDOWN: {COMMAND_COOLDOWN}s")
    print(f"DISPLAY_SCALE: {DISPLAY_SCALE}")
    print(f"Gestures     : {gesture_names}")
    print("=========================\n")


# --- 9. 메인 ---
def main():
    global TFLITE_INTERP, TFLITE_IN, TFLITE_OUT

    # HMM/라벨 로드
    gesture_hmms, ergodic_model, gesture_names = load_hmms_and_labels()
    if not all([gesture_hmms, ergodic_model, gesture_names]):
        return

    # TFLite 인코더 로드
    try:
        TFLITE_INTERP, TFLITE_IN, TFLITE_OUT = load_tflite_encoder(
            str(TFLITE_ENCODER_PATH)
        )
    except Exception as e:
        print(f"오류: TFLite 인코더 로딩 실패: {e}")
        return

    # PPT 열기
    if not open_presentation(PPT_FILENAME):
        print("경고: PPT 파일을 열지 못했습니다. ")
    time.sleep(5)

    # MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,  # 한 손만 사용
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    # 스펙 출력
    print_specs(cap, gesture_names)

    window_name = "PPT Gesture Remote"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    landmark_buffer = deque(maxlen=WINDOW_SIZE)
    last_command_times = {g: 0 for g in gesture_names}

    # UI 상태
    status_text = "No Gesture"
    command_display_end_time = 0
    last_command_text = ""
    last_predicted_gesture = None

    prev_pts1, prev_pts2 = None, None
    frame_counter = 0

    # --- 스레드 큐/워커 준비 ---
    infer_q: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
    result_q: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
    worker = threading.Thread(
        target=infer_worker,
        args=(infer_q, result_q, gesture_hmms, ergodic_model, gesture_names),
        daemon=True,
    )
    worker.start()

    # 메트릭
    metrics = Metrics()
    seq_id = 0  # 추론 요청 식별자

    # Response 지표 최신 갱신 시각(표시용 리셋에 사용)
    last_response_update_ts = 0.0

    print("\nPPT 원격 제어 프로그램을 시작합니다. 'q'를 눌러 종료하세요.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            now = time.time()
            metrics.update_fps(now)

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            canvas = np.zeros_like(frame)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(image_rgb)
            hand_is_present = results.multi_hand_landmarks is not None

            current_landmarks_for_model = np.zeros(2 * 21 * 3, dtype=np.float32)
            if hand_is_present:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    start_idx = i * 63
                    landmarks_flat = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()
                    current_landmarks_for_model[
                        start_idx : start_idx + len(landmarks_flat)
                    ] = landmarks_flat

                    pts_current = np.array(
                        [[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    )
                    if i == 0:
                        prev_pts1 = ema_update(prev_pts1, pts_current, EMA_ALPHA)
                        draw_hand(canvas, prev_pts1, COLOR_HAND1, RADIUS, THICKNESS)
                    elif i == 1:
                        prev_pts2 = ema_update(prev_pts2, pts_current, EMA_ALPHA)
                        draw_hand(canvas, prev_pts2, COLOR_HAND2, RADIUS, THICKNESS)
            else:
                prev_pts1, prev_pts2 = None, None
                last_predicted_gesture = None

            landmark_buffer.append(current_landmarks_for_model)

            # --- 비동기 추론 요청 ---
            if (len(landmark_buffer) == WINDOW_SIZE) and (
                frame_counter % PREDICTION_INTERVAL == 0
            ):
                if not infer_q.full():
                    sequence_to_recognize = np.array(landmark_buffer, dtype=np.float32)
                    seq_id += 1
                    infer_q.put((seq_id, time.time(), sequence_to_recognize))
                metrics.queue_depth = infer_q.qsize()

            # --- 결과 수거 ---
            try:
                rid, predicted, enc_ms, hmm_ms, end2end_ms = result_q.get_nowait()
                # 인식 관련 지표 업데이트
                metrics.update_times(
                    end2end_ms=end2end_ms, enc_ms=enc_ms, hmm_ms=hmm_ms
                )

                if predicted is not None:
                    last_predicted_gesture = predicted
                    metrics.add_spot(predicted, end2end_ms)

                    now2 = time.time()
                    if (
                        now2 - last_command_times[last_predicted_gesture]
                    ) > COMMAND_COOLDOWN:
                        # PPT 명령 실행 및 실행 시간(ms) 측정
                        exec_ms = execute_ppt_command(
                            last_predicted_gesture, window_name
                        )
                        if exec_ms is None:
                            exec_ms = 0.0
                        # Response Latency = End2End(손→인식완료) + exec(창 활성화+키 전송)
                        response_ms = end2end_ms + exec_ms
                        metrics.update_times(response_ms=response_ms)
                        last_response_update_ts = (
                            time.time()
                        )  # 마지막 Response 갱신 시각 기록

                        last_command_times[last_predicted_gesture] = now2
                        last_command_text = f"COMMAND: {COMMAND_MAP.get(last_predicted_gesture, 'Unknown')}"
                        command_display_end_time = now2 + 1.5
            except queue.Empty:
                pass

            # --- UI 텍스트 상태(상단) ---
            current_time = time.time()
            if current_time < command_display_end_time:
                status_text = last_command_text
            elif last_predicted_gesture:
                status_text = (
                    f"Detecting: {COMMAND_MAP.get(last_predicted_gesture, '')}"
                )
            elif hand_is_present:
                status_text = "Detecting"
            else:
                status_text = "No Gesture"

            # 상단 바
            cv2.rectangle(canvas, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            # --- 하단 메트릭 바 (FPS + Response Latency with reset) ---
            bar_h = 60
            cv2.rectangle(canvas, (0, H - bar_h), (W, H), (20, 20, 20), -1)

            # 일정 시간 응답 갱신이 없으면 0으로 표시
            if (time.time() - last_response_update_ts) > RESPONSE_RESET_SEC:
                display_response_ms = 0.0
            else:
                display_response_ms = metrics.response_ms_ema

            fps_txt = f"FPS: {metrics.fps_ema:4.1f}"
            resp_txt = f"Response: {display_response_ms:5.1f} ms"
            line_txt = f"{fps_txt} | {resp_txt}"

            cv2.putText(
                canvas,
                line_txt,
                (10, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # === 표시 축소 ===
            if DISPLAY_SCALE != 1.0:
                disp = cv2.resize(canvas, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.imshow(window_name, disp)
            else:
                cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 종료 처리
        cap.release()
        cv2.destroyAllWindows()
        mp_hands.close()
        # 워커 종료 신호
        try:
            infer_q.put_nowait(None)
        except Exception:
            pass
        worker.join(timeout=2.0)
        print("프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
