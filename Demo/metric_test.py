# main_hand_window_trials.py
# [최종 수정] Script 3의 비동기(스레드) 추론 엔진을 적용하여 인식 성능 확보
# + 기존 Script 1의 "최빈값" 평가 로직 유지
# python main10.py --gesture 5 --trials 40 --outfile plus_trials_v2.xlsx
import os, json, time, math, argparse, threading, queue  # <--- [수정] threading, queue 추가
from pathlib import Path
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd

# ===============================
# 설정 (필요시 조정)
# ===============================
ARTIFACTS_DIR = Path("./artifacts")
HMM_DIR = ARTIFACTS_DIR / "hmms"
ERGODIC_MODEL_PATH = HMM_DIR / "ergodic.pkl"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"
TFLITE_ENCODER_PATH = ARTIFACTS_DIR / "encoder_model_fp16_select.tflite"

WINDOW_SIZE = 82
THRESHOLD_DIFF = 0.0
PREDICTION_INTERVAL = 3  # (Script 3와 동일하게 3)
PROC_SCALE = 0.6
CAM_W, CAM_H = 640, 480
WINDOW_NAME = "Gesture Trials (Async)"
DISPLAY_SCALE = 1.0

P_ON = 3
P_OFF = 15

COLOR_HAND = (0, 0, 255)
RADIUS = 3
THICK = 2

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

# ===============================
# TFLite 인터프리터
# ===============================
try:
    from tensorflow.lite.python.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter

TFLITE_INTERP = None
TFLITE_IN = None
TFLITE_OUT = None


def load_tflite_encoder(path: str):
    itp = Interpreter(model_path=path, num_threads=os.cpu_count())
    itp.allocate_tensors()
    inde = itp.get_input_details()[0]
    oute = itp.get_output_details()[0]
    print(
        f"[TFLite] loaded {path} | in:{inde['shape']}/{inde['dtype']}  out:{oute['shape']}/{oute['dtype']}"
    )
    return itp, inde, oute


def encode_with_tflite(seq: np.ndarray) -> np.ndarray:
    global TFLITE_INTERP, TFLITE_IN, TFLITE_OUT
    x = seq[np.newaxis, ...]
    if x.dtype != TFLITE_IN["dtype"]:
        x = x.astype(TFLITE_IN["dtype"], copy=False)
    TFLITE_INTERP.set_tensor(TFLITE_IN["index"], x)
    TFLITE_INTERP.invoke()
    return TFLITE_INTERP.get_tensor(TFLITE_OUT["index"])[0]


# ===============================
# 모델 로딩(HMM/라벨)
# ===============================
def load_hmms_and_labels():
    if not ERGODIC_MODEL_PATH.exists():
        print(f"[오류] {ERGODIC_MODEL_PATH} 없음")
        return None, None, None
    if not LABEL_MAP_PATH.exists():
        print(f"[오류] {LABEL_MAP_PATH} 없음")
        return None, None, None
    with open(ERGODIC_MODEL_PATH, "rb") as f:
        ergodic = joblib.load(f)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    gesture_names = list(label_map.keys())
    hmms = {}
    for g in gesture_names:
        p = HMM_DIR / f"hmm_{g}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                hmms[g] = joblib.load(f)
        else:
            print(f"[경고] HMM 누락: {p}")
    print(f"[Model] labels: {gesture_names}  HMMs: {len(hmms)}")
    return hmms, ergodic, gesture_names


# --- [수정] Script 3의 비동기 추론 워커 ---
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

            t0 = time.perf_counter()
            lat = encode_with_tflite(seq)
            t1 = time.perf_counter()

            base = ergodic.score(lat)
            best, diff_max = None, -np.inf
            for g in gesture_names:
                if g not in hmms:
                    continue
                d = hmms[g].score(lat) - base
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


# -------------------------------------------


# ===============================
# 유틸 (EMA, draw_hand, open_camera, append_row)
# ===============================
def draw_hand(frame, pts, color=COLOR_HAND, r=RADIUS, th=THICK):
    HCON = list(mp.solutions.hands.HAND_CONNECTIONS)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
    for i, j in HCON:
        x1, y1 = int(pts[i, 0]), int(pts[i, 1])
        x2, y2 = int(pts[j, 0]), int(pts[j, 1])
        cv2.line(frame, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)


class EMA:
    def __init__(self, a=0.2):
        self.v, self.a, self.init = 0.0, a, False

    def upd(self, x):
        if x is None:
            return self.v
        if not self.init:
            self.v, self.init = x, True
        else:
            self.v = self.a * x + (1 - self.a) * self.v
        return self.v


def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        print("[오류] 카메라 열기 실패")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def append_row(xlsx: Path, row: dict):
    df = pd.DataFrame([row])
    if xlsx.exists():
        old = None
        try:
            old = pd.read_excel(xlsx, sheet_name="trials")
        except Exception:
            pass
        with pd.ExcelWriter(xlsx, engine="openpyxl", mode="w") as w:
            if old is not None:
                pd.concat([old, df], ignore_index=True).to_excel(
                    w, sheet_name="trials", index=False
                )
            else:
                df.to_excel(w, sheet_name="trials", index=False)
    else:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="trials", index=False)


# ===============================
# 메인: 손-윈도우 기반 트라이얼 (비동기 적용)
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gesture", required=True, type=str, help="실험할 제스처 라벨(예:'1'~'10')"
    )
    ap.add_argument("--trials", type=int, default=50, help="총 트라이얼 수")
    ap.add_argument(
        "--outfile", type=str, default="spotting_trials.xlsx", help="결과 xlsx"
    )
    args = ap.parse_args()

    gt = args.gesture
    total_trials = args.trials
    out_path = Path(args.outfile)

    # 모델
    hmms, ergodic, gesture_names = load_hmms_and_labels()
    if not all([hmms, ergodic, gesture_names]):
        return

    global TFLITE_INTERP, TFLITE_IN, TFLITE_OUT
    TFLITE_INTERP, TFLITE_IN, TFLITE_OUT = load_tflite_encoder(str(TFLITE_ENCODER_PATH))

    # MP / Cam
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = open_camera(0)
    if cap is None:
        return
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setUseOptimized(True)

    # --- [수정] 비동기 추론 큐/워커 시작 ---
    infer_q: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
    result_q: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
    worker = threading.Thread(
        target=infer_worker,
        args=(infer_q, result_q, hmms, ergodic, gesture_names),
        daemon=True,
    )
    worker.start()

    # 버퍼/상태
    buf = deque(maxlen=WINDOW_SIZE)
    frame_id = 0
    seq_id = 0  # 비동기 요청 ID

    # 손 구간 상태 (기존 Script 1 로직)
    present_cnt, absent_cnt = 0, 0
    in_segment = False
    seg_t0 = None
    first_pred_time = None
    segment_predictions = []  # 최빈값 계산용
    e2e_in_seg_ema = EMA(0.25)

    # HUD 지표 EMA
    e2e_ema = EMA(0.2)
    resp_ema = EMA(0.2)

    # FPR/FPmin
    fp_count = 0
    non_gesture_opps = 0
    non_gesture_start = time.perf_counter()

    # 진행
    done = 0
    print(
        f"[INFO] Gesture {gt} | 목표 {total_trials}회 | 결과 파일: {out_path.resolve()}"
    )
    print(
        "   [비동기 모드] 손을 넣으면 1회 측정 시작, 손을 빼면 1회 기록됩니다. q: 종료"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            # 다운스케일 및 MP 처리
            if PROC_SCALE != 1.0:
                proc = cv2.resize(
                    frame,
                    None,
                    fx=PROC_SCALE,
                    fy=PROC_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
                rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = mp_hands.process(rgb)
            hand_present = res.multi_hand_landmarks is not None

            # 입력 벡터 생성 및 그리기
            cur = np.zeros(2 * 21 * 3, dtype=np.float32)
            if hand_present:
                hs = res.multi_hand_landmarks[0]
                flat = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hs.landmark], dtype=np.float32
                ).flatten()
                cur[: len(flat)] = flat
                pts = np.array(
                    [[lm.x * W, lm.y * H] for lm in hs.landmark], dtype=np.float32
                )
                draw_hand(frame, pts)
            buf.append(cur)

            # 히스테리시스 (Script 1)
            if hand_present:
                present_cnt += 1
                absent_cnt = 0
            else:
                absent_cnt += 1
                present_cnt = 0

            # --- [수정] 추론 요청 (비동기, Script 3 방식) ---
            if (len(buf) == WINDOW_SIZE) and (frame_id % PREDICTION_INTERVAL == 0):
                if not infer_q.full():
                    seq = np.array(buf, dtype=np.float32)
                    seq_id += 1
                    infer_q.put((seq_id, time.perf_counter(), seq))

                # 비제스처 구간 기회 카운트 (Script 1)
                # (비동기이므로 추론 '요청' 시점을 기준으로 함)
                if not in_segment:
                    non_gesture_opps += 1

            # --- [수정] 결과 수거 (비동기, Script 3 방식) ---
            got_result = False
            pred_result = None

            while True:
                try:
                    rid, pred, enc_ms, hmm_ms, end2end_ms = result_q.get_nowait()

                    e2e_ema.upd(end2end_ms)  # E2E (전체)
                    if in_segment:
                        e2e_in_seg_ema.upd(end2end_ms)  # E2E (세그먼트 내)

                    # FPR (Script 1)
                    if (not in_segment) and (pred is not None):
                        fp_count += 1

                    # 세그먼트 내 예측 수집 (Script 1)
                    if in_segment and (pred is not None):
                        if first_pred_time is None:
                            # [수정] 결과가 '도착한' 시간을 기준으로 ART 측정
                            first_pred_time = time.perf_counter()
                        segment_predictions.append(pred)

                except queue.Empty:
                    break  # 큐가 비었으면 루프 탈출

            # --- [기존] Script 1 세그먼트 시작/종료 로직 ---

            # 세그먼트 시작
            if (not in_segment) and (present_cnt >= P_ON):
                in_segment = True
                seg_t0 = time.perf_counter()  # 손 등장 시간
                first_pred_time = None
                segment_predictions = []
                e2e_in_seg_ema = EMA(0.25)

            # 세그먼트 종료 → 기록
            if in_segment and (absent_cnt >= P_OFF):
                in_segment = False

                resp_ms = 0.0
                art_ms = 0.0

                # 1. 응답시간 (ART) 계산: 손 등장 ~ '첫' 예측
                if first_pred_time is not None:
                    resp_ms = (first_pred_time - seg_t0) * 1000.0
                    resp_ema.upd(resp_ms)
                    art_ms = resp_ms

                # 2. 최종 예측 결정: 구간 내 '최빈값'
                if segment_predictions:
                    pred_counts = Counter(segment_predictions)
                    pred_label = pred_counts.most_common(1)[0][0]
                    correct = "Y" if pred_label == gt else "N"
                    err = "ok" if correct == "Y" else f"substitution(pred={pred_label})"
                else:
                    pred_label = ""
                    correct = "N"
                    err = "timeout/deletion"

                # 3. FPR/FPmin
                non_gesture_sec = time.perf_counter() - non_gesture_start
                FPR = (
                    (100.0 * fp_count / non_gesture_opps)
                    if non_gesture_opps > 0
                    else 0.0
                )
                FPmin = (
                    (fp_count / (non_gesture_sec / 60.0))
                    if non_gesture_sec > 1e-6
                    else 0.0
                )

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "gesture": gt,
                    "trial_idx": done + 1,
                    "predicted": pred_label,
                    "correct": correct,
                    "error_type": err,
                    "E2E_ms": round(e2e_in_seg_ema.v, 2),  # 구간 내 평균 E2E
                    "Response_ms": round(resp_ms, 2),  # ART
                    "ART_ms": round(art_ms, 2),  # ART
                    "FPR_percent": round(FPR, 2),
                    "FP_per_min": round(FPmin, 2),
                }
                append_row(out_path, row)
                done += 1
                print(
                    f"[TRIAL {done}/{total_trials}] pred={pred_label} (n={len(segment_predictions)}) correct={correct} resp={resp_ms:.1f}ms  E2E~{e2e_in_seg_ema.v:.1f}ms"
                )
                if done >= total_trials:
                    break

            # HUD (기존 Script 1)
            non_gesture_sec = max(1e-6, time.perf_counter() - non_gesture_start)
            FPR_hud = 100.0 * fp_count / max(1, non_gesture_opps)
            FPmin_hud = fp_count / (non_gesture_sec / 60.0)

            cv2.rectangle(frame, (0, 0), (W, 80), (0, 0, 0), -1)
            line1 = f"E2E(Sys): {e2e_ema.v:5.1f} ms | FPR: {FPR_hud:4.1f}% | FP/min: {FPmin_hud:6.2f}"
            line2 = (
                f"Response(ART): {resp_ema.v:5.1f} ms | Trials: {done}/{total_trials}"
            )
            cv2.putText(
                frame,
                line1,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                line2,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (255, 255, 255),
                2,
            )

            if DISPLAY_SCALE != 1.0:
                disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.imshow(WINDOW_NAME, disp)
            else:
                cv2.imshow(WINDOW_NAME, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        # --- [수정] 워커 스레드 종료 ---
        try:
            infer_q.put_nowait(None)
        except Exception:
            pass
        try:
            worker.join(timeout=1.0)
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        mp_hands.close()
        print(f"[DONE] 저장 위치: {out_path.resolve()}")


if __name__ == "__main__":
    main()
