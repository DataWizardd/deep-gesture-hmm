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
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygetwindow as gw

# --- 1. 설정 및 상수 ---
ARTIFACTS_DIR = Path("./artifacts")
HMM_DIR = ARTIFACTS_DIR / "hmms"
ENCODER_PATH = ARTIFACTS_DIR / "encoder_model.keras"
ERGODIC_MODEL_PATH = HMM_DIR / "ergodic.pkl"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"

# PPT 파일명
PPT_FILENAME = "CM20315_12_Transformers.pptx"

# 파라미터
WINDOW_SIZE = 82
THRESHOLD_DIFF = 0.0
COMMAND_COOLDOWN = 5.0

# 예측 간격 (3프레임에 1번 예측)
PREDICTION_INTERVAL = 3

# 시각화 스타일
COLOR_HAND1 = (0, 0, 255)  # 빨강
COLOR_HAND2 = (0, 0, 255)  # 빨강
RADIUS = 3
THICKNESS = 3
EMA_ALPHA = 0.7

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


# --- 2. 모델 및 파일 관련 함수---
def load_models_from_artifacts():
    print("모델 로딩 중 (artifacts 구조)...")
    if not ARTIFACTS_DIR.exists():
        return None, None, None, None
    try:
        encoder_model = load_model(
            str(ENCODER_PATH),
            custom_objects={"Orthogonal": tf.keras.initializers.Orthogonal},
        )
        print(f" - 인코더 로드 완료: {ENCODER_PATH}")
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
        print("\n모든 모델 로딩 완료.")
        return encoder_model, gesture_hmms, ergodic_model, gesture_names
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None, None, None, None


def open_presentation(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        else:
            subprocess.Popen(["open", filepath])
        return True
    except Exception as e:
        return False


# --- 3. PPT 제어 및 제스처 인식 함수 ---
def get_ppt_window():
    try:
        ppt_windows = [
            win for win in gw.getAllWindows() if "powerpoint" in win.title.lower()
        ]
        if ppt_windows:
            return ppt_windows[0]
    except Exception:
        pass
    return None


def execute_ppt_command(gesture_label, window_name):
    command_name = COMMAND_MAP.get(gesture_label)
    if not command_name:
        return
    print(f"명령 실행 시도: 제스처 '{gesture_label}' -> {command_name}")
    ppt_window = get_ppt_window()
    if not ppt_window:
        print(" -> 경고: PowerPoint 창을 찾을 수 없습니다.")
        return
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
        time.sleep(0.1)
        if ppt_window.isMinimized:
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
        print(" -> 키 입력 전송 완료.")
    except Exception as e:
        print(f" -> 명령 실행 중 오류 발생: {e}")
    finally:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def recognize_gesture_from_window(sequence_data, encoder, hmms, ergodic, gesture_names):
    latent_seq = encoder.predict(sequence_data[np.newaxis, ...], verbose=0)[0]
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


# --- 4. 스켈레톤 시각화 함수  ---
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


# --- 5. 메인 프로그램  ---
def main():
    encoder_model, gesture_hmms, ergodic_model, gesture_names = (
        load_models_from_artifacts()
    )
    if not all([encoder_model, gesture_hmms, ergodic_model, gesture_names]):
        return

    if not open_presentation(PPT_FILENAME):
        return
    time.sleep(5)

    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    window_name = "PPT Gesture Remote (Skeleton)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    landmark_buffer = deque(maxlen=WINDOW_SIZE)
    last_command_times = {g: 0 for g in gesture_names}

    # UI 상태 표시를 위한 변수
    status_text = "No Gesture"
    command_display_end_time = 0
    last_command_text = ""
    last_predicted_gesture = None

    prev_pts1, prev_pts2 = None, None
    frame_counter = 0

    print("\nPPT 원격 제어 프로그램을 시작합니다. 'q'를 눌러 종료하세요.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        canvas = np.zeros_like(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)
        hand_is_present = results.multi_hand_landmarks is not None

        current_landmarks_for_model = np.zeros(2 * 21 * 3)
        if hand_is_present:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                start_idx = i * 63
                landmarks_flat = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                ).flatten()
                current_landmarks_for_model[
                    start_idx : start_idx + len(landmarks_flat)
                ] = landmarks_flat
                pts_current = np.array(
                    [[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark]
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

        # 예측은 예측 인터벌마다 수행
        if (
            len(landmark_buffer) == WINDOW_SIZE
            and frame_counter % PREDICTION_INTERVAL == 0
        ):
            sequence_to_recognize = np.array(landmark_buffer)
            last_predicted_gesture = recognize_gesture_from_window(
                sequence_to_recognize,
                encoder_model,
                gesture_hmms,
                ergodic_model,
                gesture_names,
            )

            current_time = time.time()
            if (
                last_predicted_gesture
                and (current_time - last_command_times[last_predicted_gesture])
                > COMMAND_COOLDOWN
            ):
                execute_ppt_command(last_predicted_gesture, window_name)
                last_command_times[last_predicted_gesture] = current_time

                last_command_text = (
                    f"COMMAND: {COMMAND_MAP.get(last_predicted_gesture, 'Unknown')}"
                )
                command_display_end_time = current_time + 1.5

        # UI 텍스트 상태 관리 로직
        current_time = time.time()

        # 1) 최근 명령 표시 중이면 그대로 유지
        if current_time < command_display_end_time:
            status_text = last_command_text

        # 2) 예측 성공(손 유무 상관없이)
        elif last_predicted_gesture:
            status_text = f"Detecting: {COMMAND_MAP.get(last_predicted_gesture, '')}"

        # 3) 손 감지됨
        elif hand_is_present:
            status_text = "Detecting"

        # 4) 그 외(초기 또는 손 없음)
        else:
            status_text = ""

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
        cv2.imshow(window_name, canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()
    print("프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
