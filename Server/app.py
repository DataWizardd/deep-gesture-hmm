from fastapi import (
    FastAPI,
    Request,
    Depends,
    HTTPException,
    Cookie,
    Response,
    WebSocket,
    WebSocketDisconnect,
    Path,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import requests
import jwt
from datetime import datetime, timedelta
import cv2
import mediapipe as mp
import numpy as np
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import tensorflow as tf
import time
import pandas as pd
import csv
from typing import List
import uvicorn
import pathlib

app_encoder_model = None
app_gesture_hmms = None
app_ergodic_model = None
app_gesture_labels = []
app_MAX_SEQ_LEN = 0
app_trim_zero_padding = None
app_sliding_window_gesture_detection = None
models_successfully_loaded = False


GESTURE_WINDOW_SIZE = 20
GESTURE_STEP = 2
GESTURE_THRESHOLD_DIFF = 0.0
GESTURE_MIN_MERGE_GAP = 5
GESTURE_MIN_INTERVAL_LENGTH = 5

try:
    import utils

    if utils.load_all_models():

        app_encoder_model = utils.encoder_model
        app_gesture_hmms = utils.gesture_hmms
        app_ergodic_model = utils.ergodic_model
        app_gesture_labels = (
            utils.gesture_labels if utils.gesture_labels is not None else []
        )
        app_MAX_SEQ_LEN = utils.MAX_SEQ_LEN
        app_trim_zero_padding = utils.trim_zero_padding
        app_sliding_window_gesture_detection = utils.sliding_window_gesture_detection

        models_successfully_loaded = True
        print("Models loaded successfully via utils.py and assigned to app variables.")
        if utils.gesture_labels is None:
            print(
                "Warning: utils.gesture_labels was None after model loading; app_gesture_labels defaulted to an empty list."
            )
    else:
        models_successfully_loaded = False
        print(
            "CRITICAL: utils.load_all_models() reported failure. Gesture detection will not work."
        )

except ImportError as e:
    models_successfully_loaded = False
    print(
        f"Error importing from utils.py: {e}. Please ensure utils.py is in the correct path and has necessary functions/models."
    )

except AttributeError as e:
    models_successfully_loaded = False
    print(
        f"Error accessing attributes from utils.py: {e}. Ensure utils.py has the necessary definitions (e.g., load_all_models, model variables)."
    )


load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 현재 파일의 디렉토리 경로
BASE_DIR = pathlib.Path(__file__).parent.parent


app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "front" / "static")), name="static"
)
templates = Jinja2Templates(directory=str(BASE_DIR / "front" / "templates"))

# Kakao Login, JWT
KAKAO_CLIENT_ID = os.getenv("KAKAO_CLIENT_ID")
KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-development")


# --- JWT ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "KAKAO_CLIENT_ID": KAKAO_CLIENT_ID,
            "KAKAO_REDIRECT_URI": KAKAO_REDIRECT_URI,
        },
    )


@app.get("/auth/kakao/callback")
async def kakao_callback(
    code: str,
):
    token_url = "https://kauth.kakao.com/oauth/token"
    token_data = {
        "grant_type": "authorization_code",
        "client_id": KAKAO_CLIENT_ID,
        "redirect_uri": KAKAO_REDIRECT_URI,
        "code": code,
    }
    try:
        token_response_post = requests.post(token_url, data=token_data)
        token_response_post.raise_for_status()
        token_json = token_response_post.json()

        if "access_token" not in token_json:
            print(
                f"Kakao token error: {token_json.get('error_description', 'No access token')}"
            )
            raise HTTPException(
                status_code=400, detail="Failed to get Kakao access token."
            )

        user_url = "https://kapi.kakao.com/v2/user/me"
        headers = {"Authorization": f"Bearer {token_json['access_token']}"}
        user_response_get = requests.get(user_url, headers=headers)
        user_response_get.raise_for_status()
        user_json = user_response_get.json()

        jwt_access_token = create_access_token(data={"sub": str(user_json["id"])})

        response = RedirectResponse(url="/main", status_code=303)
        response.set_cookie(
            key="access_token", value=jwt_access_token, httponly=True, samesite="Lax"
        )
        return response

    except requests.exceptions.RequestException as e:
        print(f"Kakao API request failed: {e}")
        raise HTTPException(
            status_code=502, detail=f"Failed to communicate with Kakao: {e}"
        )
    except Exception as e:
        print(f"Error in kakao_callback: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during Kakao callback: {e}"
        )


async def get_current_user(access_token: str = Cookie(None)):
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = verify_token(access_token)
        return payload
    except HTTPException as e:

        raise e


@app.get("/main", response_class=HTMLResponse)
async def main_page_route(
    request: Request, current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=307)
    return templates.TemplateResponse(
        "main.html", {"request": request, "user_id": current_user.get("sub")}
    )


@app.get("/speak-to-text", response_class=HTMLResponse)
async def speak_to_text_page(
    request: Request, current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=307)
    return templates.TemplateResponse("speak_to_text.html", {"request": request})


@app.get("/phrases", response_class=HTMLResponse)
async def phrases_page(
    request: Request, current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=307)
    return templates.TemplateResponse("phrases.html", {"request": request})


@app.get("/mypage", response_class=HTMLResponse)
async def mypage_route(
    request: Request, current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=307)

    return templates.TemplateResponse(
        "mypage.html", {"request": request, "user_id": current_user.get("sub")}
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response


class CollectRequest(BaseModel):
    filename_base: str = Field(..., example="my_gesture_recording")


class Interval(BaseModel):
    start: int
    end: int
    label: str


class GestureResponse(BaseModel):
    file_saved_at: str | None = None
    intervals: List[Interval]
    message: str | None = None


class TranslatedGestureResponse(BaseModel):
    translated_gestures: List[str]


@app.post("/collect_live", response_model=TranslatedGestureResponse)
async def collect_live_data_and_detect_gestures(req: CollectRequest):
    if not models_successfully_loaded or not all(
        [
            app_encoder_model,
            app_gesture_hmms,
            app_ergodic_model,
            app_sliding_window_gesture_detection,
            app_trim_zero_padding,
            app_gesture_labels is not None,
        ]
    ):

        error_detail_str = "Models not loaded or core gesture functions missing. Cannot process request."

        raise HTTPException(
            status_code=503,
            detail=error_detail_str,
        )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, "data_project", "gsn")
    os.makedirs(output_dir, exist_ok=True)

    safe_filename_base = "".join(
        c for c in req.filename_base if c.isalnum() or c in ("_", "-")
    )
    if not safe_filename_base:
        safe_filename_base = "default_collection"

    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{safe_filename_base}_{timestamp_suffix}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # Mediapipe setup
    mp_hands_local = mp.solutions.hands
    hands_processor = mp_hands_local.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hands_processor.close()
        raise HTTPException(status_code=500, detail="Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(
        f"Starting data collection. Press 'q' in OpenCV window to stop. Saving to {csv_path}"
        f"Note: The sliding window detection requires at least {GESTURE_WINDOW_SIZE} valid frames after pre-processing."
    )

    gesture_data_buffer = []
    header = ["timestamp", "hand_detected"]

    for hand_idx in range(2):  # Max 2 hands
        for lm_idx in range(21):  # 21 landmarks
            header.extend(
                [
                    f"h{hand_idx}_lm{lm_idx}_x",
                    f"h{hand_idx}_lm{lm_idx}_y",
                    f"h{hand_idx}_lm{lm_idx}_z",
                ]
            )
    gesture_data_buffer.append(header)

    try:
        collection_duration_seconds = 300
        start_time = time.time()
        recording_active = True
        final_message_on_frame = ""
        frame_count_for_rec_blink = 0

        def draw_text_with_bg(
            img,
            text,
            origin,
            font_face,
            font_scale,
            text_color,
            bg_color,
            thickness=1,
            line_type=cv2.LINE_AA,
            alpha=0.5,
        ):
            text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_w, text_h = text_size
            x, y = origin
            bg_rect = img[y - text_h - 5 : y + 5, x - 5 : x + text_w + 5].copy()
            cv2.rectangle(
                img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1
            )
            img[y - text_h - 5 : y + 5, x - 5 : x + text_w + 5] = cv2.addWeighted(
                bg_rect,
                alpha,
                img[y - text_h - 5 : y + 5, x - 5 : x + text_w + 5],
                1 - alpha,
                0,
            )
            cv2.putText(
                img,
                text,
                (x, y),
                font_face,
                font_scale,
                text_color,
                thickness,
                line_type,
            )

        while recording_active:
            elapsed_time = time.time() - start_time

            if elapsed_time >= collection_duration_seconds:
                recording_active = False
                final_message_on_frame = "Max collection time reached."

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not retrieve frame.")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands_processor.process(image_rgb)
            image_rgb.flags.writeable = True

            current_ts = time.time()
            hand_present_flag = 1 if results.multi_hand_landmarks else 0
            row_data = [current_ts, hand_present_flag]

            landmarks_flat = [0.0] * (2 * 21 * 3)

            # UI
            frame_height, frame_width = frame.shape[:2]

            if recording_active:
                frame_count_for_rec_blink = (frame_count_for_rec_blink + 1) % 20
                rec_dot_color = (
                    (0, 0, 255) if frame_count_for_rec_blink < 10 else (50, 50, 150)
                )
                cv2.circle(frame, (frame_width - 125, 25), 8, rec_dot_color, -1)
                draw_text_with_bg(
                    frame,
                    "REC",
                    (frame_width - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    (0, 0, 0),
                )

                elapsed_timer_text = f"Elapsed: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
                draw_text_with_bg(
                    frame,
                    elapsed_timer_text,
                    (frame_width - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    (0, 0, 0),
                )

            if results.multi_hand_landmarks:
                status_text = "Hand(s) Detected"
                status_color = (0, 255, 0)
                for hand_no, hand_lms in enumerate(results.multi_hand_landmarks):
                    if hand_no < 2:
                        for lm_idx, landmark in enumerate(hand_lms.landmark):
                            base_idx = hand_no * (21 * 3) + lm_idx * 3
                            landmarks_flat[base_idx] = landmark.x
                            landmarks_flat[base_idx + 1] = landmark.y
                            landmarks_flat[base_idx + 2] = landmark.z
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_lms, mp_hands_local.HAND_CONNECTIONS
                    )
            else:
                status_text = "No Hand Detected"
                status_color = (0, 0, 255)
            draw_text_with_bg(
                frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                (0, 0, 0),
            )

            # Stop
            draw_text_with_bg(
                frame,
                "Press 'q' to STOP",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                (0, 0, 0),
            )

            row_data.extend(landmarks_flat)
            # writer.writerow
            gesture_data_buffer.append(row_data)

            if final_message_on_frame:
                text_size = cv2.getTextSize(
                    final_message_on_frame, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )[0]
                text_x = (frame_width - text_size[0]) // 2
                text_y = (frame_height + text_size[1]) // 2

                cv2.rectangle(
                    frame,
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    (0, 0, 50),
                    -1,
                )
                cv2.putText(
                    frame,
                    final_message_on_frame,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Medical Gesture Translation", frame)
                cv2.waitKey(2000)
                break

            cv2.imshow("Medical Gesture Translation", frame)
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord("q"):
                recording_active = False
                final_message_on_frame = "Start Translation"

        if final_message_on_frame:
            if "frame" not in locals() or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_height, frame_width = frame.shape[:2]
            else:
                frame_height, frame_width = frame.shape[:2]

            text_size = cv2.getTextSize(
                final_message_on_frame, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            cv2.rectangle(
                frame,
                (text_x - 10, text_y - text_size[1] - 10),
                (text_x + text_size[0] + 10, text_y + 10),
                (0, 0, 50),
                -1,
            )
            cv2.putText(
                frame,
                final_message_on_frame,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Medical Gesture Translation", frame)
            cv2.waitKey(2000)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_processor.close()

    print(
        f"Data collection finished. Data captured in memory. Original intended path for debug: {csv_path}"
    )

    try:
        response_data = await asyncio.to_thread(
            _process_csv_and_detect_gestures_sync,
            gesture_data_buffer,
            app_trim_zero_padding,
            app_sliding_window_gesture_detection,
            app_encoder_model,
            app_gesture_hmms,
            app_ergodic_model,
        )
        return response_data
    except Exception as e:
        print(f"Error during threaded gesture detection (using in-memory data): {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during gesture detection (in-memory): {str(e)}",
        )


def _process_csv_and_detect_gestures_sync(
    gesture_data: list,
    trim_zero_padding_func,
    sliding_window_detection_func,
    encoder_model_obj,
    gesture_hmms_obj,
    ergodic_model_obj,
):
    try:

        if not gesture_data or len(gesture_data) < 2:
            print(f"No data collected or data buffer is empty/contains only header.")
            return TranslatedGestureResponse(translated_gestures=[])

        header = gesture_data[0]
        data_rows = gesture_data[1:]

        if not data_rows:
            print(f"No actual data rows found in the buffer after header.")
            return TranslatedGestureResponse(translated_gestures=[])

        df = pd.DataFrame(data_rows, columns=header)

        if df.empty:
            print(f"DataFrame created from buffer is empty.")
            return TranslatedGestureResponse(translated_gestures=[])

        if df.shape[1] < 3:
            print(
                f"DataFrame from buffer does not contain enough columns for landmark data."
            )
            return TranslatedGestureResponse(translated_gestures=[])

        landmark_data = df.iloc[:, 2:].values.astype(np.float32)
        processed_sequence = trim_zero_padding_func(landmark_data)

        if (
            processed_sequence.ndim == 0
            or processed_sequence.size == 0
            or processed_sequence.shape[0] < GESTURE_WINDOW_SIZE
        ):
            detailed_message = (
                f"Not enough data after pre-processing for gesture detection. "
                f"The sliding window approach requires at least {GESTURE_WINDOW_SIZE} valid frames for a window, "
                f"but only {processed_sequence.shape[0]} were available after pre-processing. "
                f"Possible reasons: 1) Recording was too short. 2) Hands were not consistently detected. "
                f"3) 'trim_zero_padding' removed too many frames."
            )
            print(f"Gesture Detection Failure (in-memory data): {detailed_message}")

            return TranslatedGestureResponse(translated_gestures=[])

        detected_intervals_tuples = sliding_window_detection_func(
            continuous_sequence=processed_sequence,
            current_encoder_model=encoder_model_obj,
            current_gesture_hmms=gesture_hmms_obj,
            current_final_model=ergodic_model_obj,
            window_size=GESTURE_WINDOW_SIZE,
            step=GESTURE_STEP,
            threshold_diff=GESTURE_THRESHOLD_DIFF,
            min_merge_gap=GESTURE_MIN_MERGE_GAP,
            min_interval_length=GESTURE_MIN_INTERVAL_LENGTH,
        )

        print(f"Raw detected intervals (in-memory data): {detected_intervals_tuples}")

        print("Gesture detection complete (in-memory data).")

        # Format
        formatted_gestures = [
            f"- {lab}({int(s)}~{int(e)})" for s, e, lab in detected_intervals_tuples
        ]

        return TranslatedGestureResponse(translated_gestures=formatted_gestures)

    except FileNotFoundError:

        print(
            f"Unexpected FileNotFoundError in sync helper (should be using in-memory data)."
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error related to file handling (expected in-memory).",
        )
    except pd.errors.EmptyDataError:
        print(f"Pandas EmptyDataError with in-memory data processing.")
        raise ValueError("Pandas encountered an issue with the in-memory data.")
    except ValueError as ve:
        print(f"Error processing in-memory data in sync helper: {str(ve)}")
        raise
    except Exception as e:
        print(f"Generic error during sync gesture detection (in-memory data): {e}")

        raise


if __name__ == "__main__":
    print("Starting uvicorn server for the combined application...")
    if not models_successfully_loaded:
        print(
            "WARNING: Models from utils.py were NOT loaded or failed to initialize correctly. Gesture recognition endpoints might fail."
        )
    else:
        print(
            "Models from utils.py appear to be loaded and initialized. Gesture recognition should be available."
        )

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
