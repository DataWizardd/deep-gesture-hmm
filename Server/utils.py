import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 모델 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.keras")
ERGODIC_PATH = os.path.join(MODEL_DIR, "ergodic_model.pkl")
GESTURE_HMM_PATH = os.path.join(MODEL_DIR, "gesture_hmms.pkl")

# 전역 변수로 모델 및 관련 정보 저장
encoder_model = None
ergodic_model = None
gesture_hmms = None
gesture_labels = []
MAX_SEQ_LEN = 0
NUM_FEATURES = 0


def load_all_models():
    global encoder_model, ergodic_model, gesture_hmms, gesture_labels, MAX_SEQ_LEN, NUM_FEATURES

    models_loaded_successfully = True

    try:
        if os.path.exists(ENCODER_PATH):
            encoder_model = load_model(ENCODER_PATH, compile=False)
            print("Encoder model loaded successfully from utils.py.")
            _, MAX_SEQ_LEN, NUM_FEATURES = encoder_model.input_shape
            print(
                f"Model expects MAX_SEQ_LEN: {MAX_SEQ_LEN}, NUM_FEATURES: {NUM_FEATURES}"
            )
        else:
            print(f"CRITICAL Error: Encoder model not found at {ENCODER_PATH}")
            encoder_model = None
            models_loaded_successfully = False

        if os.path.exists(ERGODIC_PATH):
            with open(ERGODIC_PATH, "rb") as f:
                ergodic_model = pickle.load(f)
            print("Ergodic model loaded successfully from utils.py.")
        else:
            print(f"CRITICAL Error: Ergodic model not found at {ERGODIC_PATH}")
            ergodic_model = None
            models_loaded_successfully = False

        if os.path.exists(GESTURE_HMM_PATH):
            with open(GESTURE_HMM_PATH, "rb") as f:
                gesture_hmms = pickle.load(f)
            if gesture_hmms:
                gesture_labels = list(gesture_hmms.keys())
                print(
                    f"Gesture HMMs loaded successfully from utils.py. Labels: {gesture_labels}"
                )
            else:
                print(
                    f"Warning: Gesture HMMs file loaded but content is empty or invalid from {GESTURE_HMM_PATH}"
                )
                gesture_hmms = None
                gesture_labels = []
                models_loaded_successfully = False
        else:
            print(f"CRITICAL Error: Gesture HMMs not found at {GESTURE_HMM_PATH}")
            gesture_hmms = None
            gesture_labels = []
            models_loaded_successfully = False

    except Exception as e:
        print(f"CRITICAL Error loading models in utils.py: {e}")

        encoder_model = None
        ergodic_model = None
        gesture_hmms = None
        gesture_labels = []
        MAX_SEQ_LEN = 0
        NUM_FEATURES = 0
        models_loaded_successfully = False

    return models_loaded_successfully


# 0-padding 제거
def trim_zero_padding(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim == 0 or sequence.size == 0:  # 빈 배열이거나 스칼라이면 그대로 반환
        return sequence

    if sequence.ndim == 2:
        non_zero_mask = np.any(sequence != 0, axis=1)
        return sequence[non_zero_mask]
    return sequence


# 짧은 구간 필터링
def filter_short_intervals(
    intervals: list[tuple[int, int, str]], min_length: int = 5
) -> list[tuple[int, int, str]]:
    return [(s, e, label) for s, e, label in intervals if (e - s) >= min_length]


# 라벨 동일 + gap 이하인 구간 병합
def merge_gesture_intervals(
    intervals: list[tuple[int, int, str]], min_merge_gap: int = 0
) -> list[tuple[int, int, str]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = []

    cs, ce, cl = sorted_intervals[0]
    for i in range(1, len(sorted_intervals)):
        s, e, label = sorted_intervals[i]
        if label == cl and s <= ce + min_merge_gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce, cl))
            cs, ce, cl = s, e, label
    merged.append((cs, ce, cl))
    return merged


# 슬라이딩 윈도우 기반 제스처 탐지(패딩 적용)
def sliding_window_gesture_detection(
    continuous_sequence: np.ndarray,
    current_encoder_model,
    current_gesture_hmms: dict[str, object],
    current_final_model,
    window_size: int = 20,
    step: int = 2,
    threshold_diff: float = 3.0,
    min_merge_gap: int = 5,
    min_interval_length: int = 5,
) -> list[tuple[int, int, str]]:

    local_max_seq_len = 0
    local_num_features = 0

    if not all([current_encoder_model, current_gesture_hmms, current_final_model]):
        print(
            "Error in sliding_window_gesture_detection: One or more critical models not provided."
        )
        return []

    try:

        if hasattr(current_encoder_model, "input_shape"):
            shape = current_encoder_model.input_shape
            if isinstance(shape, (list, tuple)) and len(shape) == 3:
                _, local_max_seq_len, local_num_features = shape
                if not (
                    isinstance(local_max_seq_len, int)
                    and isinstance(local_num_features, int)
                    and local_max_seq_len > 0
                    and local_num_features > 0
                ):
                    print(
                        f"Warning: Derived model input_shape {shape} might be invalid. Using global MAX_SEQ_LEN/NUM_FEATURES if available."
                    )

                    if MAX_SEQ_LEN > 0 and NUM_FEATURES > 0:
                        local_max_seq_len = MAX_SEQ_LEN
                        local_num_features = NUM_FEATURES
                    else:
                        print(
                            "Error: Cannot determine model's expected input dimensions (MAX_SEQ_LEN, NUM_FEATURES) from passed model or globals."
                        )
                        return []
            else:
                print(
                    f"Warning: Passed encoder_model.input_shape {shape} is not as expected (None, seq_len, features). Using global values."
                )
                if MAX_SEQ_LEN > 0 and NUM_FEATURES > 0:
                    local_max_seq_len = MAX_SEQ_LEN
                    local_num_features = NUM_FEATURES
                else:
                    print(
                        "Error: Global MAX_SEQ_LEN or NUM_FEATURES not set. Cannot proceed."
                    )
                    return []
        else:
            print(
                "Warning: Passed encoder_model has no input_shape. Using global MAX_SEQ_LEN/NUM_FEATURES."
            )
            if MAX_SEQ_LEN > 0 and NUM_FEATURES > 0:
                local_max_seq_len = MAX_SEQ_LEN
                local_num_features = NUM_FEATURES
            else:
                print(
                    "Error: Global MAX_SEQ_LEN or NUM_FEATURES not set. Cannot proceed."
                )
                return []
    except Exception as e_shape:
        print(
            f"Error determining model input shape: {e_shape}. Fallback to global MAX_SEQ_LEN/NUM_FEATURES."
        )
        if MAX_SEQ_LEN > 0 and NUM_FEATURES > 0:
            local_max_seq_len = MAX_SEQ_LEN
            local_num_features = NUM_FEATURES
        else:
            print(
                "Error: Global MAX_SEQ_LEN or NUM_FEATURES not set after shape determination error. Cannot proceed."
            )
            return []

    if not (local_max_seq_len > 0 and local_num_features > 0):
        print(
            "Error: Model expected input dimensions (local_max_seq_len, local_num_features) are invalid. Cannot proceed."
        )
        return []

    T = continuous_sequence.shape[0]
    if T < window_size:
        print(
            f"Info: Continuous sequence length ({T}) is less than window_size ({window_size}). Returning empty."
        )
        return []

    try:
        gesture_names_local = list(current_gesture_hmms.keys())
        if not gesture_names_local:
            print(
                "Error: No gesture names found in current_gesture_hmms. Cannot perform detection."
            )
            return []
    except Exception as e_keys:
        print(f"Error getting gesture names from HMMs: {e_keys}")
        return []

    detected = []
    n_windows = (T - window_size) // step + 1

    for w in range(n_windows):
        start = w * step
        end = start + window_size
        window = continuous_sequence[start:end, :]

        current_window_actual_len = window.shape[0]
        num_actual_features = window.shape[1]

        if num_actual_features != local_num_features:
            print(
                f"Error: Feature mismatch in window {w}. Expected {local_num_features}, got {num_actual_features}. Skipping."
            )
            continue

        if current_window_actual_len < local_max_seq_len:
            pad_len = local_max_seq_len - current_window_actual_len
            padded_window = np.vstack(
                [window, np.zeros((pad_len, num_actual_features), dtype=window.dtype)]
            )
        elif current_window_actual_len > local_max_seq_len:
            padded_window = window[:local_max_seq_len, :]
        else:
            padded_window = window

        try:
            latent_seq = current_encoder_model.predict(
                padded_window[np.newaxis, ...], verbose=0
            )[0]

            if latent_seq.ndim == 0 or latent_seq.size == 0:
                print(
                    f"Warning: Encoder produced empty or invalid latent_seq for window {w}. Skipping."
                )
                continue

            lengths_for_hmm = [latent_seq.shape[0]]

            f_ll = current_final_model.score(latent_seq, lengths_for_hmm)

            current_max_diff = -np.inf
            current_best_label = None
            for name in gesture_names_local:
                hmm_model = current_gesture_hmms.get(name)
                if hmm_model:
                    g_ll = hmm_model.score(latent_seq, lengths_for_hmm)
                    diff = g_ll - f_ll
                    if diff > current_max_diff:
                        current_max_diff = diff
                        current_best_label = name
                else:
                    print(
                        f"Warning: HMM model for gesture '{name}' not found during scoring for window {w}."
                    )

            if current_max_diff >= threshold_diff and current_best_label is not None:
                detected.append((start, end, current_best_label))
        except Exception as e_predict:
            print(
                f"Error during HMM scoring or prediction in window {w} (data shape: {padded_window.shape}, latent_seq shape: {latent_seq.shape if 'latent_seq' in locals() else 'N/A'}): {e_predict}"
            )

            continue

    merged = merge_gesture_intervals(detected, min_merge_gap)
    final = filter_short_intervals(merged, min_interval_length)
    return final
