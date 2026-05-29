import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

# ─── PATHS ───────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_model_36.h5"
LABELS_PATH = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels.json"

# ─── SETTINGS ────────────────────────────────────────────
CONFIDENCE  = 0.5
SMOOTH      = 15

# ─── LOAD ────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

print(f"✅ Model loaded — {len(labels)} classes")

# ─── MEDIAPIPE ───────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─── SMOOTHING ───────────────────────────────────────────
prediction_buffer = deque(maxlen=SMOOTH)

# ─── WEBCAM ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit()

print("✅ Webcam opened — press Q to quit\n")

current_word = "..."
current_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True

        # ─── DRAW LANDMARKS ──────────────────────────────
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
            )

        # ─── EXTRACT KEYPOINTS ───────────────────────────
        keypoints = []
        for hand_idx in range(2):
            if hand_idx < len(result.multi_hand_landmarks):
                lm = result.multi_hand_landmarks[hand_idx]
                for point in lm.landmark:
                    keypoints.extend([point.x, point.y, point.z])
            else:
                keypoints.extend([0.0] * 63)

        # ─── PREDICT ─────────────────────────────────────
        inp        = np.array(keypoints).reshape(1, -1)
        preds      = model.predict(inp, verbose=0)[0]
        confidence = float(np.max(preds))
        class_idx  = int(np.argmax(preds))

        if confidence >= CONFIDENCE:
            prediction_buffer.append(class_idx)

        if prediction_buffer:
            smoothed_idx  = Counter(prediction_buffer).most_common(1)[0][0]
            current_word  = labels[smoothed_idx].upper()
            current_conf  = confidence

    else:
        prediction_buffer.clear()
        current_word = "No hand detected"
        current_conf = 0.0

    # ─── DISPLAY ─────────────────────────────────────────
    # top bar
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    hand_count   = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    status_text  = f"Hands detected: {hand_count}"
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # class count top right
    cv2.putText(frame, f"36 classes", (w - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # bottom prediction bar
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)

    if hand_detected and current_conf >= CONFIDENCE:
        label_text = f"{current_word}  ({current_conf*100:.1f}%)"
        color      = (0, 255, 0)
    else:
        label_text = current_word
        color      = (0, 165, 255)

    cv2.putText(frame, label_text, (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    cv2.imshow("ASL Detection — 36 Classes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Detection stopped.")