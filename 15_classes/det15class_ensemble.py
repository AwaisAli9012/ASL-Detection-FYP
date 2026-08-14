import os
import cv2
import json
import pickle
import time
import threading
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from groq import Groq
import pyttsx3
from pathlib import Path

# --- PATHS ---
BASE_DIR    = Path(__file__).resolve().parent
REPO_ROOT   = BASE_DIR.parent
MODELS_DIR  = REPO_ROOT / "Models"

NN_PATH     = str(MODELS_DIR / "keypoint_model_15_v4_ensemble_nn.h5")
RF_PATH     = str(MODELS_DIR / "keypoint_model_15_v4_rf.pkl")
META_PATH   = str(MODELS_DIR / "keypoint_model_15_v4_meta.pkl")
LABELS_PATH = str(MODELS_DIR / "keypoint_labels_15_v4.json")

# --- SETTINGS & TUNING ---
CONFIDENCE    = 0.60   # Lowered from 0.75 for better detection on fist/folded signs like "YES"
SMOOTH        = 15     # Majority voting buffer size
ENSEMBLE_MODE = "stacking"
MIRROR_FRAME  = True   # Set to False if training data was collected without horizontal flipping
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")

# --- COLORS ---
BG_DARK      = (18, 18, 18)
BG_CARD      = (38, 38, 38)
ACCENT_GREEN = (0, 220, 100)
ACCENT_BLUE  = (100, 180, 255)
ACCENT_GOLD  = (0, 200, 255)
TEXT_WHITE   = (240, 240, 240)
TEXT_GRAY    = (150, 150, 150)
TEXT_DIM     = (80, 80, 80)
BORDER       = (60, 60, 60)

# --- LOAD ENSEMBLE MODELS ---
print("Loading ensemble models...")
nn_model   = tf.keras.models.load_model(NN_PATH)
rf_model   = pickle.load(open(RF_PATH,   'rb'))
meta_model = pickle.load(open(META_PATH, 'rb'))

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

@tf.function(reduce_retracing=True)
def fast_nn_predict(x_tensor):
    return nn_model(x_tensor, training=False)

def ensemble_predict(keypoints_1d):
    x = tf.convert_to_tensor(keypoints_1d.reshape(1, -1), dtype=tf.float32)
    nn_probs = fast_nn_predict(x).numpy()[0]
    rf_probs = rf_model.predict_proba(keypoints_1d.reshape(1, -1))[0]

    if ENSEMBLE_MODE == "stacking":
        stack = np.hstack([nn_probs, rf_probs]).reshape(1, -1)
        probs = meta_model.predict_proba(stack)[0]
    else:
        probs = (nn_probs + rf_probs) / 2.0

    return int(np.argmax(probs)), float(np.max(probs))


# --- THREADED CAMERA STREAM ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        self.cap.release()


# --- BACKGROUND INFERENCE WORKER ---
class AsyncInferenceWorker:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.has_new_frame = False

        # Output states
        self.hand_detected = False
        self.current_word = "..."
        self.current_conf = 0.0
        self.landmarks_data = []
        self.prediction_buffer = deque(maxlen=SMOOTH)

        self.thread = threading.Thread(target=self._run_inference, daemon=True)
        self.thread.start()

    def submit_frame(self, frame):
        with self.lock:
            if not self.has_new_frame:  # Skip frame if worker is currently busy
                self.latest_frame = frame
                self.has_new_frame = True

    def get_results(self):
        with self.lock:
            return self.hand_detected, self.current_word, self.current_conf, self.landmarks_data

    def _run_inference(self):
        mp_hands = mp.solutions.hands
        # Set back to model_complexity=1 for full landmark accuracy on fist/folded gestures
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        while self.running:
            frame_to_process = None
            with self.lock:
                if self.has_new_frame:
                    frame_to_process = self.latest_frame
                    self.has_new_frame = False

            if frame_to_process is None:
                time.sleep(0.005)
                continue

            rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_detected = bool(result.multi_hand_landmarks)
            curr_word = "..."
            curr_conf = 0.0
            lm_list = []

            if hand_detected:
                keypoints = []
                for hand_idx in range(2):
                    if hand_idx < len(result.multi_hand_landmarks):
                        lm = result.multi_hand_landmarks[hand_idx]
                        lm_list.append([(p.x, p.y) for p in lm.landmark])
                        for point in lm.landmark:
                            keypoints.extend([point.x, point.y, point.z])
                    else:
                        keypoints.extend([0.0] * 63)

                class_idx, confidence = ensemble_predict(np.array(keypoints, dtype=np.float32))

                if confidence >= CONFIDENCE:
                    self.prediction_buffer.append(class_idx)

                if self.prediction_buffer:
                    smoothed_idx = Counter(self.prediction_buffer).most_common(1)[0][0]
                    curr_word = labels[smoothed_idx].upper() if isinstance(labels, list) else labels[str(smoothed_idx)].upper()
                    curr_conf = confidence
            else:
                self.prediction_buffer.clear()

            with self.lock:
                self.hand_detected = hand_detected
                if hand_detected and curr_conf >= CONFIDENCE:
                    self.current_word = curr_word
                    self.current_conf = curr_conf
                elif not hand_detected:
                    self.current_word = "..."
                    self.current_conf = 0.0
                self.landmarks_data = lm_list

        hands.close()

    def stop(self):
        self.running = False


# --- GROQ & TTS ---
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tts_lock = threading.Lock()

def generate_sentence(words):
    if not groq_client:
        return "Error: GROQ_API_KEY missing."
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an ASL interpreter. Given ASL signs, provide exactly 2 interpretations on separate lines. Line 1 must start with 'Self:' and show first person meaning. Line 2 must start with 'To:' and show instruction meaning. Reply with only these 2 lines, nothing else."},
                {"role": "user",   "content": f"ASL signs in order: {', '.join(words)}"}
            ],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def speak(text):
    if not text.strip(): return
    def tts_worker(phrase):
        if tts_lock.acquire(blocking=False):
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(phrase)
                engine.runAndWait()
            except Exception: pass
            finally: tts_lock.release()
    threading.Thread(target=tts_worker, args=(text,), daemon=True).start()


# --- DRAWING HELPERS ---
def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
    if x2 <= x1 + 2*radius or y2 <= y1 + 2*radius:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def put_text_centered(img, text, cx, y, font, scale, color, thickness=1):
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = cx - size[0] // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def wrap_text(text, max_chars):
    max_chars = max(1, max_chars)
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            if current: lines.append(current)
            current = w
    if current: lines.append(current)
    return lines if lines else [""]


# --- INITIALIZATION ---
cam = ThreadedCamera(0)
worker = AsyncInferenceWorker()

sentence_words  = []
last_valid_word = "..."
generated_sentence, self_text, to_text = "", "", ""

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

WIN_NAME = "ASL Ensemble Detection System"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# --- MAIN LOOP ---
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    if MIRROR_FRAME:
        frame = cv2.flip(frame, 1)

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
    except Exception:
        win_w, win_h = 1280, 720

    win_w, win_h = max(win_w, 640), max(win_h, 480)

    PANEL_RATIO = 0.35
    panel_w  = int(win_w * PANEL_RATIO)
    cam_w    = win_w - panel_w
    cam_h    = win_h
    S        = win_h / 720.0
    PAD      = int(12 * S)
    INNER    = panel_w - 2 * PAD

    fs_title  = max(0.5,  0.9  * S)
    fs_sign   = max(0.6,  1.3  * S)
    fs_normal = max(0.35, 0.55 * S)
    fs_small  = max(0.28, 0.42 * S)
    lh        = int(22 * S)

    frame_resized = cv2.resize(frame, (cam_w, cam_h))
    
    # Send frame to background thread
    worker.submit_frame(frame_resized)

    # Fetch latest prediction asynchronously
    hand_detected, current_word, current_conf, landmarks_data = worker.get_results()

    if hand_detected and current_word != "...":
        last_valid_word = current_word

    # Render hand landmarks
    for hand_lms in landmarks_data:
        coords = [(int(pt[0] * cam_w), int(pt[1] * cam_h)) for pt in hand_lms]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame_resized, coords[start_idx], coords[end_idx], (255, 255, 255), max(1, int(2*S)))
        for pt in coords:
            cv2.circle(frame_resized, pt, max(2, int(3*S)), ACCENT_GREEN, -1)

    canvas = np.full((win_h, win_w, 3), BG_DARK, dtype=np.uint8)
    canvas[0:cam_h, 0:cam_w] = frame_resized
    cv2.rectangle(canvas, (0, 0), (cam_w - 1, cam_h - 1), BORDER, 2)

    # Status Indicator
    dot_r = max(6, int(8 * S))
    dot_x, dot_y = dot_r + 8, dot_r + 8
    dot_color = ACCENT_GREEN if hand_detected else (0, 0, 200)
    cv2.circle(canvas, (dot_x, dot_y), dot_r, dot_color, -1)
    cv2.putText(canvas, "Hand Detected" if hand_detected else "No Hand",
                (dot_x + dot_r + 6, dot_y + int(5*S)), FONT, fs_small, dot_color, 1, cv2.LINE_AA)

    # UI Panel Layout
    px  = cam_w + PAD
    px2 = cam_w + PAD + INNER
    cy  = int(30 * S)

    cv2.putText(canvas, "ASL DETECTION", (px, cy), FONT_BOLD, fs_title, ACCENT_BLUE, max(1,int(2*S)), cv2.LINE_AA)
    cy += int(22 * S)
    cv2.putText(canvas, "Ensemble: NN + Random Forest", (px, cy), FONT, fs_small, ACCENT_GOLD, 1, cv2.LINE_AA)
    cy += int(10 * S)
    cv2.line(canvas, (px, cy), (px2, cy), BORDER, 1)
    cy += int(14 * S)

    # Current Sign Display
    cv2.putText(canvas, "CURRENT SIGN", (px, cy), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)
    cy += int(8 * S)
    card_h = int(70 * S)
    draw_rounded_rect(canvas, px, cy, px2, cy + card_h, BG_CARD, radius=max(4,int(8*S)))

    mid_x  = px + INNER // 2
    sign_y = cy + int(45 * S)
    
    if hand_detected and current_conf >= CONFIDENCE:
        put_text_centered(canvas, current_word, mid_x, sign_y, FONT_BOLD, fs_sign, ACCENT_GREEN, max(1,int(2*S)))
        put_text_centered(canvas, f"{current_conf*100:.1f}% confidence", mid_x, cy + card_h - int(6*S), FONT, fs_small, TEXT_GRAY, 1)
    else:
        if last_valid_word != "...":
            put_text_centered(canvas, last_valid_word, mid_x, sign_y, FONT_BOLD, fs_sign, ACCENT_BLUE, max(1,int(2*S)))
            put_text_centered(canvas, "STAGED - Press ENTER to Add", mid_x, cy + card_h - int(6*S), FONT, fs_small, ACCENT_BLUE, 1)
        else:
            put_text_centered(canvas, "...", mid_x, sign_y, FONT_BOLD, fs_normal, TEXT_DIM, max(1,int(2*S)))

    cy += card_h + int(6 * S)

    if hand_detected and current_conf > 0:
        bar_h_px = max(4, int(6 * S))
        cv2.rectangle(canvas, (px, cy), (px2, cy + bar_h_px), BG_CARD, -1)
        filled    = int(INNER * current_conf)
        bar_color = ACCENT_GREEN if current_conf >= CONFIDENCE else (0, 100, 200)
        cv2.rectangle(canvas, (px, cy), (px + filled, cy + bar_h_px), bar_color, -1)
        cy += bar_h_px

    cy += int(16 * S)

    # Signed Words Sequence
    cv2.putText(canvas, "SIGNED WORDS", (px, cy), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)
    cy += int(8 * S)
    card_h2 = int(55 * S)
    draw_rounded_rect(canvas, px, cy, px2, cy + card_h2, BG_CARD, radius=max(4,int(8*S)))

    sentence_text  = " ".join(sentence_words) if sentence_words else "No words yet..."
    sentence_color = TEXT_WHITE if sentence_words else TEXT_DIM
    max_chars      = max(10, int(INNER / (fs_normal * 11)))
    sw_lines       = wrap_text(sentence_text, max_chars)
    line_start_y   = cy + int(20 * S)
    for li, ln in enumerate(sw_lines[:2]):
        put_text_centered(canvas, ln, mid_x, line_start_y + li * lh, FONT, fs_normal, sentence_color, 1)
    cv2.putText(canvas, f"{len(sentence_words)} word(s)", (px + int(6*S), cy + card_h2 - int(5*S)), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)

    cy += card_h2 + int(16 * S)

    # AI Interpretation Box
    cv2.putText(canvas, "AI INTERPRETATION", (px, cy), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)
    cy += int(8 * S)
    ai_top    = cy
    ai_card_h = int(140 * S)
    draw_rounded_rect(canvas, px, ai_top, px2, ai_top + ai_card_h, BG_CARD, radius=max(4,int(8*S)))

    if generated_sentence:
        iy = ai_top + int(20 * S)
        max_ai_chars = max(10, int(INNER / (fs_normal * 11)))

        cv2.putText(canvas, "Self:", (px + int(8*S), iy), FONT, fs_small, ACCENT_GREEN, 1, cv2.LINE_AA)
        iy += lh
        for sl in wrap_text(self_text, max_ai_chars)[:3]:
            cv2.putText(canvas, sl, (px + int(8*S), iy), FONT, fs_normal, TEXT_WHITE, 1, cv2.LINE_AA)
            iy += lh

        sep_y = ai_top + int(70 * S)
        cv2.line(canvas, (px + int(8*S), sep_y), (px2 - int(8*S), sep_y), BORDER, 1)
        iy = sep_y + int(16 * S)

        cv2.putText(canvas, "To:", (px + int(8*S), iy), FONT, fs_small, ACCENT_GOLD, 1, cv2.LINE_AA)
        iy += lh
        for tl in wrap_text(to_text, max_ai_chars)[:3]:
            cv2.putText(canvas, tl, (px + int(8*S), iy), FONT, fs_normal, TEXT_WHITE, 1, cv2.LINE_AA)
            iy += lh
    else:
        put_text_centered(canvas, "Press G to generate",  mid_x, ai_top + int(55*S), FONT, fs_normal, TEXT_DIM, 1)
        put_text_centered(canvas, "AI interpretation",    mid_x, ai_top + int(55*S) + lh, FONT, fs_normal, TEXT_DIM, 1)

    cy = ai_top + ai_card_h + int(16 * S)

    # Controls Menu
    controls = [
        ("ENTER", "Add last sign"),
        ("BKSP",  "Remove last word"),
        ("G",     "AI Generate"),
        ("1",     "Speak Self"),
        ("2",     "Speak To"),
        ("SPACE", "Clear All"),
        ("Q",     "Quit"),
    ]

    remaining_h = win_h - cy - int(30 * S)
    if remaining_h > int(30 * S):
        cv2.putText(canvas, "CONTROLS", (px, cy), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)
        cy += int(8 * S)
        ctrl_card_h = min(remaining_h, int(len(controls) * 22 * S + 16 * S))
        draw_rounded_rect(canvas, px, cy, px2, cy + ctrl_card_h, BG_CARD, radius=max(4,int(8*S)))
        row_h  = max(18, int(22 * S))
        key_w  = int(55 * S)
        for i, (k, desc) in enumerate(controls):
            ky = cy + int(14*S) + i * row_h
            if ky + row_h > cy + ctrl_card_h: break
            draw_rounded_rect(canvas, px + int(6*S), ky - int(12*S), px + int(6*S) + key_w, ky + int(4*S), BG_DARK, radius=4)
            cv2.putText(canvas, k,    (px + int(10*S), ky), FONT, fs_small, ACCENT_BLUE, 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (px + int(6*S) + key_w + int(8*S), ky), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Model: Ensemble (NN + RF) | Mode: {ENSEMBLE_MODE} | 15 Classes",
                (int(8*S), win_h - int(8*S)), FONT, fs_small, TEXT_DIM, 1, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, canvas)

    # Key Listeners
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (13, 10): # ENTER
        if last_valid_word != "...":
            sentence_words.append(last_valid_word)
            last_valid_word = "..."
            generated_sentence, self_text, to_text = "", "", ""
    elif key == 8: # BACKSPACE
        if sentence_words:
            sentence_words.pop()
            generated_sentence, self_text, to_text = "", "", ""
    elif key == ord('g'):
        if sentence_words:
            generated_sentence = generate_sentence(sentence_words)
            lines = [line.strip() for line in generated_sentence.split('\n') if line.strip()]
            self_text, to_text = "", ""
            for line in lines:
                if line.lower().startswith("self:"):
                    self_text = line.replace('Self:', '', 1).replace('self:', '', 1).strip()
                elif line.lower().startswith("to:"):
                    to_text = line.replace('To:', '', 1).replace('to:', '', 1).strip()
            if not self_text and len(lines) >= 1: 
                self_text = lines[0].replace('Self:', '', 1).replace('self:', '', 1).strip()
            if not to_text and len(lines) >= 2: 
                to_text = lines[1].replace('To:', '', 1).replace('to:', '', 1).strip()
    elif key == ord('1'):
        if generated_sentence and self_text: speak(self_text)
    elif key == ord('2'):
        if generated_sentence and to_text: speak(to_text)
    elif key == 32: # SPACE
        sentence_words.clear()
        last_valid_word = "..."
        generated_sentence, self_text, to_text = "", "", ""

cam.stop()
worker.stop()
cv2.destroyAllWindows()