import cv2
import json
import pickle
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from groq import Groq
import os
import pyttsx3
import threading

# --- PATHS ---
NN_PATH     = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_model_15_v4_ensemble_nn.h5"
RF_PATH     = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_model_15_v4_rf.pkl"
META_PATH   = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_model_15_v4_meta.pkl"
LABELS_PATH = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels_15_v4.json"

# --- SETTINGS ---
CONFIDENCE   = 0.75
SMOOTH       = 25
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Ensemble mode: "soft_vote" | "stacking"
ENSEMBLE_MODE = "stacking"

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

print(f"Ensemble loaded - {len(labels)} classes | mode: {ENSEMBLE_MODE}")


# --- ENSEMBLE PREDICT ---
def ensemble_predict(keypoints_1d):
    x = tf.convert_to_tensor(keypoints_1d.reshape(1, -1), dtype=tf.float32)
    nn_probs = nn_model(x, training=False).numpy()[0]
    rf_probs = rf_model.predict_proba(keypoints_1d.reshape(1, -1))[0]

    if ENSEMBLE_MODE == "stacking":
        stack = np.hstack([nn_probs, rf_probs]).reshape(1, -1)
        probs = meta_model.predict_proba(stack)[0]
    else:  # soft_vote
        probs = (nn_probs + rf_probs) / 2.0

    return int(np.argmax(probs)), float(np.max(probs))


# --- GROQ SETUP ---
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None
generated_sentence = ""

def generate_sentence(words):
    if not groq_client:
        return "Error: GROQ_API_KEY environment variable missing."
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


# --- THREADED TTS ---
tts_lock = threading.Lock()

def speak(text):
    if not text.strip():
        return
        
    def tts_worker(phrase):
        with tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(phrase)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Thread Error: {e}")

    threading.Thread(target=tts_worker, args=(text,), daemon=True).start()


# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# --- HELPERS ---
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
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines if lines else [""]


# --- INIT ---
prediction_buffer = deque(maxlen=SMOOTH)
sentence_words    = []
current_word      = "..."
last_valid_word   = "..."  # Memory snapshot to hold key inputs when hand drops out
current_conf      = 0.0
cap               = cv2.VideoCapture(0)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

WIN_NAME = "ASL Ensemble Detection System"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

self_text = ""
to_text = ""

print("Controls: ENTER=Add | BKSP=Remove | G=AI Generate | SPACE=Clear All | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
    if win_w < 100 or win_h < 100:
        win_w, win_h = 1280, 720
    
    win_w, win_h = max(win_w, 640), max(win_h, 480)

    PANEL_RATIO = 0.35
    panel_w  = int(win_w * PANEL_RATIO)
    cam_w    = win_w - panel_w
    cam_h    = win_h

    S     = win_h / 720.0
    PAD   = int(12 * S)
    INNER = panel_w - 2 * PAD

    fs_title  = max(0.5,  0.9  * S)
    fs_sub    = max(0.35, 0.5  * S)
    fs_sign   = max(0.6,  1.3  * S)
    fs_normal = max(0.35, 0.55 * S)
    fs_small  = max(0.28, 0.42 * S)

    lh = int(22 * S)

    frame_resized = cv2.resize(frame, (cam_w, cam_h))
    canvas = np.full((win_h, win_w, 3), BG_DARK, dtype=np.uint8)

    rgb    = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_resized, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=ACCENT_GREEN, thickness=max(1,int(2*S)), circle_radius=max(2,int(3*S))),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=max(1,int(2*S)))
            )

        keypoints = []
        for hand_idx in range(2):
            if hand_idx < len(result.multi_hand_landmarks):
                lm = result.multi_hand_landmarks[hand_idx]
                for point in lm.landmark:
                    keypoints.extend([point.x, point.y, point.z])
            else:
                keypoints.extend([0.0] * 63)

        class_idx, confidence = ensemble_predict(np.array(keypoints, dtype=np.float32))

        if confidence >= CONFIDENCE:
            prediction_buffer.append(class_idx)

        if prediction_buffer:
            smoothed_idx = Counter(prediction_buffer).most_common(1)[0][0]
            
            if isinstance(labels, list):
                current_word = labels[smoothed_idx].upper()
            else:
                current_word = labels[str(smoothed_idx)].upper()
                
            current_conf = confidence
            last_valid_word = current_word  # Save persistent snapshot of the last made sign
    else:
        prediction_buffer.clear()
        current_conf = 0.0
        current_word = "..."

    canvas[0:cam_h, 0:cam_w] = frame_resized
    cv2.rectangle(canvas, (0, 0), (cam_w - 1, cam_h - 1), BORDER, 2)

    dot_r = max(6, int(8 * S))
    dot_x, dot_y = dot_r + 8, dot_r + 8
    dot_color = ACCENT_GREEN if hand_detected else (0, 0, 200)
    cv2.circle(canvas, (dot_x, dot_y), dot_r, dot_color, -1)
    cv2.putText(canvas, "Hand Detected" if hand_detected else "No Hand",
                (dot_x + dot_r + 6, dot_y + int(5*S)), FONT, fs_small, dot_color, 1, cv2.LINE_AA)

    px  = cam_w + PAD
    px2 = cam_w + PAD + INNER
    cy  = int(30 * S)

    cv2.putText(canvas, "ASL DETECTION", (px, cy), FONT_BOLD, fs_title, ACCENT_BLUE, max(1,int(2*S)), cv2.LINE_AA)
    cy += int(22 * S)
    cv2.putText(canvas, f"Ensemble: NN + Random Forest", (px, cy), FONT, fs_small, ACCENT_GOLD, 1, cv2.LINE_AA)
    cy += int(10 * S)
    cv2.line(canvas, (px, cy), (px2, cy), BORDER, 1)
    cy += int(14 * S)

    cv2.putText(canvas, "CURRENT SIGN", (px, cy), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)
    cy += int(8 * S)
    card_h = int(70 * S)
    draw_rounded_rect(canvas, px, cy, px2, cy + card_h, BG_CARD, radius=max(4,int(8*S)))

    mid_x  = px + INNER // 2
    sign_y = cy + int(45 * S)
    
    if hand_detected and current_conf >= CONFIDENCE:
        put_text_centered(canvas, current_word, mid_x, sign_y, FONT_BOLD, fs_sign, ACCENT_GREEN, max(1,int(2*S)))
        put_text_centered(canvas, f"{current_conf*100:.1f}% confidence",
                          mid_x, cy + card_h - int(6*S), FONT, fs_small, TEXT_GRAY, 1)
    else:
        # Usability Upgrade: If hand is down but a valid staged sign exists, keep it visible
        if last_valid_word != "...":
            put_text_centered(canvas, last_valid_word, mid_x, sign_y, FONT_BOLD, fs_sign, ACCENT_BLUE, max(1,int(2*S)))
            put_text_centered(canvas, "STAGED - Press ENTER to Add",
                              mid_x, cy + card_h - int(6*S), FONT, fs_small, ACCENT_BLUE, 1)
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
    cv2.putText(canvas, f"{len(sentence_words)} word(s)",
                (px + int(6*S), cy + card_h2 - int(5*S)), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)

    cy += card_h2 + int(16 * S)

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
        self_lines = wrap_text(self_text, max_ai_chars)
        for sl in self_lines[:3]:
            cv2.putText(canvas, sl, (px + int(8*S), iy), FONT, fs_normal, TEXT_WHITE, 1, cv2.LINE_AA)
            iy += lh

        sep_y = ai_top + int(70 * S)
        cv2.line(canvas, (px + int(8*S), sep_y), (px2 - int(8*S), sep_y), BORDER, 1)
        iy = sep_y + int(16 * S)

        cv2.putText(canvas, "To:", (px + int(8*S), iy), FONT, fs_small, ACCENT_GOLD, 1, cv2.LINE_AA)
        iy += lh
        to_lines = wrap_text(to_text, max_ai_chars)
        for tl in to_lines[:3]:
            cv2.putText(canvas, tl, (px + int(8*S), iy), FONT, fs_normal, TEXT_WHITE, 1, cv2.LINE_AA)
            iy += lh
    else:
        put_text_centered(canvas, "Press G to generate",  mid_x, ai_top + int(55*S), FONT, fs_normal, TEXT_DIM, 1)
        put_text_centered(canvas, "AI interpretation",    mid_x, ai_top + int(55*S) + lh, FONT, fs_normal, TEXT_DIM, 1)

    cy = ai_top + ai_card_h + int(16 * S)

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
            if ky + row_h > cy + ctrl_card_h:
                break
            draw_rounded_rect(canvas, px + int(6*S), ky - int(12*S),
                              px + int(6*S) + key_w, ky + int(4*S), BG_DARK, radius=4)
            cv2.putText(canvas, k,    (px + int(10*S), ky), FONT, fs_small, ACCENT_BLUE, 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (px + int(6*S) + key_w + int(8*S), ky), FONT, fs_small, TEXT_GRAY, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Model: Ensemble (NN + RF) | Mode: {ENSEMBLE_MODE} | 15 Classes",
                (int(8*S), win_h - int(8*S)), FONT, fs_small, TEXT_DIM, 1, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13: # ENTER
        # Inputs according to your command: locks in the persistent memory snapshot of the last sign made
        if last_valid_word != "...":
            sentence_words.append(last_valid_word)
            print(f"Inputted Sign: {last_valid_word} | Sequence: {' '.join(sentence_words)}")
            last_valid_word = "..." # Clears snapshot cache so you must perform another sign before adding again
            generated_sentence = ""
            self_text = ""
            to_text = ""
    elif key == 8: # BACKSPACE
        # Removes the last sign added to your input list
        if sentence_words:
            removed = sentence_words.pop()
            generated_sentence = ""
            self_text = ""
            to_text = ""
            print(f"Removed Last Input: {removed}")
    elif key == ord('g'):
        if sentence_words:
            print("Generating...")
            generated_sentence = generate_sentence(sentence_words)
            print(f"Generated:\n{generated_sentence}")
            
            lines = [line.strip() for line in generated_sentence.split('\n') if line.strip()]
            self_text = ""
            to_text = ""
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
        if generated_sentence and self_text:
            speak(self_text)
    elif key == ord('2'):
        if generated_sentence and to_text:
            speak(to_text)
    elif key == 32: # SPACE
        # Completely resets the system arrays according to inputs
        sentence_words.clear()
        last_valid_word = "..."
        generated_sentence = ""
        self_text = ""
        to_text = ""
        print("Cleared all inputs.")

cap.release()
cv2.destroyAllWindows()
hands.close()