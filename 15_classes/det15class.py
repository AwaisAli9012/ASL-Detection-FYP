import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from groq import Groq
import os
import pyttsx3

# --- PATHS ---
MODEL_PATH  = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_model_15_v4.h5"
LABELS_PATH = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels_15_v4.json"

# --- SETTINGS ---
CONFIDENCE   = 0.75
SMOOTH       = 25
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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

# --- LOAD MODEL ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

print(f"Model loaded - {len(labels)} classes")

# --- GROQ SETUP ---
groq_client        = Groq(api_key=GROQ_API_KEY)
generated_sentence = ""

def generate_sentence(words):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an ASL interpreter. Given ASL signs, provide exactly 2 interpretations on separate lines. Line 1 must start with 'Self:' and show first person meaning. Line 2 must start with 'To:' and show instruction meaning. Reply with only these 2 lines, nothing else."
                },
                {
                    "role": "user",
                    "content": f"ASL signs in order: {', '.join(words)}"
                }
            ],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- TEXT TO SPEECH ---
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- HELPER FUNCTIONS ---
def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
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

# --- INIT ---
prediction_buffer  = deque(maxlen=SMOOTH)
sentence_words     = []
current_word       = "..."
current_conf       = 0.0
cap                = cv2.VideoCapture(0)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

print("Controls: ENTER=Add | BKSP=Remove | G=Generate | 1=Speak Self | 2=Speak To | SPACE=Clear | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    cam_h, cam_w = frame.shape[:2]

    # --- CANVAS ---
    panel_w  = 420
    canvas_w = cam_w + panel_w
    canvas_h = cam_h
    canvas   = np.full((canvas_h, canvas_w, 3), BG_DARK, dtype=np.uint8)

    # --- PROCESS FRAME ---
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=ACCENT_GREEN, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        keypoints = []
        for hand_idx in range(2):
            if hand_idx < len(result.multi_hand_landmarks):
                lm = result.multi_hand_landmarks[hand_idx]
                for point in lm.landmark:
                    keypoints.extend([point.x, point.y, point.z])
            else:
                keypoints.extend([0.0] * 63)

        inp        = np.array(keypoints).reshape(1, -1)
        preds      = model.predict(inp, verbose=0)[0]
        confidence = float(np.max(preds))
        class_idx  = int(np.argmax(preds))

        if confidence >= CONFIDENCE:
            prediction_buffer.append(class_idx)

        if prediction_buffer:
            smoothed_idx = Counter(prediction_buffer).most_common(1)[0][0]
            current_word = labels[smoothed_idx].upper()
            current_conf = confidence
    else:
        prediction_buffer.clear()
        current_conf = 0.0

    # --- PLACE WEBCAM ---
    canvas[0:cam_h, 0:cam_w] = frame

    # --- WEBCAM BORDER ---
    cv2.rectangle(canvas, (0, 0), (cam_w, cam_h), BORDER, 2)

    # --- STATUS DOT ---
    dot_color = ACCENT_GREEN if hand_detected else (0, 0, 200)
    cv2.circle(canvas, (20, 20), 8, dot_color, -1)
    cv2.putText(canvas, "Hand Detected" if hand_detected else "No Hand",
                (35, 25), FONT, 0.5, dot_color, 1, cv2.LINE_AA)

    # --- RIGHT PANEL ---
    px = cam_w + 10
    pw = panel_w - 20

    # Title
    cv2.putText(canvas, "ASL DETECTION", (px, 35), FONT_BOLD, 0.8, ACCENT_BLUE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Sign Language to Text System", (px, 55), FONT, 0.4, TEXT_GRAY, 1, cv2.LINE_AA)
    cv2.line(canvas, (px, 62), (px + pw, 62), BORDER, 1)

    # Current Sign Card
    cv2.putText(canvas, "CURRENT SIGN", (px, 85), FONT, 0.4, TEXT_GRAY, 1, cv2.LINE_AA)
    draw_rounded_rect(canvas, px, 92, px + pw, 155, BG_CARD, radius=8)

    if hand_detected and current_conf >= CONFIDENCE:
        put_text_centered(canvas, current_word, px + pw//2, 135, FONT_BOLD, 1.2, ACCENT_GREEN, 2)
        put_text_centered(canvas, f"{current_conf*100:.1f}% confidence",
                          px + pw//2, 152, FONT, 0.4, TEXT_GRAY, 1)
    else:
        put_text_centered(canvas, current_word, px + pw//2, 135, FONT_BOLD, 1.0, TEXT_DIM, 2)

    # Confidence bar
    if hand_detected and current_conf > 0:
        bar_x = px + 10
        bar_y = 158
        bar_w = pw - 20
        bar_h = 6
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), BG_CARD, -1)
        filled    = int(bar_w * current_conf)
        bar_color = ACCENT_GREEN if current_conf >= CONFIDENCE else (0, 100, 200)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)

    # Signed Words Card
    cv2.putText(canvas, "SIGNED WORDS", (px, 185), FONT, 0.4, TEXT_GRAY, 1, cv2.LINE_AA)
    draw_rounded_rect(canvas, px, 192, px + pw, 240, BG_CARD, radius=8)
    sentence_text  = " ".join(sentence_words) if sentence_words else "No words yet..."
    sentence_color = TEXT_WHITE if sentence_words else TEXT_DIM
    words_display  = sentence_text if len(sentence_text) <= 28 else sentence_text[:25] + "..."
    put_text_centered(canvas, words_display, px + pw//2, 222, FONT, 0.6, sentence_color, 1)
    cv2.putText(canvas, f"{len(sentence_words)} word(s)", (px + 10, 237), FONT, 0.35, TEXT_GRAY, 1, cv2.LINE_AA)

    # AI Interpretation Card
    cv2.putText(canvas, "AI INTERPRETATION", (px, 260), FONT, 0.4, TEXT_GRAY, 1, cv2.LINE_AA)
    draw_rounded_rect(canvas, px, 267, px + pw, 390, BG_CARD, radius=8)

    if generated_sentence:
        lines = generated_sentence.split('\n')
        if len(lines) >= 1:
            cv2.putText(canvas, "Self:", (px + 10, 290), FONT, 0.4, ACCENT_GREEN, 1, cv2.LINE_AA)
            self_text = lines[0].replace('Self:', '').strip().split()
            cv2.putText(canvas, ' '.join(self_text[:5]), (px + 10, 308), FONT, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
            if len(self_text) > 5:
                cv2.putText(canvas, ' '.join(self_text[5:]), (px + 10, 324), FONT, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)

        cv2.line(canvas, (px + 10, 335), (px + pw - 10, 335), BORDER, 1)

        if len(lines) >= 2:
            cv2.putText(canvas, "To:", (px + 10, 352), FONT, 0.4, ACCENT_GOLD, 1, cv2.LINE_AA)
            to_text = lines[1].replace('To:', '').strip().split()
            cv2.putText(canvas, ' '.join(to_text[:5]), (px + 10, 370), FONT, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
            if len(to_text) > 5:
                cv2.putText(canvas, ' '.join(to_text[5:]), (px + 10, 386), FONT, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
    else:
        put_text_centered(canvas, "Press G to generate", px + pw//2, 315, FONT, 0.45, TEXT_DIM, 1)
        put_text_centered(canvas, "AI interpretation", px + pw//2, 335, FONT, 0.45, TEXT_DIM, 1)

    # Controls Card
    cv2.putText(canvas, "CONTROLS", (px, 408), FONT, 0.4, TEXT_GRAY, 1, cv2.LINE_AA)
    draw_rounded_rect(canvas, px, 415, px + pw, canvas_h - 10, BG_CARD, radius=8)

    controls = [
        ("ENTER", "Add word to sentence"),
        ("BKSP",  "Remove last word"),
        ("G",     "Generate AI sentence"),
        ("1",     "Speak Self interpretation"),
        ("2",     "Speak To interpretation"),
        ("SPACE", "Clear everything"),
        ("Q",     "Quit"),
    ]

    for i, (k, desc) in enumerate(controls):
        y = 433 + i * 26
        draw_rounded_rect(canvas, px + 8, y - 13, px + 55, y + 5, BG_DARK, radius=4)
        cv2.putText(canvas, k, (px + 12, y), FONT, 0.38, ACCENT_BLUE, 1, cv2.LINE_AA)
        cv2.putText(canvas, desc, (px + 62, y), FONT, 0.38, TEXT_GRAY, 1, cv2.LINE_AA)

    # Footer
    cv2.putText(canvas, "Model: 15 ASL Classes | Accuracy: 83.5%",
                (10, cam_h - 10), FONT, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

    cv2.imshow("ASL Sign Language Detection System", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 13:  # ENTER
        if current_word not in ["...", "No hand detected"]:
            sentence_words.append(current_word)
            generated_sentence = ""
            print(f"Added: {current_word} | Words: {' '.join(sentence_words)}")
    elif key == 8:   # BACKSPACE
        if sentence_words:
            removed = sentence_words.pop()
            generated_sentence = ""
            print(f"Removed: {removed} | Words: {' '.join(sentence_words)}")
    elif key == ord('g'):
        if sentence_words:
            print("Generating sentences...")
            generated_sentence = generate_sentence(sentence_words)
            print(f"Generated:\n{generated_sentence}")
        else:
            print("No words to generate from.")
    elif key == ord('1'):
        if generated_sentence:
            lines = generated_sentence.split('\n')
            if lines:
                self_line = lines[0].replace('Self:', '').strip()
                print(f"Speaking Self: {self_line}")
                speak(self_line)
        else:
            print("Generate a sentence first with G.")
    elif key == ord('2'):
        if generated_sentence:
            lines = generated_sentence.split('\n')
            if len(lines) >= 2:
                to_line = lines[1].replace('To:', '').strip()
                print(f"Speaking To: {to_line}")
                speak(to_line)
        else:
            print("Generate a sentence first with G.")
    elif key == 32:  # SPACE
        sentence_words.clear()
        generated_sentence = ""
        print("Cleared.")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Detection stopped.")
print(f"Final words: {' '.join(sentence_words)}")