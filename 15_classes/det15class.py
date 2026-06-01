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

# --- LOAD MODEL ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

print(f"Model loaded - {len(labels)} classes")
print(f"Classes: {labels}\n")

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

# --- TEXT TO SPEECH SETUP ---
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- INIT ---
prediction_buffer  = deque(maxlen=SMOOTH)
sentence_words     = []
current_word       = "..."
current_conf       = 0.0
cap                = cv2.VideoCapture(0)

print("Controls:")
print("  ENTER     - Add current word to sentence")
print("  BACKSPACE - Remove last word from sentence")
print("  G         - Generate both sentence interpretations")
print("  1         - Speak Self interpretation")
print("  2         - Speak To interpretation")
print("  SPACE     - Clear everything")
print("  Q         - Quit\n")

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

        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
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

    # --- TOP BAR - SIGNED WORDS ---
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    sentence_text = " ".join(sentence_words) if sentence_words else "Press ENTER to add word..."
    cv2.putText(frame, sentence_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # --- SECOND BAR - GENERATED SENTENCES ---
    cv2.rectangle(frame, (0, 55), (w, 140), (10, 10, 40), -1)
    if generated_sentence:
        lines = generated_sentence.split('\n')
        line1 = lines[0] if len(lines) > 0 else ""
        line2 = lines[1] if len(lines) > 1 else ""
        cv2.putText(frame, line1, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(frame, line2, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    else:
        cv2.putText(frame, "Press G to generate sentence...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    # --- MIDDLE BAR - HAND COUNT ---
    cv2.rectangle(frame, (0, 140), (w, 180), (0, 0, 0), -1)
    hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    cv2.putText(frame, f"Hands: {hand_count}", (10, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if hand_detected else (0, 0, 255), 2)
    cv2.putText(frame, "15 Classes | ASL Detection", (w - 320, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # --- BOTTOM BAR - CURRENT WORD ---
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    if hand_detected and current_conf >= CONFIDENCE:
        label_text = f"{current_word}  ({current_conf*100:.1f}%)"
        color = (0, 255, 0)
    else:
        label_text = current_word
        color = (0, 165, 255)

    cv2.putText(frame, label_text, (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    # --- CONTROLS HINT ---
    cv2.putText(frame, "ENTER=Add  BKSP=Remove  G=Generate  1=Self  2=To  SPACE=Clear  Q=Quit",
                (10, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    cv2.imshow("ASL Detection", frame)

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
            if len(lines) >= 1:
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