import cv2
import json
import pickle
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from groq import Groq
import os
import threading
import time
import queue
import tempfile
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime
from gtts import gTTS
import pygame

# ── PATHS ────────────────────────────────────────────────────────────────────
MODELS_DIR = "/home/xero1/Documents/ASL-Detection-FYP/Models"
NN_PATH            = os.path.join(MODELS_DIR, "keypoint_model_15_v4_ensemble_nn.h5")
RF_PATH            = os.path.join(MODELS_DIR, "keypoint_model_15_v4_rf.pkl")
META_PATH          = os.path.join(MODELS_DIR, "keypoint_model_15_v4_meta.pkl")
LABELS_PATH        = os.path.join(MODELS_DIR, "keypoint_labels_15_v4.json")
EMOTION_MODELS_DIR = "emotion_models"

# ── SETTINGS ──────────────────────────────────────────────────────────────────
CONFIDENCE    = 0.75
SMOOTH        = 25
ENSEMBLE_MODE = "stacking"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")

# ── LANGUAGE CONFIG ───────────────────────────────────────────────────────────
BASE_ENGLISH_LOGIC = (
    "The ASL signs should be interpreted literally and simply. "
    "For example: 'FINISH NO' means 'I am done saying no' or 'Stop saying no'. "
    "Do not invent meaning beyond what the signs say. "
    "Keep sentences short and natural."
)

LANGUAGES = {
    "english": {
        "label":       "English",
        "gtts_lang":   "en",
        "groq_prompt": (
            "You are a literal ASL sign language interpreter. "
            "Given a list of ASL signs, produce exactly 2 short natural English sentences. "
            + BASE_ENGLISH_LOGIC +
            " Line 1 must start with 'Self:' — what the signer is expressing about themselves. "
            "Line 2 must start with 'To:' — what the signer is communicating to another person. "
            "Reply with ONLY these 2 lines. No explanations, no extra text."
        )
    },
    "arabic": {
        "label":       "العربية",
        "gtts_lang":   "ar",
        "groq_prompt": (
            "You are a literal ASL sign language interpreter. "
            "Given a list of ASL signs, first understand their literal English meaning, "
            "then translate that meaning naturally into Arabic. "
            + BASE_ENGLISH_LOGIC +
            " Produce exactly 2 short Arabic sentences. "
            "Line 1 must start with 'Self:' — what the signer means about themselves in Arabic. "
            "Line 2 must start with 'To:' — what the signer is saying to someone else in Arabic. "
            "Reply with ONLY these 2 lines in Arabic. No English, no explanations."
        )
    },
    "urdu": {
        "label":       "اردو",
        "gtts_lang":   "ur",
        "groq_prompt": (
            "You are a literal ASL sign language interpreter. "
            "Given a list of ASL signs, first understand their literal English meaning, "
            "then translate that meaning naturally into Urdu. "
            + BASE_ENGLISH_LOGIC +
            " Produce exactly 2 short Urdu sentences. "
            "Line 1 must start with 'Self:' — what the signer means about themselves in Urdu. "
            "Line 2 must start with 'To:' — what the signer is saying to someone else in Urdu. "
            "Reply with ONLY these 2 lines in Urdu. No English, no explanations."
        )
    }
}

# ── LOAD ASL ENSEMBLE MODELS ──────────────────────────────────────────────────
print("Loading ASL ensemble models...")
nn_model   = tf.keras.models.load_model(NN_PATH)
rf_model   = pickle.load(open(RF_PATH,   'rb'))
meta_model = pickle.load(open(META_PATH, 'rb'))
with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)
print(f"ASL models loaded — {len(labels)} classes")

# ── LOAD EMOTION ENSEMBLE MODELS ──────────────────────────────────────────────
print("Loading emotion ensemble models...")
emotion_rf     = pickle.load(open(os.path.join(EMOTION_MODELS_DIR, "emotion_rf.pkl"),     'rb'))
emotion_svm    = pickle.load(open(os.path.join(EMOTION_MODELS_DIR, "emotion_svm.pkl"),    'rb'))
emotion_meta   = pickle.load(open(os.path.join(EMOTION_MODELS_DIR, "emotion_meta.pkl"),   'rb'))
emotion_scaler = pickle.load(open(os.path.join(EMOTION_MODELS_DIR, "emotion_scaler.pkl"), 'rb'))
with open(os.path.join(EMOTION_MODELS_DIR, "emotion_labels.json")) as f:
    EMOTION_LABELS = json.load(f)
print(f"Emotion models loaded — {EMOTION_LABELS}")

EMOTION_EMOJI = {'happy': '😊', 'sad': '😢', 'angry': '😠'}

# ── FACE DETECTION ────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── MEDIAPIPE ────────────────────────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_draw      = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
hands        = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
face_mesh_live = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── FACIAL FEATURE EXTRACTION (matches ex_emotion.py) ────────────────────────
LEFT_EYE_TOP=159; LEFT_EYE_BOT=145; LEFT_EYE_LEFT=33;  LEFT_EYE_RIGHT=133
RIGHT_EYE_TOP=386;RIGHT_EYE_BOT=374;RIGHT_EYE_LEFT=362;RIGHT_EYE_RIGHT=263
LEFT_BROW_INNER=107;LEFT_BROW_OUTER=46;RIGHT_BROW_INNER=336;RIGHT_BROW_OUTER=276
LEFT_BROW_MID=65;RIGHT_BROW_MID=295
MOUTH_LEFT=61;MOUTH_RIGHT=291;MOUTH_TOP=13;MOUTH_BOT=14
UPPER_LIP_TOP=0;LOWER_LIP_BOT=17
NOSE_TIP=4;NOSE_LEFT=129;NOSE_RIGHT=358
CHIN=152;FOREHEAD=10;LEFT_CHEEK=234;RIGHT_CHEEK=454

def lm_dist(a, b):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def extract_emotion_features(lm_list):
    L = lm_list
    features = []
    face_h = lm_dist(L[FOREHEAD], L[CHIN]) + 1e-6

    l_eye_open  = lm_dist(L[LEFT_EYE_TOP],  L[LEFT_EYE_BOT])  / face_h
    r_eye_open  = lm_dist(L[RIGHT_EYE_TOP], L[RIGHT_EYE_BOT]) / face_h
    l_eye_width = lm_dist(L[LEFT_EYE_LEFT], L[LEFT_EYE_RIGHT]) / face_h
    r_eye_width = lm_dist(L[RIGHT_EYE_LEFT],L[RIGHT_EYE_RIGHT])/ face_h
    features += [l_eye_open, r_eye_open, l_eye_width, r_eye_width, (l_eye_open+r_eye_open)/2]

    l_brow_eye = lm_dist(L[LEFT_BROW_MID],  L[LEFT_EYE_TOP])  / face_h
    r_brow_eye = lm_dist(L[RIGHT_BROW_MID], L[RIGHT_EYE_TOP]) / face_h
    brow_dist  = lm_dist(L[LEFT_BROW_INNER],L[RIGHT_BROW_INNER]) / face_h
    l_brow_h   = L[LEFT_BROW_MID].y  - L[LEFT_EYE_TOP].y
    r_brow_h   = L[RIGHT_BROW_MID].y - L[RIGHT_EYE_TOP].y
    features += [l_brow_eye, r_brow_eye, brow_dist, l_brow_h, r_brow_h]

    mouth_w    = lm_dist(L[MOUTH_LEFT],    L[MOUTH_RIGHT]) / face_h
    mouth_h    = lm_dist(L[MOUTH_TOP],     L[MOUTH_BOT])   / face_h
    lip_ratio  = mouth_w / (mouth_h + 1e-6)
    mouth_open = lm_dist(L[UPPER_LIP_TOP], L[LOWER_LIP_BOT]) / face_h
    l_corner_y = L[MOUTH_LEFT].y  - L[MOUTH_TOP].y
    r_corner_y = L[MOUTH_RIGHT].y - L[MOUTH_TOP].y
    features += [mouth_w, mouth_h, lip_ratio, mouth_open, l_corner_y, r_corner_y, (l_corner_y+r_corner_y)/2]

    nose_w   = lm_dist(L[NOSE_LEFT], L[NOSE_RIGHT]) / face_h
    nose_eye = lm_dist(L[NOSE_TIP],  L[LEFT_EYE_BOT]) / face_h
    features += [nose_w, nose_eye]

    cheek_w = lm_dist(L[LEFT_CHEEK], L[RIGHT_CHEEK]) / face_h
    features += [cheek_w]

    eye_mouth  = lm_dist(L[LEFT_EYE_BOT], L[MOUTH_TOP]) / face_h
    brow_mouth = lm_dist(L[LEFT_BROW_MID],L[MOUTH_TOP]) / face_h
    features += [eye_mouth, brow_mouth]

    key_indices = [
        LEFT_EYE_TOP, LEFT_EYE_BOT, RIGHT_EYE_TOP, RIGHT_EYE_BOT,
        LEFT_BROW_INNER, RIGHT_BROW_INNER, LEFT_BROW_MID, RIGHT_BROW_MID,
        MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOT,
        NOSE_TIP, CHIN, FOREHEAD
    ]
    nose_x = L[NOSE_TIP].x
    nose_y = L[NOSE_TIP].y
    for idx in key_indices:
        features.append((L[idx].x - nose_x) / face_h)
        features.append((L[idx].y - nose_y) / face_h)

    return np.array(features, dtype=np.float32)

def emotion_predict(features):
    x         = emotion_scaler.transform(features.reshape(1, -1))
    rf_probs  = emotion_rf.predict_proba(x)[0]
    svm_probs = emotion_svm.predict_proba(x)[0]
    meta_x    = np.hstack([rf_probs, svm_probs]).reshape(1, -1)
    pred      = emotion_meta.predict(meta_x)[0]
    proba     = emotion_meta.predict_proba(meta_x)[0]
    return EMOTION_LABELS[pred], float(np.max(proba)) * 100

# ── GROQ ─────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── PYGAME FOR AUDIO ─────────────────────────────────────────────────────────
pygame.mixer.init()
_tts_lock = threading.Lock()

def speak_gtts(text, lang_code):
    def _run():
        with _tts_lock:
            try:
                tts = gTTS(text=text, lang=lang_code, slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_path = f.name
                tts.save(tmp_path)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                os.remove(tmp_path)
            except Exception as e:
                print(f"TTS Error: {e}")
    threading.Thread(target=_run, daemon=True).start()

# ── APP STATE ────────────────────────────────────────────────────────────────
state = {
    "current_word":         "...",
    "current_conf":         0.0,
    "hand_detected":        False,
    "sentence_words":       [],
    "generated_sentence":   "",
    "conversation_history": [],
    "fps":                  0,
    "inference_ms":         0,
    "language":             "english",
    "emotion":              "...",
    "emotion_conf":         0.0,
    "face_detected":        False,
}

prediction_buffer = deque(maxlen=SMOOTH)
emotion_buffer    = deque(maxlen=7)

# ── CAMERA SETUP ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

_raw_frame_queue = queue.Queue(maxsize=1)
_out_frame_lock  = threading.Lock()
_out_frame       = None

# ── ENSEMBLE PREDICT (ASL) ────────────────────────────────────────────────────
def ensemble_predict(keypoints_1d):
    x        = keypoints_1d.reshape(1, -1)
    nn_probs = nn_model.predict(x, verbose=0)[0]
    rf_probs = rf_model.predict_proba(x)[0]
    if ENSEMBLE_MODE == "stacking":
        stack = np.hstack([nn_probs, rf_probs]).reshape(1, -1)
        probs = meta_model.predict_proba(stack)[0]
    else:
        probs = (nn_probs + rf_probs) / 2.0
    return int(np.argmax(probs)), float(np.max(probs))

# ── THREAD 1: CAMERA ─────────────────────────────────────────────────────────
def camera_thread():
    global _out_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        with _out_frame_lock:
            _out_frame = frame.copy()
        try:
            _raw_frame_queue.put_nowait(frame)
        except queue.Full:
            pass

# ── THREAD 2: INFERENCE (ASL) ─────────────────────────────────────────────────
def inference_thread():
    global _out_frame
    while True:
        try:
            frame = _raw_frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        t0     = time.time()
        small  = cv2.resize(frame, (320, 240))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        annotated     = frame.copy()
        hand_detected = False
        current_conf  = 0.0
        current_word  = state["current_word"]

        if result.multi_hand_landmarks:
            hand_detected = True
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    annotated, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 220, 100), thickness=2, circle_radius=3),
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

            class_idx, confidence = ensemble_predict(np.array(keypoints))
            current_conf = confidence
            if confidence >= CONFIDENCE:
                prediction_buffer.append(class_idx)
            if prediction_buffer:
                smoothed_idx = Counter(prediction_buffer).most_common(1)[0][0]
                current_word = labels[smoothed_idx].upper()
        else:
            prediction_buffer.clear()
            current_conf = 0.0

        inference_ms = (time.time() - t0) * 1000
        state["hand_detected"]  = hand_detected
        state["current_word"]   = current_word
        state["current_conf"]   = current_conf
        state["inference_ms"]   = round(inference_ms, 1)

        with _out_frame_lock:
            _out_frame = annotated

# ── THREAD 3: EMOTION DETECTION ──────────────────────────────────────────────
def emotion_thread():
    print("Emotion thread started.")
    debug_count = 0
    while True:
        time.sleep(1.5)
        debug_count += 1

        with _out_frame_lock:
            frame = _out_frame
        if frame is None:
            if debug_count % 3 == 0:
                print(f"  Emotion: waiting for frame...")
            continue

        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
            )
            print(f"Emotion check #{debug_count}: found {len(faces)} faces")

            if len(faces) == 0:
                state["face_detected"] = False
                state["emotion"]       = "..."
                state["emotion_conf"]  = 0.0
                continue

            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            print(f"  Face detected: {w}x{h} at ({x},{y})")
            
            pad = int(0.2 * w)
            x1  = max(0, x - pad)
            y1  = max(0, y - pad)
            x2  = min(frame.shape[1], x + w + pad)
            y2  = min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (256, 256))

            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            result   = face_mesh_live.process(rgb_crop)
            
            if not result.multi_face_landmarks:
                print(f"  Face Mesh: 0 landmarks (no face detected)")
                state["face_detected"] = False
                state["emotion"]       = "..."
                state["emotion_conf"]  = 0.0
                continue

            lm = result.multi_face_landmarks[0].landmark
            landmark_count = len(lm)
            print(f"  Face Mesh: {landmark_count} landmarks")

            # Check if we have enough landmarks (need at least 300 of 468)
            if landmark_count < 300:
                print(f"  Warning: Not enough landmarks ({landmark_count}/468), skipping")
                state["face_detected"] = False
                state["emotion"]       = "..."
                state["emotion_conf"]  = 0.0
                continue

            features = extract_emotion_features(lm)
            emotion, conf = emotion_predict(features)
            print(f"  Emotion: {emotion} ({conf:.1f}%)")

            emotion_buffer.append(emotion)
            voted = Counter(emotion_buffer).most_common(1)[0][0]

            state["face_detected"] = True
            state["emotion"]       = voted
            state["emotion_conf"]  = round(conf, 1)

        except Exception as e:
            print(f"Emotion error: {e}")
            state["face_detected"] = False
            state["emotion"]       = "..."
            state["emotion_conf"]  = 0.0

# ── START THREADS ─────────────────────────────────────────────────────────────
threading.Thread(target=camera_thread,    daemon=True).start()
threading.Thread(target=inference_thread, daemon=True).start()
threading.Thread(target=emotion_thread,   daemon=True).start()

# ── STREAM ────────────────────────────────────────────────────────────────────
def generate_frames():
    prev_time = time.time()
    while True:
        with _out_frame_lock:
            frame = _out_frame
        if frame is None:
            time.sleep(0.01)
            continue
        curr_time    = time.time()
        fps          = 1 / (curr_time - prev_time + 1e-9)
        prev_time    = curr_time
        state["fps"] = round(fps, 1)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', labels=labels, languages=LANGUAGES)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify({
        "current_word":  state["current_word"],
        "current_conf":  round(state["current_conf"] * 100, 1),
        "hand_detected": state["hand_detected"],
        "words":         state["sentence_words"],
        "sentence":      state["generated_sentence"],
        "fps":           state["fps"],
        "inference_ms":  state["inference_ms"],
        "language":      state["language"],
        "emotion":       state["emotion"],
        "emotion_conf":  state["emotion_conf"],
        "face_detected": state["face_detected"],
    })

@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get("language", "english")
    if lang in LANGUAGES:
        state["language"]           = lang
        state["generated_sentence"] = ""
    return jsonify({"language": state["language"]})

@app.route('/add_word', methods=['POST'])
def add_word():
    word = state["current_word"]
    if word not in ["...", "No hand detected", ""]:
        state["sentence_words"].append(word)
        state["generated_sentence"] = ""
    return jsonify({"words": state["sentence_words"]})

@app.route('/remove_word', methods=['POST'])
def remove_word():
    if state["sentence_words"]:
        state["sentence_words"].pop()
        state["generated_sentence"] = ""
    return jsonify({"words": state["sentence_words"]})

@app.route('/clear', methods=['POST'])
def clear():
    state["sentence_words"].clear()
    state["generated_sentence"] = ""
    return jsonify({"ok": True})

@app.route('/generate', methods=['POST'])
def generate():
    if not state["sentence_words"]:
        return jsonify({"sentence": ""})
    try:
        words    = state["sentence_words"]
        lang_key = state["language"]
        prompt   = LANGUAGES[lang_key]["groq_prompt"]
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": f"ASL signs in order: {', '.join(words)}"}
            ],
            max_tokens=120
        )
        sentence = response.choices[0].message.content.strip()
        state["generated_sentence"] = sentence
        state["conversation_history"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "language":  LANGUAGES[lang_key]["label"],
            "signs":     " → ".join(words),
            "sentence":  sentence
        })
    except Exception as e:
        state["generated_sentence"] = f"Error: {str(e)}"
    return jsonify({"sentence": state["generated_sentence"]})

@app.route('/speak', methods=['POST'])
def speak():
    data      = request.get_json()
    text      = data.get("text", "")
    lang_key  = state["language"]
    lang_code = LANGUAGES[lang_key]["gtts_lang"]
    if text:
        speak_gtts(text, lang_code)
    return jsonify({"ok": True})

@app.route('/reply', methods=['POST'])
def reply():
    data    = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"ok": False})
    lang_key  = state["language"]
    lang_code = LANGUAGES[lang_key]["gtts_lang"]
    state["conversation_history"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "language":  LANGUAGES[lang_key]["label"],
        "signs":     "[Text Reply]",
        "sentence":  f"Reply: {message}"
    })
    speak_gtts(message, lang_code)
    return jsonify({"ok": True})

@app.route('/history')
def history():
    return jsonify({"history": state["conversation_history"]})

@app.route('/export_history')
def export_history():
    lines = []
    for entry in state["conversation_history"]:
        lines.append(f"[{entry['timestamp']}] [{entry.get('language','English')}]")
        lines.append(f"Signs    : {entry['signs']}")
        lines.append(f"Sentence : {entry['sentence']}")
        lines.append("")
    text = "\n".join(lines) if lines else "No conversation history yet."
    return Response(text, mimetype='text/plain',
                    headers={"Content-Disposition": "attachment; filename=asl_conversation.txt"})

if __name__ == '__main__':
    print("\nASL Flask App running → http://127.0.0.1:5000\n")
    app.run(debug=False, threaded=True)