import cv2
import numpy as np
import os
import mediapipe as mp


# --- PATHS ---------------------------------------------------
# Dynamically locate dataset directories relative to home directory or script location
BASE_DATASET = os.path.expanduser("~/Documents/Dataset")

FER_TRAIN_DIR = os.path.join(BASE_DATASET, "train")
RAFDB_TRAIN   = os.path.join(BASE_DATASET, "DATASET", "train")
SAVE_DIR      = os.path.join(os.path.dirname(__file__), "emotion_keypoints")
os.makedirs(SAVE_DIR, exist_ok=True)

EMOTIONS = ['angry', 'happy', 'sad']
TARGET_SAMPLES = 800
IMG_SIZE       = 512
RAFDB_MAP = {'4': 'happy', '5': 'sad', '6': 'angry'}

# ── MEDIAPIPE FACE MESH ───────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.3
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── KEY LANDMARK INDICES (MediaPipe Face Mesh) ────────────────────────────────
# Eyes
LEFT_EYE_TOP     = 159
LEFT_EYE_BOT     = 145
LEFT_EYE_LEFT    = 33
LEFT_EYE_RIGHT   = 133
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOT    = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

# Eyebrows
LEFT_BROW_INNER  = 107
LEFT_BROW_OUTER  = 46
RIGHT_BROW_INNER = 336
RIGHT_BROW_OUTER = 276
LEFT_BROW_MID    = 65
RIGHT_BROW_MID   = 295

# Mouth
MOUTH_LEFT       = 61
MOUTH_RIGHT      = 291
MOUTH_TOP        = 13
MOUTH_BOT        = 14
UPPER_LIP_TOP    = 0
LOWER_LIP_BOT    = 17
MOUTH_TOP_L      = 82
MOUTH_TOP_R      = 312
MOUTH_BOT_L      = 87
MOUTH_BOT_R      = 317

# Nose
NOSE_TIP         = 4
NOSE_LEFT        = 129
NOSE_RIGHT       = 358

# Chin & Forehead
CHIN             = 152
FOREHEAD         = 10
LEFT_CHEEK       = 234
RIGHT_CHEEK      = 454

def dist(a, b):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def extract_features(lm_list):
    L = lm_list  # shorthand
    features = []

    # Face size normalizer
    face_h = dist(L[FOREHEAD], L[CHIN]) + 1e-6

    # ── EYE OPENNESS ──────────────────────────────────────────────────────────
    l_eye_open  = dist(L[LEFT_EYE_TOP],  L[LEFT_EYE_BOT])  / face_h
    r_eye_open  = dist(L[RIGHT_EYE_TOP], L[RIGHT_EYE_BOT]) / face_h
    l_eye_width = dist(L[LEFT_EYE_LEFT], L[LEFT_EYE_RIGHT]) / face_h
    r_eye_width = dist(L[RIGHT_EYE_LEFT],L[RIGHT_EYE_RIGHT])/ face_h
    features += [l_eye_open, r_eye_open, l_eye_width, r_eye_width]
    features += [(l_eye_open+r_eye_open)/2]  # avg eye openness

    # ── EYEBROW RAISE & FURROW ────────────────────────────────────────────────
    l_brow_eye  = dist(L[LEFT_BROW_MID],  L[LEFT_EYE_TOP])  / face_h
    r_brow_eye  = dist(L[RIGHT_BROW_MID], L[RIGHT_EYE_TOP]) / face_h
    brow_dist   = dist(L[LEFT_BROW_INNER],L[RIGHT_BROW_INNER]) / face_h
    l_brow_h    = L[LEFT_BROW_MID].y - L[LEFT_EYE_TOP].y
    r_brow_h    = L[RIGHT_BROW_MID].y - L[RIGHT_EYE_TOP].y
    features += [l_brow_eye, r_brow_eye, brow_dist, l_brow_h, r_brow_h]

    # ── MOUTH FEATURES ────────────────────────────────────────────────────────
    mouth_w     = dist(L[MOUTH_LEFT],  L[MOUTH_RIGHT]) / face_h
    mouth_h     = dist(L[MOUTH_TOP],   L[MOUTH_BOT])   / face_h
    lip_ratio   = mouth_w / (mouth_h + 1e-6)
    mouth_open  = dist(L[UPPER_LIP_TOP], L[LOWER_LIP_BOT]) / face_h
    # Mouth corner angles (up = happy, down = sad)
    l_corner_y  = L[MOUTH_LEFT].y  - L[MOUTH_TOP].y
    r_corner_y  = L[MOUTH_RIGHT].y - L[MOUTH_TOP].y
    features += [mouth_w, mouth_h, lip_ratio, mouth_open, l_corner_y, r_corner_y]
    features += [(l_corner_y + r_corner_y) / 2]  # avg corner direction

    # ── NOSE ──────────────────────────────────────────────────────────────────
    nose_w      = dist(L[NOSE_LEFT], L[NOSE_RIGHT]) / face_h
    nose_eye    = dist(L[NOSE_TIP],  L[LEFT_EYE_BOT]) / face_h
    features += [nose_w, nose_eye]

    # ── CHEEK PUFF (happy indicator) ──────────────────────────────────────────
    cheek_w     = dist(L[LEFT_CHEEK], L[RIGHT_CHEEK]) / face_h
    features += [cheek_w]

    # ── COMBINED RATIOS ───────────────────────────────────────────────────────
    # Eye-to-mouth ratio
    eye_mouth   = dist(L[LEFT_EYE_BOT], L[MOUTH_TOP]) / face_h
    # Brow-to-mouth ratio
    brow_mouth  = dist(L[LEFT_BROW_MID], L[MOUTH_TOP]) / face_h
    features += [eye_mouth, brow_mouth]

    # ── RAW NORMALIZED COORDS for key points ─────────────────────────────────
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

# ── AUGMENTATION ─────────────────────────────────────────────────────────────
def augment_features(f):
    f = f.copy()
    f += np.random.normal(0, 0.005, f.shape)
    scale = np.random.uniform(0.95, 1.05)
    f *= scale
    return f

# ── EXTRACT FROM IMAGE ────────────────────────────────────────────────────────
def extract_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Haar cascade crop
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        pad = int(0.2 * w)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    try:
        return extract_features(lm)
    except Exception:
        return None

# ── COLLECT IMAGE PATHS ───────────────────────────────────────────────────────
print("Scanning datasets...")
emotion_images = {e: [] for e in EMOTIONS}

for emotion in EMOTIONS:
    folder = os.path.join(FER_TRAIN_DIR, emotion)
    if os.path.exists(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg','.png','.jpeg'))]
        emotion_images[emotion].extend(files)
        print(f"  FER2013 {emotion}: {len(files)} images")

for folder_num, emotion in RAFDB_MAP.items():
    folder = os.path.join(RAFDB_TRAIN, folder_num)
    if os.path.exists(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg','.png','.jpeg'))]
        emotion_images[emotion].extend(files)
        print(f"  RAF-DB  {emotion} (folder {folder_num}): {len(files)} images")

for emotion in EMOTIONS:
    print(f"Total {emotion}: {len(emotion_images[emotion])} images")

# ── EXTRACT FEATURES PER EMOTION ─────────────────────────────────────────────
FEATURE_DIM = None

for emotion in EMOTIONS:
    print(f"\nExtracting features for {emotion.upper()}...")
    files = emotion_images[emotion]
    np.random.shuffle(files)

    features_list = []
    failed = 0

    for i, fpath in enumerate(files):
        if i % 200 == 0:
            print(f"  {i}/{len(files)} | extracted: {len(features_list)}")
        feat = extract_from_image(fpath)
        if feat is not None:
            features_list.append(feat)
            if FEATURE_DIM is None:
                FEATURE_DIM = len(feat)
        else:
            failed += 1
        if len(features_list) >= TARGET_SAMPLES:
            break

    print(f"  Extracted: {len(features_list)} | Failed: {failed}")

    if len(features_list) == 0:
        print(f"  ERROR: No features for {emotion}!")
        continue

    features_arr = np.array(features_list)
    augmented    = list(features_arr)
    needed       = TARGET_SAMPLES - len(features_arr)
    if needed > 0:
        print(f"  Augmenting {len(features_arr)} -> {TARGET_SAMPLES}...")
        for i in range(needed):
            augmented.append(augment_features(features_arr[i % len(features_arr)]))

    augmented = np.array(augmented[:TARGET_SAMPLES])
    np.random.shuffle(augmented)
    np.save(os.path.join(SAVE_DIR, f"{emotion}.npy"), augmented)
    print(f"  Saved {len(augmented)} samples -> {emotion}.npy")

print(f"\nFeature dimension: {FEATURE_DIM}")
print("Done! Run train_emotion_ensemble.py next")