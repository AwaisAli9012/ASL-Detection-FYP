import os
import cv2
import json
import numpy as np
import mediapipe as mp

# --- PATHS ---
INPUT_DIR   = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\frames_20"
OUTPUT_DIR  = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\keypoints_15_v2"
LABELS_PATH = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels_15_v2.json"

# --- CLASSES ---
CLASSES_15 = [
    'help', 'yes', 'no', 'drink', 'go',
    'finish', 'play', 'mother', 'computer', 'eat',
    'want', 'who', 'family', 'like', 'enjoy'
]

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_keypoints = []
all_labels    = []

print("Extracting keypoints for 15 classes with flipped augmentation...\n")

for idx, cls in enumerate(CLASSES_15):
    cls_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(cls_path):
        print(f"[NOT FOUND] {cls}")
        continue

    images = os.listdir(cls_path)
    found  = 0

    for img_name in images:
        img_path = os.path.join(cls_path, img_name)
        img      = cv2.imread(img_path)
        if img is None:
            continue

        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            keypoints = []
            for hand_idx in range(2):
                if hand_idx < len(result.multi_hand_landmarks):
                    lm = result.multi_hand_landmarks[hand_idx]
                    for point in lm.landmark:
                        keypoints.extend([point.x, point.y, point.z])
                else:
                    keypoints.extend([0.0] * 63)

            # Original
            all_keypoints.append(keypoints)
            all_labels.append(idx)

            # Flipped (mirror X coordinate)
            flipped = keypoints.copy()
            for i in range(0, len(flipped), 3):
                flipped[i] = 1.0 - flipped[i]
            all_keypoints.append(flipped)
            all_labels.append(idx)

            found += 1

    print(f"[{idx+1}/15] {cls}: {found} original + {found} flipped = {found*2} total")

hands.close()

# --- SAVE ---
np.save(os.path.join(OUTPUT_DIR, 'keypoints.npy'), np.array(all_keypoints))
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'),    np.array(all_labels))

with open(LABELS_PATH, 'w') as f:
    json.dump(CLASSES_15, f)

print(f"\nTotal samples: {len(all_keypoints)}")
print(f"Total classes: 15")
print(f"Saved to: {OUTPUT_DIR}")