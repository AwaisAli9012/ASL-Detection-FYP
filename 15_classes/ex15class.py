import os
import cv2
import json
import numpy as np
import mediapipe as mp

# --- PATHS ---
INPUT_DIR   = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\frames_20"
OUTPUT_DIR  = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\keypoints_15_v3"
LABELS_PATH = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels_15_v3.json"

# --- CLASSES ---
CLASSES_15 = [
    'help', 'yes', 'no', 'before', 'go',
    'finish', 'play', 'mother', 'computer', 'cool',
    'want', 'who', 'family', 'like', 'enjoy'
]

TARGET_PER_CLASS = 300

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

def augment_keypoints(keypoints):
    augmented = []
    kp = np.array(keypoints)

    # Small random noise
    noise = kp + np.random.normal(0, 0.005, kp.shape)
    augmented.append(noise.tolist())

    # Slight scale variation
    scale = kp * np.random.uniform(0.95, 1.05)
    augmented.append(scale.tolist())

    # Slight translation
    translation = kp.copy()
    translation[0::3] += np.random.uniform(-0.02, 0.02)
    translation[1::3] += np.random.uniform(-0.02, 0.02)
    augmented.append(translation.tolist())

    return augmented

print("Extracting keypoints for 15 classes with balanced augmentation...\n")

for idx, cls in enumerate(CLASSES_15):
    cls_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(cls_path):
        print(f"[NOT FOUND] {cls}")
        continue

    images = os.listdir(cls_path)
    class_keypoints = []

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
            class_keypoints.append(keypoints)

            # Flipped (mirror X)
            flipped = keypoints.copy()
            for i in range(0, len(flipped), 3):
                flipped[i] = 1.0 - flipped[i]
            class_keypoints.append(flipped)

    # Balance to TARGET_PER_CLASS using augmentation
    original_count = len(class_keypoints)
    while len(class_keypoints) < TARGET_PER_CLASS:
        base = class_keypoints[len(class_keypoints) % original_count]
        augmented = augment_keypoints(base)
        for aug in augmented:
            if len(class_keypoints) < TARGET_PER_CLASS:
                class_keypoints.append(aug)

    # Add to main list
    for kp in class_keypoints[:TARGET_PER_CLASS]:
        all_keypoints.append(kp)
        all_labels.append(idx)

    print(f"[{idx+1}/15] {cls}: {original_count} extracted -> {TARGET_PER_CLASS} after augmentation")

hands.close()

np.save(os.path.join(OUTPUT_DIR, 'keypoints.npy'), np.array(all_keypoints))
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'),    np.array(all_labels))

with open(LABELS_PATH, 'w') as f:
    json.dump(CLASSES_15, f)

print(f"\nTotal samples: {len(all_keypoints)}")
print(f"Total classes: 15")
print(f"Each class:    {TARGET_PER_CLASS} samples")
print(f"Saved to: {OUTPUT_DIR}")