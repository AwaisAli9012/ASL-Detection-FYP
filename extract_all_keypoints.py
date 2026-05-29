import os
import cv2
import json
import numpy as np
import mediapipe as mp

INPUT_DIR   = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\frames_20"
OUTPUT_DIR  = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\keypoints"
LABELS_PATH = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels.json"

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_keypoints = []
all_labels    = []
label_list    = []

classes = sorted(os.listdir(INPUT_DIR))
print(f"Total classes found: {len(classes)}\n")

for idx, cls in enumerate(classes):
    cls_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    label_list.append(cls)
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

            all_keypoints.append(keypoints)
            all_labels.append(idx)
            found += 1

    print(f"✅ [{idx+1}/{len(classes)}] {cls}: {found} keypoints extracted")

hands.close()

# save
np.save(os.path.join(OUTPUT_DIR, 'keypoints.npy'), np.array(all_keypoints))
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'),    np.array(all_labels))

with open(LABELS_PATH, 'w') as f:
    json.dump(label_list, f)

print(f"\n✅ Total samples:  {len(all_keypoints)}")
print(f"✅ Total classes:  {len(label_list)}")
print(f"✅ Labels: {label_list}")
print(f"✅ Saved to: {OUTPUT_DIR}")