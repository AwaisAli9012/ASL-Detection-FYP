import cv2
import os
import shutil
import mediapipe as mp

# ─── PATHS ───────────────────────────────────────────────
FRAMES_DIR   = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\frames"
FILTERED_DIR = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\frames_filtered"

# ─── MEDIAPIPE SETUP ─────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

# ─── FILTER ──────────────────────────────────────────────
total_kept    = 0
total_removed = 0

classes = sorted(os.listdir(FRAMES_DIR))
print(f"Processing {len(classes)} classes...\n")

for idx, class_name in enumerate(classes):
    class_path    = os.path.join(FRAMES_DIR, class_name)
    filtered_path = os.path.join(FILTERED_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    os.makedirs(filtered_path, exist_ok=True)

    images = os.listdir(class_path)
    kept    = 0
    removed = 0

    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # convert to RGB for mediapipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            # hand detected — keep this frame
            shutil.copy(img_path, os.path.join(filtered_path, img_name))
            kept += 1
        else:
            removed += 1

    total_kept    += kept
    total_removed += removed
    print(f"[{idx+1}/{len(classes)}] {class_name}: kept {kept}, removed {removed}")

hands.close()

print(f"\n✅ Filtering complete!")
print(f"✅ Total kept:    {total_kept}")
print(f"✅ Total removed: {total_removed}")
print(f"📁 Filtered frames saved to: {FILTERED_DIR}")