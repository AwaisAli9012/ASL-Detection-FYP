import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EMOTIONS = ['angry', 'happy', 'sad']
SAVE_DIR = "my_faces"
SAMPLES  = 200

for emotion in EMOTIONS:
    os.makedirs(f"{SAVE_DIR}/{emotion}", exist_ok=True)

cap = cv2.VideoCapture(0)

for emotion in EMOTIONS:
    input(f"\nPress ENTER when ready to make a {emotion.upper()} face...")
    count = 0
    print(f"Hold your expression! Collecting {SAMPLES} samples...")
    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60,60))
        for (x,y,w,h) in faces:
            cv2.imwrite(f"{SAVE_DIR}/{emotion}/{count}.jpg", frame[y:y+h, x:x+w])
            count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{emotion}: {count}/{SAMPLES}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collecting", frame)
        cv2.waitKey(1)
    print(f"Done {emotion}!")

cap.release()
cv2.destroyAllWindows()
print("\nDone! Switch to extract_env and run ex_emotion.py")