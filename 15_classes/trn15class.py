import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- PATHS ---
KEYPOINTS_DIR = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\keypoints_15_v2"
MODEL_SAVE    = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_model_15_v2.h5"
LABELS_PATH   = r"C:\Users\Abdullah\Documents\MyWork\FYP\Models\keypoint_labels_15_v2.json"

# --- LOAD DATA ---
print("Loading keypoints...")
X = np.load(os.path.join(KEYPOINTS_DIR, 'keypoints.npy'))
y = np.load(os.path.join(KEYPOINTS_DIR, 'labels.npy'))

with open(LABELS_PATH) as f:
    labels = json.load(f)

NUM_CLASSES = len(labels)
print(f"Total samples:  {len(X)}")
print(f"Total classes:  {NUM_CLASSES}")
print(f"Labels: {labels}\n")

# --- PREPARE DATA ---
y_cat = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}\n")

# --- BUILD MODEL ---
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- CALLBACKS ---
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-7, verbose=1)
]

# --- TRAIN ---
print("Starting training...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print(f"\nModel saved to: {MODEL_SAVE}")
print("Training complete!")