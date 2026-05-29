import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── PATHS ───────────────────────────────────────────────
FRAMES_DIR  = r"C:\Users\Abdullah\Documents\MyWork\FYP\Dataset\frames"
MODEL_SAVE  = r"C:\Users\Abdullah\Documents\MyWork\FYP\NEW\asl_model.h5"

# ─── LOAD MODEL ──────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_SAVE)

# ─── VALIDATION DATA ─────────────────────────────────────
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_gen = val_datagen.flow_from_directory(
    FRAMES_DIR,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ─── EVALUATE ────────────────────────────────────────────
print("\nEvaluating model on validation set...")
loss, accuracy = model.evaluate(val_gen, verbose=1)

print(f"\n✅ Validation Loss:     {loss:.4f}")
print(f"✅ Validation Accuracy: {accuracy*100:.2f}%")