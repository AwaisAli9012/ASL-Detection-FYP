import os
import json
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize

# ── CONFIGURE PATHS (LINUX VERSION) ──────────────────────────────────────────────
# For Linux: /home/xero1/Documents/ASL-Detection-FYP/15_classes

BASE_DIR      = Path(__file__).resolve().parent  # /home/xero1/.../15_classes
REPO_ROOT     = BASE_DIR.parent                   # /home/xero1/.../ASL-Detection-FYP

# Linux paths (adjusted from Windows)
DATASET_DIR   = REPO_ROOT / "Dataset"
KEYPOINTS_DIR = str(DATASET_DIR / "keypoints_15_v4")
MODELS_DIR    = str(REPO_ROOT / "Models")

# Create Models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

NN_SAVE      = os.path.join(MODELS_DIR, "keypoint_model_15_v4_ensemble_nn.h5")
RF_SAVE      = os.path.join(MODELS_DIR, "keypoint_model_15_v4_rf.pkl")
META_SAVE    = os.path.join(MODELS_DIR, "keypoint_model_15_v4_meta.pkl")
LABELS_PATH  = os.path.join(MODELS_DIR, "keypoint_labels_15_v4.json")
RESULTS_PATH = os.path.join(MODELS_DIR, "ensemble_results.json")

print("=" * 70)
print("  PATH CONFIGURATION")
print("=" * 70)
print(f"BASE_DIR      : {BASE_DIR}")
print(f"REPO_ROOT     : {REPO_ROOT}")
print(f"KEYPOINTS_DIR : {KEYPOINTS_DIR}")
print(f"MODELS_DIR    : {MODELS_DIR}")
print("=" * 70)

# ── SETTINGS ─────────────────────────────────────────────────────────────────
N_FOLDS    = 5       # StratifiedKFold splits
RF_TREES   = 300     # Random Forest estimators
EPOCHS     = 150     # max epochs per fold
BATCH_SIZE = 32
PATIENCE   = 15      # EarlyStopping patience


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SHUFFLE DATA
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  ASL Ensemble Trainer  (NN + RF + K-Fold)")
print("=" * 60)

# Check if data exists
keypoints_file = os.path.join(KEYPOINTS_DIR, "keypoints.npy")
labels_file = os.path.join(KEYPOINTS_DIR, "labels.npy")

if not os.path.exists(keypoints_file):
    print(f"ERROR: {keypoints_file} not found!")
    print(f"Please run ex15class.py first to generate keypoints")
    exit(1)

X = np.load(keypoints_file)
y = np.load(labels_file)

# CRITICAL FIX: Shuffle indices globally so that Keras validation_split 
# receives a globally balanced mixture of classes instead of a single blocked class.
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

with open(LABELS_PATH) as f:
    labels = json.load(f)

NUM_CLASSES  = len(labels)
INPUT_DIM    = X.shape[1]

print(f"\nLoaded  {len(X)} samples | {NUM_CLASSES} classes | {INPUT_DIM} features")
print(f"Classes: {labels}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 2. HELPER — build a fresh NN
# ═════════════════════════════════════════════════════════════════════════════
def build_nn(input_dim, num_classes):
    m = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2), 
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2), 
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1), 
        Dense(num_classes, activation='softmax')
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return m


# ═════════════════════════════════════════════════════════════════════════════
# 3. K-FOLD CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_nn_probs = np.zeros((len(X), NUM_CLASSES))
oof_rf_probs = np.zeros((len(X), NUM_CLASSES))

fold_results = []
best_nn_val_acc = 0.0
best_nn_model   = None

print(f"Running {N_FOLDS}-Fold Stratified Cross-Validation...\n")
print("-" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n── Fold {fold}/{N_FOLDS} ────────────────────────────")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)

    # ── 3a. Train NN ──────────────────────────────────────────────────────
    nn = build_nn(INPUT_DIM, NUM_CLASSES)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=0)
    ]
    history = nn.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )
    nn_val_acc = max(history.history['val_accuracy'])
    nn_probs   = nn.predict(X_val, verbose=0)
    oof_nn_probs[val_idx] = nn_probs

    nn_preds   = np.argmax(nn_probs, axis=1)
    nn_acc     = accuracy_score(y_val, nn_preds)

    print(f"  NN   val accuracy : {nn_acc*100:.2f}%  (best epoch val_acc: {nn_val_acc*100:.2f}%)")

    if nn_val_acc > best_nn_val_acc:
        best_nn_val_acc = nn_val_acc
        best_nn_model   = nn

    # ── 3b. Train Random Forest ───────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)
    oof_rf_probs[val_idx] = rf_probs

    rf_preds = rf.predict(X_val)
    rf_acc   = accuracy_score(y_val, rf_preds)
    print(f"  RF   val accuracy : {rf_acc*100:.2f}%")

    # ── 3c. Soft-vote ensemble ────────────────────────────────────────────
    avg_probs    = (nn_probs + rf_probs) / 2.0
    ens_preds    = np.argmax(avg_probs, axis=1)
    ens_acc      = accuracy_score(y_val, ens_preds)
    print(f"  Ensemble (avg)    : {ens_acc*100:.2f}%")

    fold_results.append({
        "fold":     fold,
        "nn_acc":   round(nn_acc  * 100, 2),
        "rf_acc":   round(rf_acc  * 100, 2),
        "ens_acc":  round(ens_acc * 100, 2),
    })


# ═════════════════════════════════════════════════════════════════════════════
# 4. FINAL MODELS — train on ALL data
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Training final models on ALL data...")
print("=" * 60)

# ── Final NN ─────────────────────────────────────────────────────────────────
y_all_cat = tf.keras.utils.to_categorical(y, NUM_CLASSES)
final_nn  = build_nn(INPUT_DIM, NUM_CLASSES)
final_nn.fit(
    X, y_all_cat,
    validation_split=0.1, 
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=0)
    ],
    verbose=1
)
final_nn.save(NN_SAVE)
print(f"\nNN saved  → {NN_SAVE}")

# ── Final RF ─────────────────────────────────────────────────────────────────
print("\nTraining final Random Forest...")
final_rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=-1, random_state=42)
final_rf.fit(X, y)
with open(RF_SAVE, 'wb') as f:
    pickle.dump(final_rf, f)
print(f"RF saved  → {RF_SAVE}")

# ── Meta-Learner ─────────────────────────────────────────────────────────────
print("\nTraining meta-learner on out-of-fold predictions...")
oof_stack = np.hstack([oof_nn_probs, oof_rf_probs])   
meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
meta.fit(oof_stack, y)
meta_preds    = meta.predict(oof_stack)
meta_oof_acc  = accuracy_score(y, meta_preds)
print(f"Meta-learner OOF accuracy: {meta_oof_acc*100:.2f}%")
with open(META_SAVE, 'wb') as f:
    pickle.dump(meta, f)
print(f"Meta saved → {META_SAVE}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
nn_accs  = [r['nn_acc']  for r in fold_results]
rf_accs  = [r['rf_acc']  for r in fold_results]
ens_accs = [r['ens_acc'] for r in fold_results]

print("\n" + "=" * 60)
print("  K-FOLD SUMMARY")
print("=" * 60)
print(f"{'Fold':<6} {'NN':>8} {'RF':>8} {'Ensemble':>10}")
print("-" * 36)
for r in fold_results:
    print(f"{r['fold']:<6} {r['nn_acc']:>7.2f}% {r['rf_acc']:>7.2f}% {r['ens_acc']:>9.2f}%")
print("-" * 36)
print(f"{'Mean':<6} {np.mean(nn_accs):>7.2f}% {np.mean(rf_accs):>7.2f}% {np.mean(ens_accs):>9.2f}%")
print(f"{'Std':<6} {np.std(nn_accs):>7.2f}% {np.std(rf_accs):>7.2f}% {np.std(ens_accs):>9.2f}%")
print(f"\nMeta-learner OOF accuracy : {meta_oof_acc*100:.2f}%")

results_summary = {
    "folds":              fold_results,
    "nn_mean_acc":        round(float(np.mean(nn_accs)),  2),
    "rf_mean_acc":        round(float(np.mean(rf_accs)),  2),
    "ensemble_mean_acc":  round(float(np.mean(ens_accs)), 2),
    "meta_oof_acc":       round(meta_oof_acc * 100,       2),
    "saved_models": {
        "nn":   NN_SAVE,
        "rf":   RF_SAVE,
        "meta": META_SAVE,
    }
}
with open(RESULTS_PATH, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nResults saved → {RESULTS_PATH}")
print("\nDone! Your core files are untouched.")


# ═════════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE PREDICT HELPER
# ═════════════════════════════════════════════════════════════════════════════
def ensemble_predict(keypoints_1d: np.ndarray,
                     nn_model, rf_model, meta_model=None,
                     mode: str = "soft_vote") -> tuple[int, float]:
    """
    Predict using ensemble of NN + RF + Meta-learner
    
    Args:
        keypoints_1d: 1D array of hand keypoints (126 dimensions)
        nn_model: Trained TensorFlow NN model
        rf_model: Trained scikit-learn Random Forest
        meta_model: Trained meta-learner (Logistic Regression)
        mode: "soft_vote", "stacking", "nn_only", "rf_only"
        
    Returns:
        (class_idx, confidence): Predicted class and confidence
    """
    x = keypoints_1d.reshape(1, -1)
    nn_probs = nn_model.predict(x, verbose=0)[0]
    rf_probs = rf_model.predict_proba(x)[0]

    if mode == "soft_vote":
        probs = (nn_probs + rf_probs) / 2.0
    elif mode == "stacking" and meta_model is not None:
        stack = np.hstack([nn_probs, rf_probs]).reshape(1, -1)
        probs = meta_model.predict_proba(stack)[0]
    elif mode == "nn_only":
        probs = nn_probs
    elif mode == "rf_only":
        probs = rf_probs
    else:
        probs = (nn_probs + rf_probs) / 2.0

    class_idx  = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return class_idx, confidence