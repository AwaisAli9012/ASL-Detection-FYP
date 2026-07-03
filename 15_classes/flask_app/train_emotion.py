import os
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ── PATHS ─────────────────────────────────────────────────────────────────────
KEYPOINTS_DIR = "emotion_keypoints"
SAVE_DIR      = "emotion_models"
os.makedirs(SAVE_DIR, exist_ok=True)

EMOTIONS  = ['angry', 'happy', 'sad']
N_FOLDS   = 5
INPUT_DIM = 52

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading keypoints...")
X_list, y_list = [], []
for label, emotion in enumerate(EMOTIONS):
    path = os.path.join(KEYPOINTS_DIR, f"{emotion}.npy")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        exit()
    data = np.load(path)
    print(f"  {emotion}: {len(data)} samples")
    X_list.append(data)
    y_list.append(np.full(len(data), label))

X = np.vstack(X_list)
y = np.concatenate(y_list)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"\nTotal: {len(X)} samples, {X.shape[1]} features, {len(EMOTIONS)} classes")

# ── 5-FOLD CROSS VALIDATION ───────────────────────────────────────────────────
print(f"\nRunning {N_FOLDS}-Fold Stratified Cross Validation...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

rf_scores  = []
svm_scores = []
all_rf_probs  = np.zeros((len(X), len(EMOTIONS)))
all_svm_probs = np.zeros((len(X), len(EMOTIONS)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)
    rf_acc   = accuracy_score(y_val, np.argmax(rf_probs, axis=1))
    rf_scores.append(rf_acc)
    all_rf_probs[val_idx] = rf_probs
    print(f"    RF  accuracy: {rf_acc:.2%}")

    # SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_probs = svm.predict_proba(X_val)
    svm_acc   = accuracy_score(y_val, np.argmax(svm_probs, axis=1))
    svm_scores.append(svm_acc)
    all_svm_probs[val_idx] = svm_probs
    print(f"    SVM accuracy: {svm_acc:.2%}")

print(f"\nRF  mean CV: {np.mean(rf_scores):.2%}")
print(f"SVM mean CV: {np.mean(svm_scores):.2%}")

# ── TRAIN META-LEARNER ────────────────────────────────────────────────────────
print("\nTraining meta-learner (stacking RF + SVM)...")
meta_X = np.hstack([all_rf_probs, all_svm_probs])
meta   = LogisticRegression(max_iter=1000, C=1.0)
meta.fit(meta_X, y)
meta_preds = meta.predict(meta_X)
meta_acc   = accuracy_score(y, meta_preds)
print(f"Meta-learner accuracy: {meta_acc:.2%}")

# ── TRAIN FINAL MODELS ON FULL DATA ──────────────────────────────────────────
print("\nTraining final RF on full dataset...")
final_rf  = RandomForestClassifier(n_estimators=500, max_depth=20,
                                   random_state=42, n_jobs=-1)
final_rf.fit(X, y)

print("Training final SVM on full dataset...")
final_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
final_svm.fit(X, y)

# ── SAVE ──────────────────────────────────────────────────────────────────────
rf_path      = os.path.join(SAVE_DIR, "emotion_rf.pkl")
svm_path     = os.path.join(SAVE_DIR, "emotion_svm.pkl")
meta_path    = os.path.join(SAVE_DIR, "emotion_meta.pkl")
scaler_path  = os.path.join(SAVE_DIR, "emotion_scaler.pkl")
labels_path  = os.path.join(SAVE_DIR, "emotion_labels.json")

pickle.dump(final_rf,  open(rf_path,     'wb'))
pickle.dump(final_svm, open(svm_path,    'wb'))
pickle.dump(meta,      open(meta_path,   'wb'))
pickle.dump(scaler,    open(scaler_path, 'wb'))
with open(labels_path, 'w') as f:
    json.dump(EMOTIONS, f)

print(f"\nSaved all models to emotion_models/")
print(f"\nFinal CV Summary:")
print(f"  RF  : {np.mean(rf_scores):.2%}")
print(f"  SVM : {np.mean(svm_scores):.2%}")
print(f"  Meta: {meta_acc:.2%}")
print("\nDone!")