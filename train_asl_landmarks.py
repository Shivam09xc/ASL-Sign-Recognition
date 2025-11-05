import os, glob, json
import numpy as np
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

DATA_DIR = "dataset"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

mp_hands = mp.solutions.hands

def extract_features(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks: return None
        hl = res.multi_hand_landmarks[0]
        # build coords
        pts = np.array([[lm.x, lm.y] for lm in hl.landmark], dtype=np.float32)  # 21x2 normalized [0..1]
        # relative to wrist (index 0)
        origin = pts[0].copy()
        pts_rel = pts - origin
        # scale normalize by max |x|/|y|
        scale = np.max(np.linalg.norm(pts_rel, axis=1))
        if scale < 1e-6: return None
        pts_rel /= scale
        return pts_rel.flatten()  # 42 dims

def load_dataset():
    X, y, labels = [], [], []
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not classes: raise RuntimeError("No class folders found under 'dataset/'.")
    print("Classes:", classes)
    for ci, cls in enumerate(classes):
        files = glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, cls, "*.png"))
        kept = 0
        for f in files:
            img = cv2.imread(f)
            if img is None: continue
            feat = extract_features(img)
            if feat is None: continue
            X.append(feat); y.append(ci); kept += 1
        print(f"{cls}: {kept} samples")
        labels.append(cls)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels

def build_model(num_classes, in_dim=42):
    m = Sequential([
        Input(shape=(in_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def main():
    X, y, labels = load_dataset()
    if len(X) == 0: raise RuntimeError("No landmarks extracted. Check your dataset images.")
    num_classes = len(labels)
    y_cat = to_categorical(y, num_classes)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

    model = build_model(num_classes)
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy')
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                        epochs=200, batch_size=64, callbacks=[es], verbose=1)

    model.save(os.path.join(MODEL_DIR, "asl_landmarks_mlp.h5"))
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        for lbl in labels: f.write(lbl + "\n")

    print("Saved model/model files.")
    # quick report
    va_acc = model.evaluate(X_va, y_va, verbose=0)[1]
    print(f"Validation accuracy: {va_acc:.3f}")

if __name__ == "__main__":
    main()
