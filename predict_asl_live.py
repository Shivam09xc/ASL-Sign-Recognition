import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import os

MODEL_PATH = "model/asl_landmarks_mlp.h5"
LABELS_PATH = "model/labels.txt"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
DOT = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
LINE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)

def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def to_features(hl):
    pts = np.array([[lm.x, lm.y] for lm in hl.landmark], dtype=np.float32)
    origin = pts[0].copy()
    pts_rel = pts - origin
    scale = np.max(np.linalg.norm(pts_rel, axis=1))
    if scale < 1e-6: return None
    pts_rel /= scale
    return pts_rel.flatten().reshape(1, -1)  # (1,42)

def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        print("Model/labels not found. Train first.")
        return
    model = load_model(MODEL_PATH)
    labels = load_labels()

    cap = cv2.VideoCapture(0)
    conf_thresh = 0.60
    last = time.time(); fps = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                         min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label_txt = "No hand"
            label_conf = 0.0

            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS, DOT, LINE)
                feat = to_features(hl)
                if feat is not None:
                    pred = model.predict(feat, verbose=0)[0]
                    ci = int(np.argmax(pred))
                    label_conf = float(np.max(pred))
                    if label_conf >= conf_thresh:
                        label_txt = f"{labels[ci]} ({label_conf*100:.0f}%)"
                    else:
                        label_txt = "Unknown"

            # FPS
            now = time.time()
            if now - last >= 0.5:
                fps = 1.0 / max(1e-6, (now - last))
                last = now

            cv2.putText(frame, f"Pred: {label_txt}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}  Thr:{int(conf_thresh*100)}%", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(frame, "Keys: +/- conf | Q quit", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("ASL Live (RED landmarks)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('+') or k == ord('='):
                conf_thresh = min(0.95, conf_thresh + 0.05)
            elif k == ord('-') or k == ord('_'):
                conf_thresh = max(0.40, conf_thresh - 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
