import cv2
import mediapipe as mp
import os
from datetime import datetime

# Drawing with RED dots/lines
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
DOT = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
LINE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)

def safe_roi(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1: return None
    return frame[y1:y2, x1:x2]

def main():
    sign = input("Enter ASL letter (A–Z): ").strip().upper()
    os.makedirs(f"dataset/{sign}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    counter, saved = 0, 0
    autosave = True         # auto-save every valid frame
    target = 200            # images per class

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                         min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                h, w = frame.shape[:2]
                hl = res.multi_hand_landmarks[0]
                # draw landmarks in RED
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS, DOT, LINE)

                xs = [int(pt.x * w) for pt in hl.landmark]
                ys = [int(pt.y * h) for pt in hl.landmark]
                x1, x2 = min(xs)-30, max(xs)+30
                y1, y2 = min(ys)-30, max(ys)+30

                crop = safe_roi(frame, x1, y1, x2, y2)
                if crop is not None and autosave and saved < target:
                    # Save a clean crop (no drawings)
                    raw = safe_roi(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), x1, y1, x2, y2)
                    if raw is not None:
                        fname = f"dataset/{sign}/{sign}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                        cv2.imwrite(fname, raw)
                        saved += 1

                # show crop box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

            cv2.putText(frame, f"Sign:{sign}  Saved:{saved}/{target}  [A]utosave:{'ON' if autosave else 'OFF'}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Press A toggle autosave | C manual capture | Q quit",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            cv2.imshow("ASL Capture (RED dots/lines)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                autosave = not autosave
            elif key == ord('c') and crop is not None:
                raw = safe_roi(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), x1, y1, x2, y2)
                if raw is not None:
                    fname = f"dataset/{sign}/{sign}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    cv2.imwrite(fname, raw); saved += 1
            elif key == ord('q') or (saved >= target):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images for {sign}")

if __name__ == "__main__":
    main()
