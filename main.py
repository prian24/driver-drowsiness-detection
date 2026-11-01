"""
Real-time Driver Drowsiness Detection
Uses dlib facial landmarks + Eye Aspect Ratio (EAR), with optional CNN eye-state model.
"""
import argparse, time
from pathlib import Path
import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound import playsound
from threading import Thread

from eye_cnn import load_eye_model, predict_eye_state

ALARM_PATH = Path("assets/alarm.wav")
PREDICTOR_PATH = Path("models/shape_predictor_68_face_landmarks.dat")
CNN_PATH = Path("models/eye_state_cnn.keras")

def ear(eye):
    # Eye Aspect Ratio: vertical distances / twice horizontal distance
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return (a + b) / (2.0 * c) if c else 0.0

def alarm_loop():
    if ALARM_PATH.exists():
        try:
            playsound(str(ALARM_PATH))
        except Exception as exc:
            print("Alarm playback failed:", exc)
    else:
        print("\aDROWSINESS ALERT! (Add assets/alarm.wav for audio alarm)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ear-threshold", type=float, default=0.23)
    parser.add_argument("--closed-frames", type=int, default=18)
    parser.add_argument("--use-cnn", action="store_true", help="Use trained CNN model alongside EAR")
    args = parser.parse_args()

    if not PREDICTOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTOR_PATH}. Download dlib's 68-point landmark predictor; "
            "see README.md for the official source and setup."
        )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    left_idx, right_idx = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cnn = load_eye_model(CNN_PATH) if args.use_cnn else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index/permissions.")
    closed_count, alarm_active = 0, False
    last_alarm = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            status, score = "AWAKE", 0.0

            for face in faces:
                shape = predictor(gray, face)
                points = face_utils.shape_to_np(shape)
                le, re = points[left_idx[0]:left_idx[1]], points[right_idx[0]:right_idx[1]]
                score = (ear(le) + ear(re)) / 2.0
                closed = score < args.ear_threshold

                # Optional CNN confirms eye state from aligned eye crops.
                if cnn is not None:
                    crops = []
                    for eye_points in (le, re):
                        x1, y1 = eye_points.min(axis=0); x2, y2 = eye_points.max(axis=0)
                        crop = gray[max(0,y1-5):min(gray.shape[0],y2+5), max(0,x1-5):min(gray.shape[1],x2+5)]
                        if crop.size:
                            crops.append(crop)
                    if crops:
                        cnn_closed = any(predict_eye_state(cnn, crop) == "closed" for crop in crops)
                        closed = closed and cnn_closed

                closed_count = closed_count + 1 if closed else 0
                status = "DROWSY" if closed_count >= args.closed_frames else "AWAKE"
                x, y = face.left(), face.top()
                cv2.rectangle(frame, (x,y), (face.right(),face.bottom()), (0,0,255) if status=="DROWSY" else (0,255,0), 2)
                cv2.putText(frame, f"{status} | EAR: {score:.2f}", (x, max(25,y-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255) if status=="DROWSY" else (0,255,0), 2)
                cv2.polylines(frame, [le, re], True, (255, 180, 0), 1)

            if status == "DROWSY" and time.time() - last_alarm > 2.0:
                Thread(target=alarm_loop, daemon=True).start()
                last_alarm = time.time()
            cv2.imshow("Driver Drowsiness Detection | Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
