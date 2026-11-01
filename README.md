# Driver Drowsiness Detection System

A webcam-based prototype that detects prolonged eye closure using facial landmarks and the Eye Aspect Ratio (EAR). An optional CNN eye-state classifier can be used with EAR for combined detection. An audio alert is triggered after sustained closure.

## Features
- Real-time webcam processing with OpenCV
- dlib 68-point facial landmarks and eye-region visualization
- EAR threshold + consecutive-frame rule to reduce single-frame false alarms
- Optional CNN classifier for open/closed eye crops
- Audio alert with configurable threshold and frame duration

> **Safety note:** This is an educational prototype, not a certified driver-safety product. Do not use it as a substitute for rest or a production vehicle system. Webcam-based results vary with lighting, camera angle, glasses, and face visibility.

## Project layout
```text
driver-drowsiness-detection/
├── main.py
├── eye_cnn.py
├── train_cnn.py
├── requirements.txt
├── models/                 # add downloaded landmark file and trained CNN here
├── assets/                 # optional alarm.wav
└── dataset/                # local training data; do not commit datasets
```

## Setup (Windows)
Use Python 3.10 or 3.11 in a virtual environment. dlib installation can be platform-sensitive; if `pip install dlib` fails, use a compatible prebuilt wheel or Conda environment.

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the official dlib `shape_predictor_68_face_landmarks.dat` model and place it at:
`models/shape_predictor_68_face_landmarks.dat`

The file is large and has its own license/terms; do not commit it without checking redistribution permissions. Obtain it from the dlib project: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## Run landmark + EAR detector
```bash
python main.py
```
Adjust sensitivity:
```bash
python main.py --ear-threshold 0.22 --closed-frames 18
```
Press `Q` to quit.

## Optional CNN training
Prepare eye crop images locally in this structure:
```text
dataset/
├── train/
│   ├── open/
│   └── closed/
└── val/
    ├── open/
    └── closed/
```
Train and save the model:
```bash
python train_cnn.py
```
Then run the combined detector:
```bash
python main.py --use-cnn
```
The CNN training code maps `open` to 0 and `closed` to 1. Use a suitable, legally obtained dataset and keep it out of GitHub unless its license permits redistribution.

## Evaluation and honest reporting
Measure on a held-out subject/video set. Report confusion matrix, precision, recall, F1, and accuracy; avoid evaluating on frames from the same recording used for training. The `92%` figure in a resume should only be claimed if you can reproduce it with documented test data and methodology. EAR threshold is a heuristic, not a trained model.

## GitHub
```bash
git init
git add .
git commit -m "Build driver drowsiness detection prototype"
```
Create an empty repository on GitHub, then:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git
git push -u origin main
```
