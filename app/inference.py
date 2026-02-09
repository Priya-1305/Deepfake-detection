import cv2
import torch
import numpy as np
from torchvision import transforms
import os

from .model import load_model, DEVICE

# ---------------------------------------------------
# Path setup (robust for local + deployment)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "assets", "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------------------------------------------------
# Image preprocessing
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------
# Load model once (inference mode)
# ---------------------------------------------------
model = load_model()

# ---------------------------------------------------
# Face extraction helper
# ---------------------------------------------------
def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    return face

# ---------------------------------------------------
# Video inference
# ---------------------------------------------------
def predict_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = extract_face(frame)
        if face is None:
            continue

        inp = transform(face).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()
            scores.append(score)

    cap.release()

    if len(scores) == 0:
        return None

    return float(np.mean(scores))
