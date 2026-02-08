import cv2
import torch
import numpy as np
from torchvision import transforms

from .model import load_model, DEVICE

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = load_model()

def predict_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()
            scores.append(score)

    cap.release()

    if len(scores) == 0:
        return None

    return float(np.mean(scores))
