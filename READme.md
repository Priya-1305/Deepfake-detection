# Deepfake Detection API

A production-style FastAPI service for detecting deepfake videos using
face-based inference and a pretrained EfficientNet model.

## Features
- Video upload API
- Face extraction per frame
- Aggregated video-level prediction
- REST API with Swagger UI

## Tech Stack
- Python 3.10
- PyTorch
- OpenCV
- FastAPI
- Uvicorn

## Run Locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
