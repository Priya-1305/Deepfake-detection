from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import shutil
import os
import uuid

from .inference import predict_video, model  # model imported to load once

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Deepfake Detection API")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title=APP_NAME)

# --------------------------------------------------
# Startup event (model loads once)
# --------------------------------------------------
@app.on_event("startup")
def startup_event():
    print("Model loaded. API is ready.")

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    # Save uploaded video
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    score = predict_video(temp_path)

    # Cleanup
    os.remove(temp_path)

    if score is None:
        raise HTTPException(status_code=400, detail="No face detected in video")

    label = "fake" if score > THRESHOLD else "real"

    return JSONResponse({
        "prediction": label,
        "confidence": round(score, 3)
    })
