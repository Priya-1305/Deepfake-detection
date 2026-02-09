import sys
from app.inference import predict_video

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    score = predict_video(video_path)

    if score is None:
        print("No valid face detected in video")
        sys.exit(1)


    label = "fake" if score > 0.5 else "real"

    print({
        "prediction": label,
        "confidence": round(score, 3)
    })
