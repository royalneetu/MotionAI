"""FastAPI service for Human Activity Recognition dashboard."""
from __future__ import annotations
import base64, io, os, tempfile, time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "activity_model.pt"
LABEL_PATH = ROOT / "models" / "label_map.json"
SAMPLE_DIR = ROOT / "sample_data"
DB_PATH = ROOT / "database" / "activity_history.db"

import sys
sys.path.insert(0, str(ROOT))
from model_architecture import load_model
from preprocess import extract_frames
from database import db_manager

app = FastAPI(title="MotionAI HAR API", version="1.0.0", description="Human activity recognition and fall-alert API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_model = None
_classes: list[str] = []
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    global _model, _classes
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model checkpoint missing. Run the training instructions in README.md.")
        try:
            _model, _classes = load_model(str(MODEL_PATH), str(LABEL_PATH), device=_device)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc
    return _model, _classes


def normalize_label(label: str) -> str:
    return label.replace("_", " ")


def infer_frames(frames: np.ndarray):
    model, classes = get_model()
    tensor = torch.from_numpy(frames).float().unsqueeze(0).to(_device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    order = np.argsort(probs)[::-1]
    idx = int(order[0])
    return {
        "activity": classes[idx], "label": normalize_label(classes[idx]),
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": [{"activity": classes[int(i)], "label": normalize_label(classes[int(i)]), "confidence": round(float(probs[int(i)]) * 100, 2)} for i in order],
        "is_fall": classes[idx].lower().replace("_", " ") == "falling" and float(probs[idx]) >= 0.60,
    }


@app.on_event("startup")
def startup():
    db_manager.init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "MotionAI API", "model_file": MODEL_PATH.exists(), "device": str(_device), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/api/model")
def model_info():
    _, classes = get_model()
    return {"name": "LRCN (CNN + BiLSTM)", "classes": classes, "class_count": len(classes), "sequence_length": 16, "image_size": "64 × 64", "device": str(_device), "checkpoint": MODEL_PATH.name}


def read_logs(limit: int = 100):
    db_manager.init_db()
    conn = db_manager.get_connection()
    try:
        rows = conn.execute("SELECT * FROM activity_logs ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/dashboard")
def dashboard():
    logs = read_logs(500)
    activity_counts: dict[str, int] = {}
    for row in logs:
        activity_counts[row["activity"]] = activity_counts.get(row["activity"], 0) + 1
    total = len(logs)
    falls = sum(int(x.get("is_fall", 0)) for x in logs)
    avg = round(sum(float(x.get("confidence", 0)) for x in logs) / total, 1) if total else 0
    return {"total_predictions": total, "fall_alerts": falls, "avg_confidence": avg, "activities_seen": len(activity_counts), "activity_distribution": [{"activity": k, "label": normalize_label(k), "count": v} for k, v in sorted(activity_counts.items(), key=lambda kv: kv[1], reverse=True)], "recent": logs[:8], "model": "LRCN · CNN + BiLSTM"}


@app.get("/api/history")
def history(limit: int = Query(100, ge=1, le=500), activity: Optional[str] = None, falls_only: bool = False, search: Optional[str] = None):
    logs = read_logs(500)
    if activity and activity.lower() != "all":
        logs = [x for x in logs if x["activity"].lower() == activity.lower() or normalize_label(x["activity"]).lower() == activity.lower()]
    if falls_only:
        logs = [x for x in logs if x["is_fall"]]
    if search:
        needle = search.lower()
        logs = [x for x in logs if needle in str(x.get("source", "")).lower() or needle in str(x.get("activity", "")).lower() or needle in str(x.get("details", "")).lower()]
    return {"items": logs[:limit], "total": len(logs)}


@app.delete("/api/history")
def clear_history():
    db_manager.clear_history()
    return {"ok": True, "message": "Activity history cleared."}


@app.get("/api/samples")
def samples():
    return {"items": [{"name": p.name, "label": p.stem.replace("demo_", "").replace("_", " ").title(), "url": f"/api/samples/{p.name}"} for p in sorted(SAMPLE_DIR.glob("*.mp4"))]}


@app.get("/api/samples/{filename}")
def sample_video(filename: str):
    # Prevent path traversal; only serve files in the curated sample folder.
    path = (SAMPLE_DIR / filename).resolve()
    if path.parent != SAMPLE_DIR.resolve() or not path.is_file() or path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Sample video not found")
    return StreamingResponse(open(path, "rb"), media_type="video/mp4", headers={"Content-Disposition": f'inline; filename="{path.name}"'})


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Upload a supported video: MP4, AVI, MOV, MKV, or WEBM.")
    payload = await file.read()
    if not payload or len(payload) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Video must be non-empty and smaller than 100 MB.")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(payload); tmp_path = tmp.name
        frames = extract_frames(tmp_path, seq_len=16, img_size=(64, 64), augment=False)
        result = infer_frames(frames)
        result.update({"filename": Path(file.filename or "uploaded video").name, "size_bytes": len(payload), "duration_seconds": _video_duration(tmp_path), "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "Video Upload"})
        db_manager.log_activity(source=result["filename"], activity=result["activity"], confidence=result["confidence"], is_fall=result["is_fall"], duration_seconds=result["duration_seconds"], details="Uploaded video analysis")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyze this video: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)


def _video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return round(frames / fps, 2) if fps > 0 else 0.0


class FrameRequest(BaseModel):
    image: str  # data URL or base64 JPEG
    source: str = "Browser Webcam"


@app.post("/api/predict-frame")
def predict_frame(request: FrameRequest):
    try:
        encoded = request.image.split(",", 1)[-1]
        decoded = base64.b64decode(encoded, validate=True)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("Invalid image frame")
        # Build a temporal sequence by sampling the current frame repeatedly. This endpoint is
        # a responsive demo; meaningful motion classification needs a sequence of changing frames.
        rgb = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        one = np.transpose(rgb, (2, 0, 1))
        frames = np.repeat(one[np.newaxis, ...], 16, axis=0)
        result = infer_frames(frames)
        result["source"] = request.source
        result["note"] = "Single-frame preview repeated to match the model input. Use uploaded video for temporal activity analysis."
        db_manager.log_activity(source=request.source, activity=result["activity"], confidence=result["confidence"], is_fall=False, details="Webcam still-frame preview; not a verified temporal fall alert")
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not classify webcam frame: {exc}") from exc


@app.get("/api/metrics")
def metrics():
    report = ROOT / "outputs" / "classification_report.txt"
    report_text = report.read_text(encoding="utf-8") if report.exists() else "No evaluation report found. Run python evaluate.py."
    cm = ROOT / "outputs" / "confusion_matrix.png"
    return {"report": report_text, "confusion_matrix_available": cm.exists(), "warning": "Review dataset independence before interpreting reported scores; synthetic or duplicated clips are not evidence of real-world accuracy."}
