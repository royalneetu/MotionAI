# MotionAI — Human Activity Recognition & Fall-Pattern Review

A B.Tech CSE (AI) project for classifying six human activities from video clips using the existing PyTorch LRCN model (CNN frame features + bidirectional LSTM). The upgraded web interface uses **React + Vite** and a **FastAPI** backend, with SQLite history.

> **Safety and model-quality note:** This is an educational prototype, not a certified fall alarm or medical/emergency system. Predictions can be wrong. The included dataset audit finds byte-identical clips across train, validation and test splits. Consequently, any existing evaluation score is not a trustworthy estimate of performance on unseen videos until independent data is collected, the split is corrected and the model is retrained.

## Features

- Dark navy / blue / violet responsive React dashboard.
- Overview metrics, activity distribution, confidence trend and recent activity.
- Video upload analysis with 16-frame sampling, top prediction, per-class probabilities and SQLite logging.
- Browser webcam preview and explicit still-frame classification. Still-frame output is labelled as a preview; it is **not** temporal motion detection. Use video analysis for motion-aware classification.
- Activity history with search, class filter, fall-only filter, CSV export and confirmed history clearing.
- Model configuration and evaluation-artifact page.
- FastAPI health/model/dashboard/history/sample/video-analysis endpoints.
- Dataset hash audit utility and memory-efficient video frame sampling.

## Requirements

- Python 3.11 recommended (PyTorch/OpenCV dependencies may vary by operating system).
- Node.js 18+ and npm.
- The existing `models/activity_model.pt`, `models/label_map.json`, and `sample_data/` are included in the ZIP.

## Quick start — Windows

Open two terminals in the `Human-Activity-Recognition` project folder.

### Terminal 1: backend

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, run `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and launch Uvicorn with `\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`.

### Terminal 2: frontend

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. API docs are available at **http://127.0.0.1:8000/docs**. Keep the backend running while using the website.

Convenience scripts are included: `run_api.bat` and `run_frontend.bat` (run in separate terminals). The frontend uses `http://127.0.0.1:8000` by default. For a different backend URL, create `frontend/.env` with `VITE_API_URL=http://127.0.0.1:8000` and restart Vite.

## API routes

| Route | Purpose |
|---|---|
| `GET /api/health` | Backend and checkpoint status |
| `GET /api/model` | Model name, classes and input configuration |
| `GET /api/dashboard` | Aggregate metrics, distribution and recent logs |
| `GET /api/history` | Search/filter activity logs |
| `DELETE /api/history` | Clear activity history |
| `GET /api/samples` | List bundled demo videos |
| `POST /api/analyze` | Upload a video and classify sampled frames |
| `POST /api/predict-frame` | Browser webcam still-frame preview |
| `GET /api/metrics` | Existing evaluation report and artifact status |

## Training and evaluation

The existing model architecture and training scripts are retained.

```bash
python create_sample_data.py
python audit_dataset.py --dataset dataset
python train.py --epochs 10 --batch_size 6
python evaluate.py
```

**Do not blindly run training before understanding the dataset.** The included sample data appears to be copied into multiple splits. `audit_dataset.py` checks byte-identical files across train/validation/test and exits with a non-zero status if it finds leakage. Remove duplicate/derived clips, obtain independent clips (ideally split by person/source video), then retrain and evaluate. Hash checks cannot detect re-encoded or visually near-identical clips.

## Model input and output

- Input: 16 uniformly sampled frames resized to 64×64, RGB, normalized to [0, 1].
- Model: CNN spatial features + bidirectional LSTM temporal sequence classifier.
- Labels in the checkpoint: Walking, Running, Sitting, Standing, Lying_Down and Falling.
- Output: class probabilities and top predicted label.
- A confidence score is not the same as calibrated probability or validated real-world accuracy.

## Important limitations

1. The bundled dataset contains byte-identical videos across splits. Do not claim the old score represents generalization.
2. A browser webcam still frame repeated to satisfy the sequence input cannot reveal motion. The interface states this and does not label it as a verified fall event.
3. The upload endpoint samples a fixed sequence and classifies the clip once; it is not continuous per-frame tracking.
4. Browser webcam access requires localhost or HTTPS and user permission.
5. A “Falling” result is a model estimate and must not be relied on as the sole safety mechanism.

## Legacy app

The original Streamlit `app.py` and command-line `predict.py` remain in the project for reference. The new default dashboard is React + FastAPI.

## Project layout

```text
backend/main.py          FastAPI routes and model inference
frontend/                React + Vite dashboard
model_architecture.py    PyTorch LRCN architecture and checkpoint loading
preprocess.py            Video frame sampling and normalization
train.py / evaluate.py   Training and evaluation scripts
audit_dataset.py         Cross-split exact-hash audit
database/                SQLite activity history
models/                  Existing trained checkpoint and label map
sample_data/             Curated sample videos
outputs/                 Existing evaluation artifacts
```
