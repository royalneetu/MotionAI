# MotionAI — Human Activity Recognition Using Deep Learning

MotionAI is a web-based Human Activity Recognition (HAR) project designed to analyze human activities using deep learning and computer vision. It combines a React dashboard with a FastAPI backend to provide an interface for activity analysis and model-related information.

## Features

- **Live Camera Interface:** Access the webcam through the browser.
- **Activity Categories:** Supports project data organized into Walking, Running, Sitting, Standing, Falling, and Lying Down.
- **Video Samples:** Includes sample videos for supported activity categories.
- **Activity Dashboard:** View activity-related information through a web interface.
- **Backend API:** FastAPI endpoints for health checks, model information, activity analysis, history, and metrics.
- **Model Evaluation:** Includes scripts and output files for model evaluation and classification reporting.
- **Pose Tracking Integration:** MediaPipe Pose integration is being developed for displaying body landmarks and skeleton connections.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite, JavaScript |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Pose Tracking | MediaPipe Tasks Vision |
| Data Visualization | Recharts |
| API Communication | REST API |

## Project Structure

```text
MotionAI/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   └── label_map.json
├── sample_data/
├── database/
├── train.py
├── evaluate.py
├── predict.py
├── preprocess.py
├── model_architecture.py
├── requirements.txt
└── README.md
```

## Supported Activity Categories

- Walking
- Running
- Sitting
- Standing
- Falling
- Lying Down

The dataset contains example video files organized into training, validation, and test folders.

## Getting Started

### Prerequisites

Install the following software:

- Python
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/royalneetu/MotionAI.git
cd MotionAI
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

Start the frontend:

```bash
npm run dev
```

Open the local URL displayed by Vite in your browser.

### 4. Start the Backend

Open another terminal in the project root, activate the virtual environment, and run:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Check the backend health endpoint:

```text
http://127.0.0.1:8000/api/health
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Model Setup

The trained model checkpoint is not included in this repository. The backend expects the checkpoint at:

```text
models/activity_model.pt
```

Place a compatible trained checkpoint at that location before using model-dependent prediction and analysis features. Follow the project's training scripts and model architecture when preparing the checkpoint.

## Model Training and Evaluation

The repository includes scripts for training, preprocessing, prediction, and evaluation:

```bash
python train.py
python evaluate.py
python predict.py
```

Check each script's configuration and required arguments before running it. Training and evaluation require the appropriate dataset and dependencies.

## API Overview

The backend provides endpoints for:

- Health checks
- Model information
- Dashboard data
- Activity history
- Sample data
- Activity analysis
- Frame prediction
- Model metrics

See the FastAPI documentation at `/docs` for available endpoints and request formats.

## Current Development Status

- React dashboard and FastAPI backend are included.
- Dataset samples and model evaluation outputs are included.
- Model checkpoint setup is required for model-dependent inference.
- MediaPipe-based skeleton visualization is under development.

## Future Improvements

- Complete real-time skeleton tracking and activity recognition.
- Improve model accuracy and evaluate performance on unseen videos.
- Add more activity classes and diverse training samples.
- Improve error handling and inference performance.
- Add deployment instructions and demonstration screenshots.

## Author

**Neetesh Chaurasia**

GitHub: [@royalneetu](https://github.com/royalneetu)

## License

A license has not yet been specified for this repository.
