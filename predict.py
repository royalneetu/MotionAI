"""
Inference & Prediction Pipeline for Human Activity Recognition.
Supports video file inference and real-time webcam streams.
Features rolling temporal window smoothing, on-screen HUD overlay,
automated fall detection alerts, and SQLite database logging.
"""

import os
import sys
import argparse
import time
from collections import deque
import cv2
import numpy as np
import torch

from model_architecture import load_model, DEFAULT_IMAGE_SIZE, DEFAULT_SEQUENCE_LENGTH
from preprocess import preprocess_frame_buffer
from database.db_manager import log_activity, init_db

# Try importing winsound for audible fall alert on Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# Activity color styling (BGR format)
CLASS_COLORS = {
    "Walking": (50, 205, 50),       # Lime Green
    "Running": (0, 215, 255),       # Amber Gold
    "Sitting": (255, 165, 0),       # Deep Sky Blue
    "Standing": (205, 133, 63),     # Periwinkle
    "Lying_Down": (180, 105, 255),  # Pinkish Violet
    "Falling": (0, 0, 255)          # Bright Red (Alert)
}


class ActivityPredictor:
    """
    High-performance real-time Activity Predictor using LRCN.
    Maintains a rolling temporal frame queue and smoothed probability buffer.
    """
    def __init__(self, model_path="models/activity_model.pt", label_map_path="models/label_map.json", seq_len=DEFAULT_SEQUENCE_LENGTH, img_size=DEFAULT_IMAGE_SIZE, device="cpu"):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.img_size = img_size
        self.model, self.class_names = load_model(model_path, label_map_path, device=self.device)
        self.model.eval()

        # Rolling frame queue
        self.frame_buffer = deque(maxlen=seq_len)
        # Smoothing probability history
        self.prob_history = deque(maxlen=5)

        self.last_predicted_class = "Initializing..."
        self.last_confidence = 0.0
        self.last_fall_time = 0.0
        init_db()

    def process_frame(self, frame_bgr: np.ndarray, source_name="Webcam") -> tuple:
        """
        Receives an incoming BGR video frame, updates the buffer, performs inference
        when buffer is ready, and draws the HUD overlay.
        Returns: (annotated_frame, predicted_class, confidence, is_fall_alert)
        """
        h, w, _ = frame_bgr.shape
        self.frame_buffer.append(frame_bgr.copy())

        is_fall_alert = False

        if len(self.frame_buffer) >= self.seq_len:
            # Prepare tensor
            tensor_in = preprocess_frame_buffer(list(self.frame_buffer), img_size=self.img_size, seq_len=self.seq_len)
            tensor_in = tensor_in.to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            self.prob_history.append(probs)
            # Temporal smoothing: average probabilities across past few inferences
            smoothed_probs = np.mean(self.prob_history, axis=0)
            pred_idx = int(np.argmax(smoothed_probs))
            self.last_predicted_class = self.class_names[pred_idx]
            self.last_confidence = float(smoothed_probs[pred_idx] * 100.0)

            # Fall detection check
            if self.last_predicted_class == "Falling" and self.last_confidence > 60.0:
                is_fall_alert = True
                cur_t = time.time()
                # Cooldown of 3 seconds between audible beeps and DB logs
                if cur_t - self.last_fall_time > 3.0:
                    self.last_fall_time = cur_t
                    log_activity(
                        source=source_name,
                        activity="Falling",
                        confidence=self.last_confidence,
                        is_fall=True,
                        details="Critical Fall Detected by LRCN Model"
                    )
                    if HAS_WINSOUND:
                        try:
                            winsound.Beep(1200, 300)
                        except Exception:
                            pass

        annotated = self._draw_hud(frame_bgr, self.last_predicted_class, self.last_confidence, is_fall_alert)
        return annotated, self.last_predicted_class, self.last_confidence, is_fall_alert

    def _draw_hud(self, frame, label, conf, is_fall):
        """Draws aesthetic HUD telemetry, confidence meter, and fall warning banners."""
        vis = frame.copy()
        h, w, _ = vis.shape

        color = CLASS_COLORS.get(label, (200, 200, 200))

        # Top Banner Overlay
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (min(w - 10, 420), 105), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

        # Border for info box
        cv2.rectangle(vis, (10, 10), (min(w - 10, 420), 105), color, 2)

        # Activity Title
        cv2.putText(vis, f"Activity: {label.replace('_', ' ')}", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence Bar
        bar_x, bar_y, bar_w, bar_h = 25, 58, 220, 14
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int((conf / 100.0) * bar_w)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(vis, f"{conf:.1f}%", (bar_x + bar_w + 12, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        # Model Info
        cv2.putText(vis, "LRCN: CNN + BiLSTM Temporal Model", (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

        # Blinking Fall Alert Banner
        if is_fall:
            alert_bar = vis.copy()
            cv2.rectangle(alert_bar, (0, h - 60), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(alert_bar, 0.85, vis, 0.15, 0, vis)
            alert_text = "! CRITICAL ALERT: HUMAN FALL DETECTED !"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
            text_x = max(20, (w - text_size[0]) // 2)
            cv2.putText(vis, alert_text, (text_x, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        return vis


def run_video_prediction(video_path: str, output_path: str = None, model_path: str = "models/activity_model.pt", show_window: bool = True):
    """Processes a video file and visualizes activity predictions."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    predictor = ActivityPredictor(model_path=model_path)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path} ({width}x{height} @ {fps:.1f} fps)")

    frame_count = 0
    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis, pred_label, conf, is_fall = predictor.process_frame(frame, source_name=os.path.basename(video_path))

        if out:
            out.write(vis)

        if show_window:
            cv2.imshow("Human Activity Recognition - LRCN Inference", vis)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
        print(f"Output video saved to {output_path}")
    if show_window:
        cv2.destroyAllWindows()

    total_time = time.time() - start_t
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({frame_count / total_time:.1f} FPS).")


def run_webcam_prediction(camera_id: int = 0, model_path: str = "models/activity_model.pt"):
    """Runs real-time webcam inference with live HUD."""
    predictor = ActivityPredictor(model_path=model_path)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("Starting Webcam Human Activity Recognition. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        vis, pred_label, conf, is_fall = predictor.process_frame(frame, source_name="Webcam")

        # Overlay FPS
        cv2.putText(vis, f"FPS: {fps:.1f}", (vis.shape[1] - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Human Activity Recognition - Real-Time Webcam", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Activity Recognition Video/Webcam Inference")
    parser.add_argument("--video", type=str, default="", help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera index")
    parser.add_argument("--output", type=str, default="", help="Path to save annotated video output")
    parser.add_argument("--model", type=str, default="models/activity_model.pt", help="Path to model weights")
    parser.add_argument("--no_window", action="store_true", help="Disable display window")

    args = parser.parse_args()

    if args.webcam:
        run_webcam_prediction(camera_id=args.cam_id, model_path=args.model)
    elif args.video:
        run_video_prediction(args.video, output_path=args.output, model_path=args.model, show_window=not args.no_window)
    else:
        # Default test run on sample demo video if available
        sample_path = "sample_data/demo_walking.mp4"
        if os.path.exists(sample_path):
            print(f"No input provided. Running demo test on {sample_path}...")
            run_video_prediction(sample_path, output_path="outputs/demo_predicted.mp4", model_path=args.model, show_window=False)
        else:
            print("Please specify --video <path> or --webcam.")
