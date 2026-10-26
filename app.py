"""
Streamlit Web Dashboard for Human Activity Recognition (HAR) & Fall Detection System.
Provides Real-Time Webcam Stream, Video File Analysis, Evaluation Charts,
SQLite History Logs, and Academic Project Presentation documentation.
"""

import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Database and Model modules
from database.db_manager import init_db, log_activity, get_recent_logs, get_fall_alerts, get_activity_stats, clear_history
from model_architecture import load_model, DEFAULT_CLASSES, DEFAULT_IMAGE_SIZE, DEFAULT_SEQUENCE_LENGTH
from predict import ActivityPredictor, CLASS_COLORS

# Streamlit Page Configuration
st.set_page_config(
    page_title="Human Activity Recognition | Deep Learning AI System",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1a1e24;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }
    .alert-card {
        background-color: #3e1212;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #FF5252;
        color: #FFCDD2;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Ensure Database is initialized
init_db()

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/fluency/96/motion-sensor.png", width=70)
st.sidebar.title("HAR System Menu")
st.sidebar.markdown("**B.Tech CSE (AI) Capstone Project**")

menu_choice = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview & Dashboard",
        "📹 Real-Time Webcam Detection",
        "📁 Video File Analysis",
        "📊 Model Evaluation & Metrics",
        "📜 Activity Logs & Database",
        "📚 Project Architecture & Viva Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Configuration")
model_file_exists = os.path.exists("models/activity_model.pt")
status_color = "🟢" if model_file_exists else "🔴"
st.sidebar.markdown(f"**Model Status**: {status_color} {'Loaded' if model_file_exists else 'Not Trained'}")
conf_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=40, max_value=95, value=65, step=5)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tech Stack**: PyTorch · LRCN (CNN+BiLSTM) · OpenCV · Streamlit · SQLite")


# ==========================================
# 1. OVERVIEW & DASHBOARD
# ==========================================
if menu_choice == "🏠 Overview & Dashboard":
    st.markdown('<p class="main-header">🏃 Human Activity Recognition & Fall Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-Powered Spatial-Temporal Video Analysis using LRCN Architecture</p>', unsafe_allow_html=True)

    # Top KPI Metrics
    stats_df = get_activity_stats()
    total_logs = stats_df['count'].sum() if not stats_df.empty else 0
    total_falls = stats_df['total_falls'].sum() if not stats_df.empty else 0
    avg_conf = stats_df['avg_confidence'].mean() if not stats_df.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Activities Monitored", f"{int(total_logs):,}")
    with col2:
        st.metric("Active Classes", f"{len(DEFAULT_CLASSES)}")
    with col3:
        st.metric("Fall Alerts Triggered", f"{int(total_falls)}", delta="Critical Alerts" if total_falls > 0 else None, delta_color="inverse")
    with col4:
        st.metric("Avg Prediction Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # Overview Cards
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("🎯 Project Purpose & Capabilities")
        st.markdown("""
        Human Activity Recognition (HAR) automatically identifies human physical actions from continuous video streams.
        This system is built using a **Long-term Recurrent Convolutional Network (LRCN)**:
        - **Spatial Vision**: Time-Distributed CNN extracts 256-dimensional feature representations for each video frame.
        - **Temporal Dynamics**: 2-layer Bidirectional LSTM models motion continuity across time (strides, transitions, posture changes).
        - **Real-Time Safety**: Automated **Fall Detection & Alerting** with sub-second latency for eldercare and workplace safety.
        - **Full-Stack Integration**: Complete pipeline from raw video capture to database persistence and visual analytics.
        """)

        st.markdown("#### 🏃 Supported Activity Classes")
        class_cols = st.columns(3)
        for i, cls in enumerate(DEFAULT_CLASSES):
            with class_cols[i % 3]:
                st.markdown(f"✅ **{cls.replace('_', ' ')}**")

    with c2:
        st.subheader("📊 Historical Activity Distribution")
        if not stats_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(stats_df['activity'], stats_df['count'], color='#1E88E5')
            ax.set_ylabel("Inferences Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No activity records yet. Run Real-Time Webcam or Video Analysis to populate data!")

    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.markdown("**1. Real-Time Detection**")
        st.caption("Switch to 'Real-Time Webcam' to test continuous action classification with your camera.")
    with qc2:
        st.markdown("**2. Test Video Upload**")
        st.caption("Upload any MP4 video or pick a pre-loaded sample clip to analyze activity curves.")
    with qc3:
        st.markdown("**3. Model Performance**")
        st.caption("View the normalized Confusion Matrix and Loss/Accuracy training curves in 'Model Evaluation'.")


# ==========================================
# 2. REAL-TIME WEBCAM DETECTION
# ==========================================
elif menu_choice == "📹 Real-Time Webcam Detection":
    st.markdown('<p class="main-header">📹 Real-Time Webcam Activity Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Continuous Video Stream Classification with HUD Telemetry and Fall Detection Alert</p>', unsafe_allow_html=True)

    if not model_file_exists:
        st.warning("⚠️ Model weights not found in `models/activity_model.pt`. Please run `py train.py` first!")
    else:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
        with ctrl_col1:
            start_btn = st.button("▶️ Start Webcam Stream", type="primary", use_container_width=True)
        with ctrl_col2:
            stop_btn = st.button("⏹️ Stop Webcam Stream", use_container_width=True)

        status_box = st.empty()
        feed_col, metrics_col = st.columns([2, 1])

        video_placeholder = feed_col.empty()

        with metrics_col:
            st.markdown("### 📊 Live Telemetry")
            live_activity_box = st.empty()
            live_conf_box = st.empty()
            live_alert_box = st.empty()
            live_fps_box = st.empty()

        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        if start_btn:
            st.session_state.webcam_running = True
        if stop_btn:
            st.session_state.webcam_running = False

        if st.session_state.webcam_running:
            predictor = ActivityPredictor(model_path="models/activity_model.pt", label_map_path="models/label_map.json")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not connect to Webcam. Please verify camera permissions.")
                st.session_state.webcam_running = False
            else:
                status_box.success("Webcam stream active. Monitoring activities...")
                prev_time = time.time()

                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read video frame.")
                        break

                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time + 1e-6)
                    prev_time = curr_time

                    vis, pred_label, conf, is_fall = predictor.process_frame(frame, source_name="Webcam")

                    # Convert to RGB for Streamlit
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)

                    # Update Telemetry Widgets
                    color_hex = "#FF5252" if pred_label == "Falling" else "#1E88E5"
                    live_activity_box.markdown(f"""
                    <div style="background-color: #212529; padding: 15px; border-radius: 8px; border-left: 5px solid {color_hex};">
                        <span style="color: #9E9E9E; font-size: 0.85rem;">CURRENT ACTIVITY</span>
                        <h2 style="color: #FFFFFF; margin: 4px 0 0 0;">{pred_label.replace('_', ' ')}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    live_conf_box.metric("Prediction Confidence", f"{conf:.1f}%")
                    live_fps_box.metric("Inference Speed", f"{fps:.1f} FPS")

                    if is_fall:
                        live_alert_box.markdown("""
                        <div class="alert-card">
                            <h4 style="margin: 0; color: #FF5252;">⚠️ CRITICAL ALERT!</h4>
                            <p style="margin: 4px 0 0 0;">Human Fall Event Detected! Incident logged.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        live_alert_box.empty()

                    time.sleep(0.01)

                cap.release()
                status_box.info("Webcam stream stopped.")


# ==========================================
# 3. VIDEO FILE ANALYSIS
# ==========================================
elif menu_choice == "📁 Video File Analysis":
    st.markdown('<p class="main-header">📁 Video File Analysis & Action Breakdown</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an activity video clip or choose from curated benchmark demo videos</p>', unsafe_allow_html=True)

    input_mode = st.radio("Select Video Source:", ["Use Preloaded Demo Clip", "Upload Video File (.mp4, .avi, .mov)"], horizontal=True)

    video_path_to_process = None

    if input_mode == "Use Preloaded Demo Clip":
        sample_dir = "sample_data"
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".mp4")]
            selected_sample = st.selectbox("Choose Sample Action Video:", sample_files)
            if selected_sample:
                video_path_to_process = os.path.join(sample_dir, selected_sample)
        else:
            st.warning("No sample files found. Run `py create_sample_data.py` to generate sample clips.")

    else:
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path_to_process = tfile.name

    if video_path_to_process and os.path.exists(video_path_to_process):
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.markdown("#### Input Video Preview")
            st.video(video_path_to_process)

        with v_col2:
            st.markdown("#### Video Details")
            cap = cv2.VideoCapture(video_path_to_process)
            v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            st.write(f"**Resolution**: {v_w} x {v_h}")
            st.write(f"**Total Frames**: {v_frames}")
            st.write(f"**Frame Rate**: {v_fps:.1f} FPS")
            st.write(f"**Duration**: {v_frames / v_fps:.2f} seconds")

            run_analysis_btn = st.button("🚀 Run Deep Learning Analysis", type="primary", use_container_width=True)

        if run_analysis_btn:
            if not model_file_exists:
                st.error("Model weights `models/activity_model.pt` not found. Please train model first.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                predictor = ActivityPredictor(model_path="models/activity_model.pt", label_map_path="models/label_map.json")
                cap = cv2.VideoCapture(video_path_to_process)

                frame_preds = []
                frame_confs = []
                frame_falls = []
                frame_times = []

                frame_idx = 0
                sample_frames_vis = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    t_sec = frame_idx / v_fps
                    vis, pred_label, conf, is_fall = predictor.process_frame(frame, source_name=os.path.basename(video_path_to_process))

                    frame_preds.append(pred_label)
                    frame_confs.append(conf)
                    frame_falls.append(is_fall)
                    frame_times.append(t_sec)

                    if frame_idx % 8 == 0:
                        sample_frames_vis.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                    if v_frames > 0:
                        progress_bar.progress(min(1.0, frame_idx / v_frames))
                    status_text.text(f"Analyzing frame {frame_idx}/{v_frames}...")

                cap.release()
                progress_bar.progress(1.0)
                status_text.success("Analysis complete!")

                # Log overall clip summary to DB
                if frame_preds:
                    majority_class = pd.Series(frame_preds).mode()[0]
                    avg_clip_conf = float(np.mean(frame_confs))
                    fall_occurred = any(frame_falls)
                    log_activity(
                        source=os.path.basename(video_path_to_process),
                        activity=majority_class,
                        confidence=avg_clip_conf,
                        is_fall=fall_occurred,
                        duration_seconds=v_frames / v_fps,
                        details=f"Video Analysis: Majority class {majority_class}"
                    )

                # Results Presentation
                st.markdown("---")
                st.subheader("📈 Prediction Dynamics & Action Breakdown")

                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    # Timeline Chart
                    timeline_df = pd.DataFrame({
                        "Time (s)": frame_times,
                        "Confidence (%)": frame_confs,
                        "Activity": frame_preds
                    })

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(timeline_df["Time (s)"], timeline_df["Confidence (%)"], color="#1E88E5", linewidth=2.5, label="Confidence")
                    ax.fill_between(timeline_df["Time (s)"], timeline_df["Confidence (%)"], alpha=0.2, color="#1E88E5")
                    ax.axhline(conf_threshold, color="#FF9800", linestyle="--", label=f"Threshold ({conf_threshold}%)")
                    ax.set_title("Prediction Confidence Over Time", fontsize=12, fontweight="bold")
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Confidence (%)")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    st.pyplot(fig)

                with res_col2:
                    # Class Distribution Pie Chart
                    st.markdown("#### Activity Breakdown")
                    pred_counts = pd.Series(frame_preds).value_counts()
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                    ax_pie.pie(pred_counts, labels=pred_counts.index, autopct="%1.1f%%", startangle=140, colors=["#1E88E5", "#00E676", "#FFC107", "#E040FB", "#FF5252", "#00B0FF"])
                    st.pyplot(fig_pie)

                if any(frame_falls):
                    st.error("🚨 **CRITICAL NOTICE**: A Fall Event was detected during this video clip!")
                    fall_indices = [i for i, f in enumerate(frame_falls) if f]
                    first_fall_time = frame_times[fall_indices[0]]
                    st.write(f"First Fall Detected at timestamp: **{first_fall_time:.2f} seconds** (Frame {fall_indices[0]})")

                # Sample Annotated Frames Gallery
                if sample_frames_vis:
                    st.markdown("#### 🖼️ Annotated Frame Samples")
                    cols = st.columns(min(4, len(sample_frames_vis)))
                    for i, img in enumerate(sample_frames_vis[:4]):
                        with cols[i]:
                            st.image(img, use_container_width=True)


# ==========================================
# 4. MODEL EVALUATION & METRICS
# ==========================================
elif menu_choice == "📊 Model Evaluation & Metrics":
    st.markdown('<p class="main-header">📊 Model Evaluation & Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Rigorous Academic Benchmark Metrics: Accuracy, Precision, Recall, F1-Score & Confusion Matrix</p>', unsafe_allow_html=True)

    cm_path = "outputs/confusion_matrix.png"
    history_path = "outputs/training_history.png"
    report_path = "outputs/classification_report.txt"
    preds_path = "outputs/test_predictions.csv"

    eval_tabs = st.tabs(["🔥 Confusion Matrix", "📉 Training Curves", "📋 Classification Report", "📑 Test Set Predictions"])

    with eval_tabs[0]:
        st.subheader("Normalized Confusion Matrix Heatmap")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
            st.caption("Normalized class-by-class confusion matrix showing diagonal true positive rates and inter-class confusions.")
        else:
            st.info("Confusion matrix not generated yet. Run `py evaluate.py` to evaluate model on test set.")

    with eval_tabs[1]:
        st.subheader("Training & Validation Loss & Accuracy History")
        if os.path.exists(history_path):
            st.image(history_path, use_container_width=True)
            st.caption("Loss convergence and validation accuracy trajectory across training epochs.")
        else:
            st.info("Training history plot not found. Run `py train.py` to generate training curves.")

    with eval_tabs[2]:
        st.subheader("Classification Report (Precision, Recall, F1-Score)")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.text(report_content)
        else:
            st.info("Classification report not generated yet. Run `py evaluate.py` to generate.")

    with eval_tabs[3]:
        st.subheader("Test Dataset Predictions Summary")
        if os.path.exists(preds_path):
            test_df = pd.read_csv(preds_path)
            st.dataframe(test_df, use_container_width=True)
            csv_data = test_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Test Predictions CSV", data=csv_data, file_name="har_test_predictions.csv", mime="text/csv")
        else:
            st.info("Run `py evaluate.py` to view individual test set predictions.")


# ==========================================
# 5. ACTIVITY LOGS & DATABASE
# ==========================================
elif menu_choice == "📜 Activity Logs & Database":
    st.markdown('<p class="main-header">📜 Activity History & Event Logs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Persistent SQLite Database Logging for Predictions, Inferences, and Fall Alerts</p>', unsafe_allow_html=True)

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        view_mode = st.selectbox("Filter Logs:", ["All Activity Records", "Fall Alerts Only"])
    with filter_col2:
        record_limit = st.selectbox("Display Limit:", [25, 50, 100, 200], index=1)
    with filter_col3:
        clear_btn = st.button("🗑️ Clear History Logs", type="secondary")

    if clear_btn:
        clear_history()
        st.success("Activity history database cleared successfully!")

    if view_mode == "Fall Alerts Only":
        df_logs = get_fall_alerts(limit=record_limit)
    else:
        df_logs = get_recent_logs(limit=record_limit)

    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True)

        csv_export = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Logs to CSV",
            data=csv_export,
            file_name="activity_recognition_logs.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("No prediction records found in the database. Run video analysis or webcam stream to generate logs.")


# ==========================================
# 6. PROJECT ARCHITECTURE & VIVA GUIDE
# ==========================================
elif menu_choice == "📚 Project Architecture & Viva Guide":
    st.markdown('<p class="main-header">📚 B.Tech CSE (AI) Final Year Project & Viva Guide</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">System Architecture, Mathematical Formulation, Key Concepts, and Viva Q&A</p>', unsafe_allow_html=True)

    viva_tab1, viva_tab2, viva_tab3 = st.tabs(["🏗️ System Architecture", "🧠 Deep Learning Formulation", "🎓 Viva Questions & Answers"])

    with viva_tab1:
        st.markdown("""
        ### Complete System Pipeline
        ```text
        [Video Input (Webcam / Uploaded File)]
                       │
                       ▼
        [OpenCV Video Preprocessing]
          - Frame extraction & uniform temporal sampling (16 frames)
          - Spatial resize to 64x64
          - RGB conversion & [0, 1] normalization
                       │
                       ▼
        [LRCN Deep Learning Model]
          - TimeDistributed CNN: 4 Conv2D + BatchNorm blocks (Spatial Features)
          - Bidirectional LSTM: 2 layers, 128 hidden units (Temporal Dynamics)
          - Temporal Mean Pooling & Dense Classification Head
                       │
                       ▼
        [Inference & Post-Processing]
          - Softmax probability distribution
          - Rolling temporal window smoothing
          - Fall Detection Threshold Logic (> 60% confidence)
                       │
                       ▼
        [Output Delivery & Persistence]
          - Streamlit Web Dashboard & Visual HUD Overlay
          - Audio & Visual Fall Alarm
          - SQLite Activity History Logging
        ```
        """)

    with viva_tab2:
        st.markdown("""
        ### Why LRCN (Long-term Recurrent Convolutional Network)?
        Standard 2D CNNs treat every video frame independently as a static picture, completely failing to recognize movement or sequence of actions (e.g. running vs walking, standing vs falling).

        LRCN solves this with a two-fold representation:
        1. **Spatial Representation**:
           $$f_t = \\text{CNN}(I_t)$$
           Where $I_t$ is the video frame at time $t$, and $f_t \\in \\mathbb{R}^{256}$ is the spatial feature vector.
        2. **Temporal Representation**:
           $$h_t = \\text{Bi-LSTM}(f_t, h_{t-1}, h_{t+1})$$
           Where $h_t$ captures both forward and backward temporal movement dependencies.
        3. **Classification**:
           $$P(y = c \\mid I_{1:T}) = \\text{Softmax}(W \\cdot \\bar{h} + b)$$
           Where $\\bar{h}$ is the temporally pooled feature representation.
        """)

    with viva_tab3:
        st.markdown("""
        ### Top Final-Year Viva Questions & Model Answers

        **Q1: What is the main difference between 2D CNN and CNN + LSTM for video processing?**
        > *Answer*: A standard 2D CNN only learns spatial features (shapes, textures, objects) inside a single isolated frame. It has no temporal memory. An LRCN (CNN+LSTM) model feeds the spatial feature vectors of consecutive frames into an LSTM cell, allowing the network to learn velocity, strides, and temporal transition patterns over time.

        **Q2: How does the system handle videos of variable frame rates and lengths?**
        > *Answer*: In `preprocess.py`, we apply **uniform temporal sampling**. Given a video of arbitrary duration, we calculate $16$ evenly spaced indices across the entire frame sequence ($t = \\text{linspace}(0, N-1, 16)$). This standardizes inputs to fixed temporal dimensions without losing long-term context.

        **Q3: How is Fall Detection implemented and how are false positives minimized?**
        > *Answer*: We combine a trained deep learning class (`Falling`) with a **temporal smoothing buffer** (averaging probabilities over past $K$ inferences) and a confidence threshold ($>60\\%$). Sudden downward tilts and impact poses must maintain confidence over consecutive frames before triggering critical alarms.

        **Q4: What metrics did you use to evaluate performance?**
        > *Answer*: We evaluated Overall Accuracy, Macro-averaged Precision, Recall, F1-Score, and a normalized Confusion Matrix to detect class imbalances or inter-class confusion (e.g., distinguishing between Sitting and Lying Down).
        """)
