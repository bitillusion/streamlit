import streamlit as st
import plotly.graph_objects as go
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import tempfile
import os

# -------------------------------
# Core Processing Functions
# -------------------------------

def extract_brightness(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    brightness = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))

    cap.release()
    return brightness, fps


def clean_data(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[~((data < lower) | (data > upper))]


def detect_peaks(data, distance=9, width=4, rel_height=0.5):
    peaks, _ = signal.find_peaks(data, distance=distance, width=width, rel_height=rel_height)
    return peaks


def estimate_bpm(peaks, fps):
    if len(peaks) < 2:
        return None
    gaps = np.diff(peaks)
    mode_gap = stats.mode(gaps, keepdims=False)[0]
    return int(fps * 60 / mode_gap)


def plot_brightness_with_peaks(data, peaks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(data)),
        y=data,
        mode='lines',
        name='Brightness'
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(data))[peaks],
        y=data[peaks],
        mode='markers',
        name='Peaks',
        marker=dict(color='red', size=8, symbol='x')
    ))
    fig.update_layout(
        title="Brightness Data with Detected Peaks",
        xaxis_title="Frame Index",
        yaxis_title="Average Brightness",
        showlegend=True,
        height=400
    )
    return fig

# -------------------------------
# Streamlit App UI
# -------------------------------

st.set_page_config(page_title="Video Brightness BPM Estimator", layout="centered")
st.title("Heart BPM Calculator")

uploaded_file = st.file_uploader("📁 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    with st.spinner("🔄 Processing video... Please wait."):
        brightness_data, fps = extract_brightness(temp_video_path)
        os.remove(temp_video_path)

        if brightness_data is None:
            st.error("❌ Could not read the video file.")
        else:
            duration = round(len(brightness_data) / fps, 1)
            st.success(f"✅ Video processed!\n📹 FPS: {round(fps, 1)}\n⏱️ Duration: {duration} seconds")

            # Clean and analyze
            df = pd.DataFrame(brightness_data, columns=["brightness"])
            data_clean = clean_data(df["brightness"])
            peaks = detect_peaks(data_clean.values)


            # BPM
            st.write("") 
            bpm = estimate_bpm(peaks, fps)
            if bpm:
                st.metric("Estimated BPM", f"{bpm} BPM")
            else:
                st.warning("⚠️ Not enough peaks detected to estimate BPM.")

            st.markdown("---")
            # Plot
            fig = plot_brightness_with_peaks(data_clean.values, peaks)
            st.plotly_chart(fig, use_container_width=True)

