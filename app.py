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

def robust_mode_gap(gaps, neighbor_window=1, top_n=3):
    gap_counts = pd.Series(gaps).value_counts().sort_values(ascending=False)
    top_gaps = gap_counts.head(top_n)
    top_gap = top_gaps.index[0]
    neighbor_range = range(top_gap - neighbor_window, top_gap + neighbor_window + 1)
    matched = top_gaps[top_gaps.index.isin(neighbor_range)]
    print(gap_counts)
    weighted_gap = sum(gap * count for gap, count in matched.items()) / matched.sum()
    return weighted_gap

def estimate_bpm(peaks, fps):
    if len(peaks) < 2:
        return None
    gaps = np.diff(peaks)

    #mode_gap = stats.mode(gaps, keepdims=False)[0]
    #bpm = int(fps * 60 / mode_gap)
    #print(bpm)

    mode_gap = robust_mode_gap(gaps)
    bpm = int(fps * 60 / mode_gap)
    print(bpm)

    return bpm


def plot_brightness_with_peaks(data, peaks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        # x=np.arange(len(data)),
        x=np.arange(len(data)) / fps,       #add seconds data
        y=data,
        mode='lines',
        name='Brightness'
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(data))[peaks]/fps,
        y=data[peaks],
        mode='markers',
        name='Peaks',
        marker=dict(color='red', size=8, symbol='x')
    ))
    fig.update_layout(
        #title="",
        xaxis_title="Time (Seconds)",

        yaxis_title="Signal Strength",
        yaxis=dict(
            showticklabels=False,   # Hides the tick labels
            # title=None              # Removes the axis title
        ),

        showlegend=False,
        height=300,
        margin=dict(t=10) 
    )
    return fig



# -------------------------------
# Streamlit App UI
# -------------------------------


st.markdown("## 💓 Heart Rate Estimator")
st.caption("Estimate BPM from video brightness.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    with st.spinner("🔄 Processing video..."):
        brightness_data, fps = extract_brightness(temp_video_path)
        os.remove(temp_video_path)

        if brightness_data is None:
            st.error("❌ Could not read the video file.")
        else:
            duration = round(len(brightness_data) / fps, 1)
            st.success("✅ Video processed successfully!")
            st.caption(f"**Frame Rate:** {round(fps, 1)} FPS &nbsp;&nbsp; • &nbsp;&nbsp; **Duration:** {duration} seconds")

            df = pd.DataFrame(brightness_data, columns=["brightness"])
            data_clean = clean_data(df["brightness"])
            peaks = detect_peaks(data_clean.values)
            
            
            # BPM
            bpm = estimate_bpm(peaks, fps)
            if bpm:
                st.write("")
                st.metric("Estimated Heart Rate", f"{bpm} BPM")
            else:
                st.warning("⚠️ Not enough peaks detected to estimate BPM.")

            # Plot
            st.markdown("---")
            st.write("Brightness Data with Detected Peaks")
            fig = plot_brightness_with_peaks(data_clean.values, peaks)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📤 Upload a video file to begin analysis.")
