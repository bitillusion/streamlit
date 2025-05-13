import streamlit as st
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import tempfile
import os

st.title("Video Brightness BPM Estimator")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.write("Processing video...")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract brightness from each frame
        brightness_data = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            brightness_data.append(avg_brightness)
            frame_count += 1

        cap.release()
        os.remove(temp_video_path)  # Clean up temp file

        st.success(f"Video processed! FPS: {round(fps, 1)}, Duration: {round(len(brightness_data)/fps, 1)} seconds")

        # Data preprocessing
        df = pd.DataFrame(brightness_data, columns=["brightness"])
        data = df["brightness"]
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        data_clean = data[~((data < lower) | (data > upper))]

        # Peak detection
        peaks, _ = signal.find_peaks(
            data_clean.values,
            distance=9,
            width=4,
            rel_height=0.5
        )






        import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import tempfile
import os

st.title("Video Brightness BPM Estimator")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.write("Processing video...")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract brightness from each frame
        brightness_data = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            brightness_data.append(avg_brightness)
            frame_count += 1

        cap.release()
        os.remove(temp_video_path)  # Clean up temp file

        st.success(f"Video processed! FPS: {round(fps, 1)}, Duration: {round(len(brightness_data)/fps, 1)} seconds")

        # Data preprocessing
        df = pd.DataFrame(brightness_data, columns=["brightness"])
        data = df["brightness"]
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        data_clean = data[~((data < lower) | (data > upper))]

        # Peak detection
        peaks, _ = signal.find_peaks(
            data_clean.values,
            distance=9,
            width=4,
            rel_height=0.5
        )

        fig, ax = plt.subplots(figsize=(18, 4))
        x = np.arange(len(data_clean))
        ax.plot(x, data_clean, label='Brightness')
        ax.plot(x[peaks], data_clean.values[peaks], "x", label='Peaks', color='red')
        ax.legend()
        ax.set_title("Brightness Data with Peaks")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Average Brightness")
        ax.grid(True)

        # Show in Streamlit
        st.pyplot(fig)



        
        if len(peaks) > 1:
            # Calculate BPM
            gaps = np.diff(peaks)
            mode_gap = stats.mode(gaps, keepdims=False)[0]
            estimated_bpm = int(fps * 60 / mode_gap)
            st.metric("Estimated BPM", f"{estimated_bpm} BPM")
        else:
            st.warning("Not enough peaks detected to estimate BPM.")
