import streamlit as st
import numpy as np
import mne
from tensorflow.keras.models import load_model
import os

# ---------------- SETTINGS ----------------
segment_size = 256
channels_to_use = 3
sampling_rate = 128

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Seizure Detection", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_ai_model():
    return load_model("final_model.keras")   # ✅ correct model

model = load_ai_model()

mean = np.load("mean.npy")
std = np.load("std.npy")

# ✅ FIX normalization shape
mean = mean.reshape(channels_to_use, 1)
std = std.reshape(channels_to_use, 1)
std[std == 0] = 1e-6

# ---------------- LOGIN SYSTEM ----------------
users = {
    "admin": "1234",
    "mahesh": "ai123"
}

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Project Info")
st.sidebar.write("""
AI-based seizure detection using EEG signals.

- Model: CNN
- Input: EEG (.edf)
- Output: Seizure / No Seizure
""")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ---------------- HEADER ----------------
st.title("🧠 AI Seizure Detection System")
st.write("EEG Analysis with Deep Learning")
st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload EEG (.edf)", type=["edf"])

if uploaded_file is None:
    st.info("📂 Please upload an EEG (.edf) file")

else:
    st.success("File uploaded!")

    # save temp file
    temp_path = "temp.edf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # ---------------- LOAD EEG ----------------
    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    # ✅ MATCH TRAINING PIPELINE
    raw.resample(sampling_rate)
    raw.filter(1, 30, verbose=False)

    data = raw.get_data()
    signal = data[:channels_to_use]

    st.success("EEG loaded successfully!")

    # ---------------- EEG PREVIEW ----------------
    st.subheader("📡 EEG Signal Preview")
    channel = st.selectbox("Select Channel", list(range(channels_to_use)))

    st.line_chart(signal[channel][:2000])

    # ---------------- SEGMENTATION ----------------
    segments = []

    for i in range(0, signal.shape[1] - segment_size, segment_size):
        seg = signal[:, i:i+segment_size]

        # ✅ correct normalization
        seg = (seg - mean) / std

        segments.append(seg)

    segments = np.array(segments)
    segments = np.transpose(segments, (0, 2, 1))

    # ---------------- PREDICTION ----------------
    with st.spinner("Running AI model..."):
        preds = model.predict(segments)

    preds = preds.flatten()

    # ✅ improved threshold
    threshold = 0.6
    preds_binary = (preds > threshold).astype(int)

    seizure_count = int(np.sum(preds_binary))
    total_segments = len(preds_binary)
    percentage = (seizure_count / total_segments) * 100

    # ---------------- RESULTS ----------------
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Segments", total_segments)
    col2.metric("Seizure Segments", seizure_count)
    col3.metric("Seizure %", f"{percentage:.2f}%")

    # ---------------- RISK LEVEL ----------------
    if percentage < 5:
        st.success("🟢 Low Risk")
        risk_level = "Low"
    elif percentage < 20:
        st.warning("🟡 Medium Risk")
        risk_level = "Medium"
    else:
        st.error("🔴 High Risk")
        risk_level = "High"

    # ---------------- TIMELINE ----------------
    st.subheader("📈 Seizure Probability Timeline")
    st.line_chart(preds)

    # ---------------- TABLE ----------------
    st.subheader("📊 Segment-wise Predictions")

    segment_data = {
        "Segment": list(range(len(preds))),
        "Probability": preds,
        "Prediction": preds_binary
    }

    st.dataframe(segment_data)

    # ---------------- FINAL DECISION ----------------
    st.divider()

    if percentage > 5:
        st.error("🚨 Seizure Detected")
    else:
        st.success("✅ No Seizure Detected")

    # ---------------- REPORT ----------------
    report = f"""
EEG Analysis Report

Total Segments: {total_segments}
Seizure Segments: {seizure_count}
Seizure Percentage: {percentage:.2f}%

Risk Level: {risk_level}
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="seizure_report.txt"
    )

    # cleanup
    os.remove(temp_path)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Developed by Your Mahesh Teja, Venu Sai, Lokesh, Charan Sai")