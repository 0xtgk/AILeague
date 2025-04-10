import cv2
import numpy as np
import tempfile
import os
import subprocess
from ultralytics import YOLO
import mediapipe as mp
import streamlit as st
from tqdm import tqdm
from PIL import Image

# Load models
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Estimate injury risk
def estimate_injury_risk(keypoints):
    if keypoints is None:
        return 0.0
    try:
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]
        leg_movement = np.linalg.norm(np.array(left_knee) - np.array(left_ankle)) + \
                       np.linalg.norm(np.array(right_knee) - np.array(right_ankle))
        risk = min(1.0, leg_movement / 300)
        return round(risk, 2)
    except:
        return 0.0

# Process video with YOLO + MediaPipe
def process_video(input_path, output_path, risk_threshold=0.7):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use H.264 later via ffmpeg
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in tqdm(frames):
        results = yolo_model(frame)[0]

        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            player_crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_crop)

            keypoints = None
            if result.pose_landmarks:
                keypoints = [
                    (int(p.x * (x2 - x1)), int(p.y * (y2 - y1)))
                    for p in result.pose_landmarks.landmark
                ]

            risk_prob = estimate_injury_risk(keypoints)
            is_injured = risk_prob >= risk_threshold

            box_color = (0, 0, 255) if is_injured else (0, 255, 0)
            label = f"Risk: {int(risk_prob * 100)}%"
            injury_label = "INJURED" if is_injured else "OK"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(frame, injury_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        out.write(frame)

    out.release()

# Convert to browser-compatible H.264 format
def convert_to_h264(input_path):
    output_path = input_path.replace(".mp4", "_h264.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        output_path
    ])
    return output_path

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Injury Risk Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center;'>PlaySafe</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a football match video and get real-time injury risk predictions per player.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    output_path = input_path.replace(".mp4", "_processed.mp4")

    with st.spinner("⚙️ Processing video..."):
        process_video(input_path, output_path)
        output_h264 = convert_to_h264(output_path)

    st.success("✅ Analysis complete!")
    st.video(output_h264)
