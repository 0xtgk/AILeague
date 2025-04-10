# 🏥 PlaySafe – Injury Risk Analyzer for Football Players

<img src="https://img.shields.io/badge/built%20with-Python%20%7C%20YOLOv8%20%7C%20MediaPipe-blue.svg" alt="tech-stack" />
<img src="https://img.shields.io/badge/Streamlit-frontend-red.svg" />
<img src="https://img.shields.io/github/license/yourusername/playsafe" />

**PlaySafe** is an AI-powered web application that analyzes football videos and predicts injury risk for players in real time using computer vision and pose estimation.

### 🎯 Purpose

> Help teams, coaches, and analysts proactively detect potential injury risks from match footage — before injuries actually occur.

---

## ✨ Features

✅ Upload `.mp4` videos of football matches  
✅ Uses YOLOv8 for player detection  
✅ Applies MediaPipe Pose for joint tracking  
✅ Displays risk percentage above each player  
✅ Flags players as `⚠️ INJURED` if their movement indicates high risk  
✅ Automatically re-encodes videos to browser-playable format (H.264)  
✅ Streamlit UI – no installation needed by users!

---

## 🖼️ Demo

> Coming soon... (Add a video/GIF here showing the app in action)

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.9+
- FFmpeg installed and added to system `PATH`  
  [How to install FFmpeg on Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)

### 📦 Installation

```bash
git clone https://github.com/yourusername/playsafe.git
cd playsafeWebapp

# Install dependencies
pip install -r requirements.txt
