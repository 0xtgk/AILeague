import os, cv2, subprocess, sys, tempfile, shutil

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# def save_video(ouput_video_frames,output_video_path):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
#     for frame in ouput_video_frames:
#         out.write(frame)
#     out.release()

# def save_video(frames, out_path, fps: int = 24):
#     if out_path.endswith(".avi"):
#         fourcc = cv2.VideoWriter_fourcc(*"MJPG")   # universal
#     else:  # .mp4, .mkv ...
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # works on most PCs

#     h, w, _ = frames[0].shape
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

#     if not writer.isOpened():
#         raise RuntimeError("❌ OpenCV VideoWriter failed to open – check codec")

#     for f in frames:
#         writer.write(f)
#     writer.release()

FFMPEG = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.isfile(FFMPEG):
    FFMPEG = "ffmpeg"

def save_video(frames, out_path, fps: int = 24):
    """
    Writes frames to an MP4 (mp4v) that every browser can play.
    Falls back to MJPG-AVI + ffmpeg conversion if OpenCV lacks codecs.
    """

    h, w, _ = frames[0].shape
    ext = os.path.splitext(out_path)[1].lower()

    # --- try direct MP4 ---------------------------------------------------
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            return
        else:
            writer.release()          # cleanup handle
            print("⚠️  OpenCV couldn't open H.264/MP4; falling back to MJPG-AVI",
                  file=sys.stderr)

    # --- fallback: MJPG-AVI then transcode to MP4 -------------------------
    tmp_avi = out_path.replace(ext, ".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp_avi, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("❌ OpenCV VideoWriter failed (no codecs at all).")

    for f in frames:
        writer.write(f)
    writer.release()

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", tmp_avi,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",      # baseline profile—plays everywhere
        "-movflags", "+faststart",  # enables progressive playback
        out_path,
    ]

    # run the conversion
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("❌  FFmpeg failed:", e, file=sys.stderr)
        shutil.move(tmp_avi, out_path)      # leave AVI as a fallback
    else:
        os.remove(tmp_avi)                 # clean up temp AVI
