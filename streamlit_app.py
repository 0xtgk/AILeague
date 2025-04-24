# streamlit_app.py
import streamlit as st, json, subprocess, uuid, os, pandas as pd
from tempfile import NamedTemporaryFile

st.set_page_config("PlaySafe – PoC", layout="wide")

# ---------------- Sidebar: roster --------------------------------------
st.sidebar.header("Team roster")
names_txt = st.sidebar.text_area(
    "Enter player names (one per line):", height=200,
    placeholder="Ahmed\nFahad\nSaad")
names = [n.strip() for n in names_txt.splitlines() if n.strip()]

id_map = {i + 1: n for i, n in enumerate(names)}  # 1-based IDs
if names:
    st.sidebar.success(f"{len(names)} players loaded")

# ---------------- Main: upload + run -----------------------------------
st.title("PlaySafe – Proof-of-Concept")

video_file = st.file_uploader("Upload match video (MP4/MKV/AVI)", type=["mp4", "mkv", "avi"])

run_btn = st.button("▶ Run analysis", disabled=not (video_file and names))

if run_btn:
    # save video to a temp file
    tmp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video.write(video_file.read())
    tmp_video.close()

    # choose unique paths for output
    out_id = uuid.uuid4().hex[:8]
    out_video = f"output_videos/{out_id}.avi"
    out_json  = f"output_videos/{out_id}_stats.json"
    os.makedirs("output_videos", exist_ok=True)

    # call the pipeline synchronously
    with st.spinner("Running model ... this may take a while"):
        cmd = ["python", "main.py",
               "--input", tmp_video.name,
               "--weights", "models/best.pt",
               "--output", out_video,
               "--stats_out", out_json]
        subprocess.run(cmd, check=True)

        # -----------------------------------------------------------------
    #  analysis finished – show UI
    # -----------------------------------------------------------------
    st.success("✅ Analysis complete!")

    # offer download (works even if the browser won't play it inline)
    with open(out_video, "rb") as vf:
        st.download_button(
            label="⬇️  Download annotated video",
            data=vf,
            file_name=f"annotated_{out_id}.mp4",   # change ext if AVI
            mime="video/mp4"                       # or "video/x-msvideo"
        )

    # you can still TRY to play it inline (optional)
    # st.video(vf.read())   # remove or keep as you like


    # show the annotated video
    # st.video(out_video)
    with open(out_video, "rb") as vf:
        st.video(vf.read())          # send bytes → works even on localhost


    # load stats
    with open(out_json, encoding="utf-8") as f:
        stats = json.load(f)

    # overwrite names from roster
    for pid, row in stats.items():
        row["name"] = id_map.get(int(pid), row["name"])

    df = pd.DataFrame(stats).T  # transpose for nicer view
    st.subheader("Per-player statistics")
    st.dataframe(df.style.format({"speed_kmh": "{:.2f}", "distance_m": "{:.1f}", "risk_pct": "{:.1f}"}))

    # thermometer + alerts
    st.subheader("Risk gauges")
    for pid, row in stats.items():
        col1, col2 = st.columns([1, 9])
        with col1:
            st.write(f"**{row['name']}**")
        with col2:
            bar_color = "#22c55e"   # green
            if row["risk_pct"] >= 50:
                bar_color = "#facc15"  # amber
            if row["risk_pct"] >= 70:
                bar_color = "#ef4444"  # red
            percent = row["risk_pct"] / 100
            st.progress(percent, text=f"{row['risk_pct']} %")

        if row["risk_pct"] >= 63:
            st.error(f"⚠️ Substitute {row['name']} NOW – risk {row['risk_pct']:.1f}%")