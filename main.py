"""
main.py – Football-analysis end-to-end runner
(v2025-04 – includes Risk %, stub-cache, and JSON stats dump)

Typical usage
─────────────
# 1️⃣ First pass – runs YOLO + camera flow, writes pickles + video + stats
python main.py \
  --input  input_videos/your_match.mp4 \
  --weights models/best.pt \
  --output output_videos/annotated.mp4 \
  --stats_out output_videos/annotated_stats.json

# 2️⃣ Faster re-run – reuses the pickles
python main.py ... --use_stubs
"""

import argparse
import json
import os
import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from fatigue_risk_estimator.fatigue_risk_estimator import FatigueRiskEstimator


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(args):
    # 1. Read frames
    video_frames = read_video(args.input)
    if not video_frames:
        raise RuntimeError(f"No frames read from {args.input}")

    # 2. Detect & track
    tracker = Tracker(args.weights)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path="stubs/track_stubs.pkl",
    )
    tracker.add_position_to_tracks(tracks)

    # 3. Camera-motion
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_flow = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path="stubs/camera_movement_stub.pkl",
    )
    cam_est.add_adjust_positions_to_tracks(tracks, cam_flow)

    # 4. Bird’s-eye transform
    ViewTransformer().add_transformed_position_to_tracks(tracks)

    # 5. Speed / distance
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    sd_est = SpeedAndDistance_Estimator()
    sd_est.add_speed_and_distance_to_tracks(tracks)

    # 6. Fatigue / injury-risk
    risk_est = FatigueRiskEstimator(fps=24)

    # 7. Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for f_idx, player_track in enumerate(tracks["players"]):
        for pid, trk in player_track.items():
            trk["team"] = team_assigner.get_player_team(
                video_frames[f_idx], trk["bbox"], pid
            )
            trk["team_color"] = team_assigner.team_colors[trk["team"]]

    # 8. Ball possession + per-frame risk update
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for f_idx, player_track in enumerate(tracks["players"]):
        # update risk metrics every frame
        risk_est.update_metrics(f_idx, player_track)

        if not tracks["ball"][f_idx]:
            team_ball_control.append("none" if f_idx == 0 else team_ball_control[-1])
            continue

        ball_bbox = tracks["ball"][f_idx][1]["bbox"]
        owner_pid = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if owner_pid != -1:
            player_track[owner_pid]["has_ball"] = True
            team_ball_control.append(player_track[owner_pid]["team"])
        else:
            team_ball_control.append("none" if f_idx == 0 else team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # 9. Draw overlays
    out_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    out_frames = cam_est.draw_camera_movement(out_frames, cam_flow)
    sd_est.draw_speed_and_distance(out_frames, tracks)

    # Risk % overlay (coloured text below speed/distance)
    for f_idx, frame in enumerate(out_frames):
        for trk in tracks["players"][f_idx].values():
            risk = trk.get("risk")
            if risk is None:
                continue
            FatigueRiskEstimator.draw(frame, trk["bbox"], risk)

    # 10. Save video
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_video(out_frames, args.output)
    print(f"[✓] Saved annotated video → {args.output}")

    # 11. Optional JSON stats  –  take the *last valid* values
    if args.stats_out:
        summary = {}
        for pid in risk_est.risk.keys():                 # iterate players we tracked
            # walk backwards until we find a frame with speed/distance
            speed, dist, risk = 0.0, 0.0, 0.0
            for f_back in reversed(range(len(tracks["players"]))):
                trk = tracks["players"][f_back].get(pid)
                if trk and trk.get("speed") is not None:
                    speed = trk["speed"]
                    dist  = trk["distance"]
                    risk  = trk.get("risk", 0.0)
                    break

            summary[str(pid)] = {
                "name":   tracks["players"][0].get(pid, {}).get("name", f"P{pid}"),
                "speed_kmh": round(speed, 2),
                "distance_m": round(dist, 2),
                "risk_pct":  round(risk, 1),
            }

        with open(args.stats_out, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        print(f"[✓] Stats JSON written → {args.stats_out}")



# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full football-analysis pipeline on a match video"
    )
    parser.add_argument("--input", required=True, help="input video path")
    parser.add_argument(
        "--weights", default="models/best.pt", help="YOLOv5/YOLOv8 weight file (.pt)"
    )
    parser.add_argument(
        "--output", default="output_videos/output.mp4", help="output annotated video path"
    )
    parser.add_argument(
        "--stats_out", default="stats.json", help="write per-player JSON summary"
    )
    parser.add_argument(
        "--use_stubs",
        action="store_true",
        help="reuse cached pickles for tracks / camera movement",
    )

    run_pipeline(parser.parse_args())