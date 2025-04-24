import numpy as np
from collections import defaultdict
import cv2

# --- CONFIG ------------------------------------------------------------------
# Thresholds & weights – tweak freely
SPRINT_SPEED      = 7          # m·s-1  (≈ 25 km·h-1)
ACCEL_THRESHOLD   = 2          # m·s-2
WINDOW_SEC        = 30         # rolling window for metrics

METRIC_WEIGHTS = {             # literature-driven weights
    "speed_drop":      0.30,   # :contentReference[oaicite:0]{index=0}
    "sprint_count":    0.25,   # :contentReference[oaicite:1]{index=1}
    "accel_events":    0.15,   # :contentReference[oaicite:2]{index=2}
    "posture_deg":     0.20,   # :contentReference[oaicite:3]{index=3}
    "stride_reduction":0.10    # :contentReference[oaicite:4]{index=4}
}
# -----------------------------------------------------------------------------


class FatigueRiskEstimator:
    """Computes a 0-100 fatigue / injury-risk % per player each frame."""

    def __init__(self, fps: int = 24):
        self.fps = fps
        self.window = WINDOW_SEC * fps
        self.history = defaultdict(list)   # {pid: [speed_mps, ...]}
        self.risk = defaultdict(float)     # {pid: last_risk}

    # --------------------------------------------------------------------- #
    # 1.  Update rolling metrics every frame                                #
    # --------------------------------------------------------------------- #
    def update_metrics(self, frame_idx, tracks):
        for pid, trk in tracks.items():
            spd = trk.get("speed")  # km/h
            if spd is None:
                continue
            spd_mps = spd / 3.6
            self.history[pid].append(spd_mps)

            # keep only WINDOW_SEC
            if len(self.history[pid]) > self.window:
                self.history[pid] = self.history[pid][-self.window:]

            # compute every WINDOW_SEC or last frame
            if frame_idx % self.window == 0 or frame_idx == 0:
                self.risk[pid] = self._compute_fatigue(pid)

            trk["risk"] = self.risk[pid]   # inject into main tracks

    # --------------------------------------------------------------------- #
    # 2.  Draw risk% under existing overlay                                 #
    # --------------------------------------------------------------------- #
    @staticmethod
    def draw(frame, bbox, risk):
        if risk is None:
            return
        x, y, _, _ = bbox
        y += 60    # below speed & distance
        txt = f"{risk:5.1f}% risk"
        color = (0, 0, 255) if risk >= 70 else (0, 165, 255) if risk >= 40 else (0, 200, 0)
        cv2.putText(frame, txt, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # --------------------------------------------------------------------- #
    # INTERNAL: weighted-score calculator                                   #
    # --------------------------------------------------------------------- #
    def _compute_fatigue(self, pid):
        speeds = np.array(self.history[pid])        # m·s-1

        if len(speeds) < 5:                         # not enough data
            return 0.0

        # --- speed-drop ----------------------------------------------------
        baseline = np.percentile(speeds, 90)        # “fresh” top speed
        recent   = np.mean(speeds[-self.window//2:])
        speed_drop = np.clip((baseline - recent) / baseline, 0, 1)

        # --- sprint count --------------------------------------------------
        sprint_count = np.sum(speeds > SPRINT_SPEED)
        sprint_norm  = np.clip(sprint_count / (WINDOW_SEC / 10), 0, 1)

        # --- acceleration events ------------------------------------------
        accel = np.diff(speeds) * self.fps
        accel_events = np.sum(accel > ACCEL_THRESHOLD)
        accel_norm   = np.clip(accel_events / (WINDOW_SEC / 5), 0, 1)

        # --- posture degradation & stride length --------------------------
        # Placeholder  → 0.0 (hook in your pose model later)
        posture_deg = 0.0
        stride_red  = 0.0

        # ---------- weighted fatigue score 0-1 -----------------------------
        metrics = {
            "speed_drop":      speed_drop,
            "sprint_count":    sprint_norm,
            "accel_events":    accel_norm,
            "posture_deg":     posture_deg,
            "stride_reduction":stride_red
        }
        score = sum(metrics[k] * METRIC_WEIGHTS[k] for k in METRIC_WEIGHTS)

        # OPTIONAL: logistic curve to emphasise higher risk
        risk_pct = 100 / (1 + np.exp(-10*(score-0.5)))
        return float(risk_pct)
