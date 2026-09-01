"""Render the per-TR difference TIMECOURSE (one brain per second).

This directly answers "why is the diff just at t=0?" — by showing the
difference is NOT just at t=0; it is spread across all TRs and grows over
time as the two percussion patterns diverge.

Reads the neutral-video + audio-only predictions and saves, per run, a single
wide figure with one brain per TR (medial left + medial right views) for:
    |pred_bumbo(t) - pred_bumboTarol(t)|   for t = 0 .. N-1

Outputs (exp2/output/):
    plot_diff_timecourse_neutral.png
    plot_diff_timecourse_audio_only.png
    plot_mean_timecourse_bumbo_neutral.png   (sanity: mean activation over time)
    plot_mean_timecourse_tarol_neutral.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tribev2.plotting import PlotBrain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "exp2" / "output"
AO = OUT / "_audio_only"

plotter = PlotBrain(mesh="fsaverage5")


def render_timecourse(per_tr, title, fname, cmap="hot", norm_percentile=95):
    """per_tr: (n_timesteps, 20484) -> one brain per TR, two rows (L/R medial)."""
    try:
        n = per_tr.shape[0]
        neuro = {"Left": per_tr, "Right": per_tr}
        views = {"Left": "medial_left", "Right": "medial_right"}
        timestamps = list(range(n))
        fig = plotter.plot_timesteps(neuro, views=views, timestamps=timestamps,
                                     norm_percentile=norm_percentile, cmap=cmap)
        fig.suptitle(title, y=1.02)
        fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[tc] saved {fname}  ({n} TRs)")
    except Exception as e:
        print(f"[tc] {fname} skipped: {e!r}")


# --- neutral video --------------------------------------------------------
a = np.load(OUT / "bumbo" / "predictions.npy").astype(np.float32)
b = np.load(OUT / "bumboTarol" / "predictions.npy").astype(np.float32)
n = min(a.shape[0], b.shape[0])
diff_n = np.abs(a[:n] - b[:n])                 # (n, 20484) per-TR |diff|
render_timecourse(diff_n,
                  "Neutral video — |pred_bumbo(t) - pred_tarol(t)| per TR (medial)",
                  "plot_diff_timecourse_neutral.png")
render_timecourse(a[:n],
                  "Neutral video — mean activation bumbo per TR (medial)",
                  "plot_mean_timecourse_bumbo_neutral.png")
render_timecourse(b[:n],
                  "Neutral video — mean activation tarol per TR (medial)",
                  "plot_mean_timecourse_tarol_neutral.png")

# --- audio only -----------------------------------------------------------
ao_a = np.load(AO / "bumbo.npy").astype(np.float32)
ao_b = np.load(AO / "bumboTarol.npy").astype(np.float32)
n2 = min(ao_a.shape[0], ao_b.shape[0])
diff_ao = np.abs(ao_a[:n2] - ao_b[:n2])
render_timecourse(diff_ao,
                  "Audio-only — |pred_bumbo(t) - pred_tarol(t)| per TR (medial)",
                  "plot_diff_timecourse_audio_only.png")

print("[tc] done.")
