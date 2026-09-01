"""Render cortical surface images from the AUDIO-ONLY TRIBE predictions.

Reads the audio-only predictions backed up in exp2/output/_audio_only/ and
saves PNG surface maps to exp2/output/audio_only_plots/:

    plot_mean_bumbo.png          mean activation, bumbo (medial)
    plot_mean_tarol.png          mean activation, bumboTarol (medial)
    plot_difference_medial.png   |mean_bumbo - mean_tarol| (medial)
    plot_difference_lateral.png  |mean_bumbo - mean_tarol| (lateral)

Pure plotting, no analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tribev2.plotting import PlotBrain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "exp2" / "output"
AUDIO_ONLY = OUT_ROOT / "_audio_only"
PLOT_DIR = OUT_ROOT / "audio_only_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

mean_a = np.load(AUDIO_ONLY / "bumbo.npy").astype(np.float32).mean(0)
mean_b = np.load(AUDIO_ONLY / "bumboTarol.npy").astype(np.float32).mean(0)
diff_map = np.abs(mean_a - mean_b)

plotter = PlotBrain(mesh="fsaverage5")


def safe_plot(data, view_left, view_right, title, fname,
              cmap="hot", norm_percentile=95):
    try:
        d = data[np.newaxis]
        neuro = {"Left": d, "Right": d}
        views = {"Left": view_left, "Right": view_right}
        fig = plotter.plot_timesteps(neuro, views=views,
                                     norm_percentile=norm_percentile, cmap=cmap)
        fig.suptitle(title)
        fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved {fname}")
    except Exception as e:
        print(f"[plot] {fname} skipped: {e!r}")


safe_plot(mean_a, "medial_left", "medial_right",
          "Audio-only — mean activation: bumbo (medial)",
          "plot_mean_bumbo.png", cmap="hot")
safe_plot(mean_b, "medial_left", "medial_right",
          "Audio-only — mean activation: bumboTarol (medial)",
          "plot_mean_tarol.png", cmap="hot")
safe_plot(diff_map, "medial_left", "medial_right",
          "Audio-only — |mean difference| (medial)",
          "plot_difference_medial.png", cmap="hot", norm_percentile=95)
safe_plot(diff_map, "left", "right",
          "Audio-only — |mean difference| (lateral)",
          "plot_difference_lateral.png", cmap="hot", norm_percentile=95)

print("[plot] done.")
