"""Render cortical surface images from the neutral-video TRIBE predictions.

Reads the neutral-video predictions produced by run_tribe_neutral.py and the
Schaefer ROI table from compare_neutral.py, and saves PNG surface maps to
exp2/output/:

    plot_mean_bumbo.png          mean activation, bumbo (medial view)
    plot_mean_tarol.png          mean activation, bumboTarol (medial view)
    plot_difference_medial.png   |mean_bumbo - mean_tarol| (medial)
    plot_difference_lateral.png  |mean_bumbo - mean_tarol| (lateral)
    plot_roi_top20.png           top-20 ROIs by |difference| (medial)

Pure plotting — no analysis. Run after compare_neutral.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, surface

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "exp2" / "output"
STIM_A, STIM_B = "bumbo", "bumboTarol"


def load(stem):
    return np.load(OUT_ROOT / stem / "predictions.npy").astype(np.float32)


preds_a = load(STIM_A)
preds_b = load(STIM_B)
mean_a = preds_a.mean(0)
mean_b = preds_b.mean(0)
diff_map = np.abs(mean_a - mean_b)
n_vert = diff_map.shape[0]
half = n_vert // 2

# Schaefer labels (for the top-20 ROI highlight)
fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
labels = np.zeros(n_vert, dtype=int) - 1
for i, hemi in enumerate(("left", "right")):
    proj = surface.vol_to_surf(sch.maps, surf_mesh=getattr(fsa, f"pial_{hemi}"),
                               inner_mesh=getattr(fsa, f"white_{hemi}"), radius=0)
    labels[i * half:(i + 1) * half] = np.rint(proj).astype(int)

roi_table = pd.read_csv(OUT_ROOT / "schaefer_roi_table_neutral.csv")
top20_ids = set(roi_table.head(20)["roi"].tolist())
highlight = np.isin(labels, list(top20_ids)).astype(np.float32)

# --- plot -----------------------------------------------------------------
from tribev2.plotting import PlotBrain

plotter = PlotBrain(mesh="fsaverage5")


def safe_plot(data, view_left, view_right, title, fname,
              cmap="hot", norm_percentile=95):
    """Render one figure with two rows: left-hemi view, right-hemi view."""
    try:
        d = data[np.newaxis]  # (1, n_vertices)
        neuro = {"Left": d, "Right": d}
        views = {"Left": view_left, "Right": view_right}
        fig = plotter.plot_timesteps(neuro, views=views,
                                     norm_percentile=norm_percentile, cmap=cmap)
        fig.suptitle(title)
        fig.savefig(OUT_ROOT / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved {fname}")
    except Exception as e:
        print(f"[plot] {fname} skipped: {e!r}")


safe_plot(mean_a, "medial_left", "medial_right",
          f"Mean activation — {STIM_A} (neutral video)", "plot_mean_bumbo.png", cmap="hot")
safe_plot(mean_b, "medial_left", "medial_right",
          f"Mean activation — {STIM_B} (neutral video)", "plot_mean_tarol.png", cmap="hot")
safe_plot(diff_map, "medial_left", "medial_right",
          "|mean difference| (medial) — bumbo vs bumboTarol",
          "plot_difference_medial.png", cmap="hot", norm_percentile=95)
safe_plot(diff_map, "left", "right",
          "|mean difference| (lateral) — bumbo vs bumboTarol",
          "plot_difference_lateral.png", cmap="hot", norm_percentile=95)
safe_plot(highlight, "medial_left", "medial_right",
          "Top-20 ROIs by |difference| (medial)",
          "plot_roi_top20.png", cmap="Reds", norm_percentile=99)

print("[plot] done.")
