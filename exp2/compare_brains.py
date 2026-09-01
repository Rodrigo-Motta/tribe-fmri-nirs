"""Compare TRIBE v2 brain predictions for the two `stimulus_2` audio clips and
test the hypothesis:

    H: the biggest differences in predicted brain activity between
       `bumbo.aifc` and `bumboTarol.aifc` are located in the medial
       prefrontal cortex (mPFC).

What it does
------------
1. Load the per-stimulus predictions (n_timesteps, 20484) from
   `exp2/output/<stem>/predictions.npy`.
2. Reduce each to a single vertex-wise activation profile:
      - mean over time (average predicted BOLD per vertex), and
      - peak over time (max predicted BOLD per vertex).
3. Build a vertex-wise *difference map*: |mean_bumbo - mean_tarol|.
4. Parcellate the difference map with the Schaefer 2018 atlas (200 ROIs, 7
   networks) on fsaverage5, compute the mean absolute difference per ROI, and
   rank ROIs to see where the largest differences live.
5. Identify mPFC. We use two complementary definitions:
      (a) Schaefer ROIs whose Yeo-7 network label is "Default" (the medial
          prefrontal cortex is a core Default-network hub) — this is a
          liberal, network-level proxy for mPFC.
      (b) A focal mPFC mask from the Destrieux atlas: the medial orbital
          front and frontomargin / anterior cingulate parcels
          ("G_front_med", "G_and_S_frontomargin", "G_and_S_paracentral"
           are excluded; we keep the orbital/medial-frontal ones).
6. Report:
      - the top-20 Schaefer ROIs by |difference|,
      - the rank/percentile of mPFC ROIs within that ranking,
      - the fraction of total cortical difference mass that falls inside mPFC,
      - a verdict (supports / does not support the hypothesis).
7. Render cortical surface maps to PNG:
      - mean activation per stimulus (medial view, both hemis),
      - the difference map (medial + lateral views),
      - the Schaefer ROI ranking highlighted on the surface.

Outputs go to `exp2/output/`:
    difference_map.npy          : (20484,) |mean_bumbo - mean_tarol|
    schaefer_roi_table.csv      : ROI-level difference ranking
    mpfc_report.txt             : human-readable verdict
    plot_mean_bumbo.png
    plot_mean_tarol.png
    plot_difference_medial.png
    plot_difference_lateral.png
    plot_roi_top20.png

Usage:
    python exp2/compare_brains.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "exp2" / "output"

STIM_A = "bumbo"        # bumbo.aifc
STIM_B = "bumboTarol"   # bumboTarol.aifc


# ----------------------------------------------------------------------------
# 1. Load predictions
# ----------------------------------------------------------------------------
def load_preds(stem: str) -> np.ndarray:
    p = OUT_ROOT / stem / "predictions.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run `python exp2/run_tribe_audio.py` first.")
    return np.load(p).astype(np.float32)


preds_a = load_preds(STIM_A)
preds_b = load_preds(STIM_B)
n_vert = preds_a.shape[1]
assert preds_b.shape[1] == n_vert, "vertex count mismatch"
print(f"[compare] {STIM_A}: {preds_a.shape}  {STIM_B}: {preds_b.shape}")

# ----------------------------------------------------------------------------
# 2. Vertex-wise summaries
# ----------------------------------------------------------------------------
mean_a = preds_a.mean(axis=0)          # (20484,)
mean_b = preds_b.mean(axis=0)
peak_a = preds_a.max(axis=0)
peak_b = preds_b.max(axis=0)

diff_map = np.abs(mean_a - mean_b)      # (20484,) >= 0
np.save(OUT_ROOT / "difference_map.npy", diff_map)
print(f"[compare] diff_map  min={diff_map.min():.4g} "
      f"max={diff_map.max():.4g} mean={diff_map.mean():.4g}")

# ----------------------------------------------------------------------------
# 3. Schaefer 2018 parcellation on fsaverage5
# ----------------------------------------------------------------------------
from nilearn import datasets, surface  # noqa: E402

schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7,
                                             resolution_mm=2)
# `schaefer.maps` is a NIfTI volume. Project it onto the fsaverage5 surface
# (left + right) to get a per-vertex label matching the TRIBE predictions
# (first 10242 vertices = left hemisphere, next 10242 = right).
fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
half = n_vert // 2
labels = np.zeros(n_vert, dtype=int) - 1
for i, hemi in enumerate(("left", "right")):
    proj = surface.vol_to_surf(schaefer.maps, surf_mesh=getattr(fsa, f"pial_{hemi}"),
                               inner_mesh=getattr(fsa, f"white_{hemi}"),
                               radius=0)
    labels[i * half:(i + 1) * half] = np.rint(proj).astype(int)

roi_ids = np.array([l for l in np.unique(labels) if l >= 0])
roi_names = [schaefer.labels[l] for l in roi_ids]  # bytes -> handle below
roi_names = [n.decode() if isinstance(n, bytes) else str(n) for n in roi_names]

# Per-ROI mean absolute difference and mean signed difference
roi_diff = np.zeros(len(roi_ids))
roi_signed = np.zeros(len(roi_ids))
roi_size = np.zeros(len(roi_ids), dtype=int)
for i, rid in enumerate(roi_ids):
    mask = labels == rid
    roi_diff[i] = diff_map[mask].mean()
    roi_signed[i] = (mean_a[mask] - mean_b[mask]).mean()
    roi_size[i] = int(mask.sum())

# Parse Yeo-7 network from Schaefer label, e.g.
#  "7Networks_LH_Default_Temp_1" -> "Default"
def parse_network(name: str) -> str:
    parts = name.split("_")
    # labels look like 7Networks_LH_<Network>_<Region>_<Idx>
    if len(parts) >= 4 and parts[0].startswith("7Networks"):
        return parts[2]
    return "?"

networks = np.array([parse_network(n) for n in roi_names])

import pandas as pd
roi_table = pd.DataFrame({
    "roi": roi_ids,
    "name": roi_names,
    "network": networks,
    "n_vertices": roi_size,
    "mean_abs_diff": roi_diff,
    "mean_signed_diff_bumbo_minus_tarol": roi_signed,
})
roi_table = roi_table.sort_values("mean_abs_diff", ascending=False).reset_index(drop=True)
roi_table.insert(0, "rank", np.arange(1, len(roi_table) + 1))
roi_table.to_csv(OUT_ROOT / "schaefer_roi_table.csv", index=False)

# ----------------------------------------------------------------------------
# 4. mPFC definition
# ----------------------------------------------------------------------------
# (a) Network-level proxy: Default network contains medial prefrontal cortex.
default_mask = networks == "Default"
default_names = [n for n, d in zip(roi_names, default_mask) if d]

# (b) Focal mPFC mask from Destrieux atlas: medial orbital frontal +
#     frontomargin parcels (anterior mPFC). Projected onto fsaverage5.
destrieux = datasets.fetch_atlas_destrieux_2009()
# `labels` is a list of strings; the list index is the atlas value.
mpfc_tokens = ("G_front_med", "G_and_S_frontomargin", "G_rectus",
               "S_front_inf", "G_front_middle")  # medial/orbital front
mpfc_labels_set = set()
for idx, lab in enumerate(destrieux.labels):
    lab_s = lab.decode() if isinstance(lab, bytes) else str(lab)
    if any(tok in lab_s for tok in mpfc_tokens):
        mpfc_labels_set.add(int(idx))

mpfc_vertex = np.zeros(n_vert, dtype=bool)
for i, hemi in enumerate(("left", "right")):
    proj = surface.vol_to_surf(destrieux.maps, surf_mesh=getattr(fsa, f"pial_{hemi}"),
                               inner_mesh=getattr(fsa, f"white_{hemi}"),
                               radius=0)
    proj = np.rint(proj).astype(int)
    mpfc_vertex[i * half:(i + 1) * half] = np.isin(proj, list(mpfc_labels_set))

print(f"[compare] Destrieux mPFC mask: {mpfc_vertex.sum()} vertices "
      f"({100*mpfc_vertex.sum()/n_vert:.1f}% of cortex)")
print(f"[compare] Default-network ROIs: {int(default_mask.sum())}  "
      f"{default_names}")

# ----------------------------------------------------------------------------
# 5. Quantify the hypothesis
# ----------------------------------------------------------------------------
# Rank of mPFC: where do the Default-network ROIs sit in the absolute-diff
# ranking? And what fraction of total cortical difference mass is in mPFC?
default_rank_mean = roi_table.loc[roi_table.network == "Default", "rank"].mean()
default_rank_median = roi_table.loc[roi_table.network == "Default", "rank"].median()
# (column access via ["rank"] is safe; attribute access .rank collides with
# the pandas method of the same name.)
n_rois = len(roi_table)
rk = roi_table["rank"]
default_top10 = ((roi_table.network == "Default") & (rk <= 10)).sum()
default_top20 = ((roi_table.network == "Default") & (rk <= 20)).sum()

total_diff = diff_map.sum()
mpfc_diff_mass = diff_map[mpfc_vertex].sum()
mpfc_fraction = mpfc_diff_mass / total_diff if total_diff > 0 else 0.0
mpfc_share_of_cortex = mpfc_vertex.mean()
mpfc_enrichment = (mpfc_fraction / mpfc_share_of_cortex
                   if mpfc_share_of_cortex > 0 else 0.0)

# Mean abs diff inside vs outside mPFC
mpfc_mean_diff = diff_map[mpfc_vertex].mean()
non_mpfc_mean_diff = diff_map[~mpfc_vertex].mean()
mpfc_ratio = mpfc_mean_diff / non_mpfc_mean_diff if non_mpfc_mean_diff > 0 else 0.0

# Is the single largest-difference ROI a Default-network ROI?
top1_name = roi_table.iloc[0]["name"]
top1_net = roi_table.iloc[0]["network"]

print("\n================ mPFC HYPOTHESIS REPORT ================")
print(f"Top-1 ROI by |difference|: {top1_name}  (network={top1_net})")
print(f"Default-network ROIs in top-10: {default_top10}")
print(f"Default-network ROIs in top-20: {default_top20}")
print(f"Default-network mean rank: {default_rank_mean:.1f} / {n_rois} "
      f"(median {default_rank_median:.0f})")
print(f"mPFC (Destrieux) diff mass fraction: {mpfc_fraction:.3f} "
      f"(share of cortex {mpfc_share_of_cortex:.3f}, "
      f"enrichment {mpfc_enrichment:.2f}x)")
print(f"Mean |diff| inside mPFC:  {mpfc_mean_diff:.5f}")
print(f"Mean |diff| outside mPFC: {non_mpfc_mean_diff:.5f}")
print(f"Ratio (in/out): {mpfc_ratio:.2f}")

# Verdict heuristic
verdict_points = 0
verdict_reasons = []
if top1_net == "Default":
    verdict_points += 2
    verdict_reasons.append("largest-difference ROI is a Default-network ROI")
if default_top10 >= 2:
    verdict_points += 2
    verdict_reasons.append(f"{default_top10} Default-network ROIs in the top-10")
if mpfc_enrichment > 1.3:
    verdict_points += 2
    verdict_reasons.append(
        f"mPFC over-represented among high-difference vertices "
        f"({mpfc_enrichment:.2f}x its cortical share)")
if mpfc_ratio > 1.2:
    verdict_points += 1
    verdict_reasons.append(
        f"mean |diff| inside mPFC exceeds outside ({mpfc_ratio:.2f}x)")

if verdict_points >= 5:
    verdict = "SUPPORTS the mPFC hypothesis"
elif verdict_points >= 3:
    verdict = "PARTIALLY supports the mPFC hypothesis"
else:
    verdict = "does NOT support the mPFC hypothesis"

print(f"\nVERDICT: {verdict}")
for r in verdict_reasons:
    print(f"   - {r}")

# Save report
with open(OUT_ROOT / "mpfc_report.txt", "w") as f:
    f.write("mPFC HYPOTHESIS REPORT — stimulus_2 (bumbo vs bumboTarol)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Stimulus A: {STIM_A}  preds {preds_a.shape}\n")
    f.write(f"Stimulus B: {STIM_B}  preds {preds_b.shape}\n\n")
    f.write(f"Top-1 ROI: {top1_name} (network={top1_net})\n")
    f.write(f"Default-network ROIs in top-10: {default_top10}\n")
    f.write(f"Default-network ROIs in top-20: {default_top20}\n")
    f.write(f"Default-network mean rank: {default_rank_mean:.1f} / {n_rois} "
            f"(median {default_rank_median:.0f})\n")
    f.write(f"mPFC diff-mass fraction: {mpfc_fraction:.3f} "
            f"(share {mpfc_share_of_cortex:.3f}, "
            f"enrichment {mpfc_enrichment:.2f}x)\n")
    f.write(f"Mean |diff| in mPFC:  {mpfc_mean_diff:.5f}\n")
    f.write(f"Mean |diff| out mPFC: {non_mpfc_mean_diff:.5f}\n")
    f.write(f"Ratio in/out: {mpfc_ratio:.2f}\n\n")
    f.write(f"VERDICT: {verdict}\n")
    for r in verdict_reasons:
        f.write(f"   - {r}\n")
    f.write("\nTop-20 ROIs by |mean difference|:\n")
    f.write(roi_table.head(20).to_string(index=False))
    f.write("\n")

print(f"\n[compare] saved schaefer_roi_table.csv, mpfc_report.txt, difference_map.npy")

# ----------------------------------------------------------------------------
# 6. Surface plots
# ----------------------------------------------------------------------------
from tribev2.plotting import PlotBrain  # noqa: E402

plotter = PlotBrain(mesh="fsaverage5")


def safe_plot(data, view_left, view_right, title, fname,
             cmap="hot", norm_percentile=95):
    """Render one figure with two rows (left hemi view, right hemi view)."""
    try:
        d = data[np.newaxis]  # (1, n_vertices) — single timestep
        neuro = {"Left": d, "Right": d}
        views = {"Left": view_left, "Right": view_right}
        fig = plotter.plot_timesteps(neuro,
                                     views=views,
                                     norm_percentile=norm_percentile,
                                     cmap=cmap)
        fig.suptitle(title)
        fig.savefig(OUT_ROOT / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare] saved {fname}")
    except Exception as e:
        print(f"[compare] plot {fname} skipped: {e!r}")


# Per-stimulus mean activation (medial view to see mPFC)
safe_plot(mean_a, "medial_left", "medial_right",
          f"Mean activation — {STIM_A}", "plot_mean_bumbo.png", cmap="hot")
safe_plot(mean_b, "medial_left", "medial_right",
          f"Mean activation — {STIM_B}", "plot_mean_tarol.png", cmap="hot")

# Difference map: medial (mPFC visible) + lateral (whole-brain context)
safe_plot(diff_map, "medial_left", "medial_right",
          "|mean difference| (medial)", "plot_difference_medial.png",
          cmap="hot", norm_percentile=95)
safe_plot(diff_map, "left", "right",
          "|mean difference| (lateral)", "plot_difference_lateral.png",
          cmap="hot", norm_percentile=95)

# Highlight top-20 ROIs on the surface (1 = top-20 ROI, 0 = else)
top20_ids = set(roi_table.head(20)["roi"].tolist())
highlight = np.isin(labels, list(top20_ids)).astype(np.float32)
safe_plot(highlight, "medial_left", "medial_right",
          "Top-20 ROIs by |difference| (medial)", "plot_roi_top20.png",
          cmap="Reds", norm_percentile=99)

print("\n[compare] done.")
