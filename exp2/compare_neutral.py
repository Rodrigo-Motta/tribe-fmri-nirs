"""Test the mPFC hypothesis on the neutral-video TRIBE predictions.

Hypothesis: the biggest differences in predicted brain activity between
`bumbo` and `bumboTarol` percussion clips are in the medial prefrontal
cortex (mPFC).

This version is purely numerical (no plots) and adds:
  - a proper null comparison: is the mPFC difference larger than expected
    by chance, relative to same-sized non-mPFC regions?
  - a signed-difference analysis (which direction does mPFC go?)
  - comparison against the original audio-only run for reference.

Reads:
    exp2/output/bumbo/predictions.npy      (neutral video)
    exp2/output/bumboTarol/predictions.npy (neutral video)
    exp2/output/_audio_only/bumbo.npy      (old, for comparison)
    exp2/output/_audio_only/bumboTarol.npy

Writes:
    exp2/output/mpfc_report_neutral.txt
    exp2/output/schaefer_roi_table_neutral.csv
    exp2/output/difference_map_neutral.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets, surface

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "exp2" / "output"
STIM_A, STIM_B = "bumbo", "bumboTarol"

# ----------------------------------------------------------------------------
# Load predictions
# ----------------------------------------------------------------------------
def load(stem):
    p = OUT_ROOT / stem / "predictions.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run run_tribe_neutral.py first.")
    return np.load(p).astype(np.float32)

preds_a = load(STIM_A)   # neutral grey + bumbo
preds_b = load(STIM_B)   # neutral grey + tarol
preds_a_old = np.load(OUT_ROOT / "_audio_only" / "bumbo.npy")
preds_b_old = np.load(OUT_ROOT / "_audio_only" / "bumboTarol.npy")
n_vert = preds_a.shape[1]
assert preds_b.shape[1] == n_vert
half = n_vert // 2
print(f"[compare] neutral: {STIM_A} {preds_a.shape}, {STIM_B} {preds_b.shape}")
print(f"[compare] audio-only (ref): {STIM_A} {preds_a_old.shape}, {STIM_B} {preds_b_old.shape}")

# Vertex-wise summaries
mean_a = preds_a.mean(0)
mean_b = preds_b.mean(0)
diff_map = np.abs(mean_a - mean_b)           # (20484,)
signed_map = mean_a - mean_b                 # bumbo - tarol
np.save(OUT_ROOT / "difference_map_neutral.npy", diff_map)

# ----------------------------------------------------------------------------
# Parcellate with Schaefer 2018 (200 ROIs, 7 networks) on fsaverage5
# ----------------------------------------------------------------------------
fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
labels = np.zeros(n_vert, dtype=int) - 1
for i, hemi in enumerate(("left", "right")):
    proj = surface.vol_to_surf(sch.maps, surf_mesh=getattr(fsa, f"pial_{hemi}"),
                               inner_mesh=getattr(fsa, f"white_{hemi}"), radius=0)
    labels[i * half:(i + 1) * half] = np.rint(proj).astype(int)
roi_names = [n.decode() if isinstance(n, bytes) else str(n) for n in sch.labels]


def net_of(name):
    parts = name.split("_")
    return parts[2] if len(parts) >= 4 and parts[0].startswith("7Networks") else "?"


roi_ids = [int(l) for l in np.unique(labels) if l >= 0]
rows = []
for rid in roi_ids:
    mask = labels == rid
    name = roi_names[rid]
    rows.append({
        "roi": rid,
        "name": name,
        "network": net_of(name),
        "n_vertices": int(mask.sum()),
        "mean_abs_diff": float(diff_map[mask].mean()),
        "mean_signed_diff": float(signed_map[mask].mean()),
        "mean_act_bumbo": float(mean_a[mask].mean()),
        "mean_act_tarol": float(mean_b[mask].mean()),
    })
roi_table = pd.DataFrame(rows).sort_values("mean_abs_diff", ascending=False).reset_index(drop=True)
roi_table.insert(0, "rank", np.arange(1, len(roi_table) + 1))
roi_table.to_csv(OUT_ROOT / "schaefer_roi_table_neutral.csv", index=False)

# ----------------------------------------------------------------------------
# mPFC definitions
# ----------------------------------------------------------------------------
# (a) Default network (contains mPFC as a core hub) — liberal proxy
default_mask = roi_table["network"].values == "Default"

# (b) Focal mPFC from Destrieux: medial orbital frontal + frontomargin + rectus
destrieux = datasets.fetch_atlas_destrieux_2009()
mpfc_tokens = ("G_front_med", "G_and_S_frontomargin", "G_rectus", "S_front_inf", "G_front_middle")
mpfc_labels_set = set()
for idx, lab in enumerate(destrieux.labels):
    lab_s = lab.decode() if isinstance(lab, bytes) else str(lab)
    if any(tok in lab_s for tok in mpfc_tokens):
        mpfc_labels_set.add(int(idx))
mpfc_vertex = np.zeros(n_vert, dtype=bool)
for i, hemi in enumerate(("left", "right")):
    proj = surface.vol_to_surf(destrieux.maps, surf_mesh=getattr(fsa, f"pial_{hemi}"),
                               inner_mesh=getattr(fsa, f"white_{hemi}"), radius=0)
    proj = np.rint(proj).astype(int)
    mpfc_vertex[i * half:(i + 1) * half] = np.isin(proj, list(mpfc_labels_set))

# Also: specifically the PFC-labeled Schaefer ROIs (narrower than Default)
pfc_mask = np.array(["_PFC" in n for n in roi_table["name"].values])

# ----------------------------------------------------------------------------
# Quantify
# ----------------------------------------------------------------------------
rk = roi_table["rank"]
n_rois = len(roi_table)

# ROI-level: where do mPFC ROIs rank?
def rank_stats(mask, label):
    if mask.sum() == 0:
        return f"{label}: (none)"
    ranks = rk[mask]
    return (f"{label}: {int(mask.sum())} ROIs, "
            f"mean rank {ranks.mean():.1f}/{n_rois}, "
            f"median {int(ranks.median())}, "
            f"top-10: {int((ranks <= 10).sum())}, top-20: {int((ranks <= 20).sum())}")

# Vertex-level: diff mass fraction inside mPFC
total_diff = diff_map.sum()
mpfc_diff_mass = diff_map[mpfc_vertex].sum()
mpfc_fraction = mpfc_diff_mass / total_diff if total_diff > 0 else 0
mpfc_share = mpfc_vertex.mean()
mpfc_enrichment = mpfc_fraction / mpfc_share if mpfc_share > 0 else 0
mpfc_mean_diff = diff_map[mpfc_vertex].mean()
non_mpfc_mean_diff = diff_map[~mpfc_vertex].mean()
mpfc_ratio = mpfc_mean_diff / non_mpfc_mean_diff if non_mpfc_mean_diff > 0 else 0

# ----------------------------------------------------------------------------
# NULL TEST: is the mPFC difference larger than chance?
# ----------------------------------------------------------------------------
# Build a null distribution by sampling random same-sized vertex sets and
# computing their mean |diff|. If mPFC's mean |diff| exceeds the 95th
# percentile of this null, it's statistically enriched.
rng = np.random.default_rng(42)
n_mpfc = int(mpfc_vertex.sum())
null_means = np.zeros(5000)
all_diffs = diff_map.copy()
for i in range(5000):
    idx = rng.choice(n_vert, size=n_mpfc, replace=False)
    null_means[i] = all_diffs[idx].mean()
null_p95 = np.percentile(null_means, 95)
null_mean = null_means.mean()
mpfc_pvalue = float((null_means >= mpfc_mean_diff).mean())

# Also: ROI-level null — rank of mPFC ROIs vs random ROI subsets of same size
# (how many random subsets of the same size would have a better mean rank?)
n_default = int(default_mask.sum())
default_ranks = rk[default_mask].values
default_mean_rank = float(default_ranks.mean())
# null: pick n_default random ROIs, compute their mean rank, repeat
null_roi_ranks = np.zeros(10000)
for i in range(10000):
    null_roi_ranks[i] = rng.choice(np.arange(1, n_rois + 1), size=n_default, replace=False).mean()
default_rank_pvalue = float((null_roi_ranks <= default_mean_rank).mean())

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
top1 = roi_table.iloc[0]
lines = []
lines.append("=" * 70)
lines.append("mPFC HYPOTHESIS TEST — neutral-video TRIBE predictions")
lines.append("=" * 70)
lines.append("")
lines.append(f"Stimulus A: {STIM_A} (neutral grey + bumbo audio)  {preds_a.shape}")
lines.append(f"Stimulus B: {STIM_B} (neutral grey + tarol audio)  {preds_b.shape}")
lines.append(f"Difference map: |mean_bumbo - mean_tarol|  "
             f"(max={diff_map.max():.5f}, mean={diff_map.mean():.5f})")
lines.append("")
lines.append("ROI-LEVEL RANKING (Schaefer 200, by mean |difference|):")
lines.append(f"  {rank_stats(default_mask, 'Default network (mPFC hub proxy)')}")
lines.append(f"  {rank_stats(pfc_mask, 'PFC-labeled Schaefer ROIs')}")
lines.append(f"  Top-1 ROI: {top1['name']} (network={top1['network']}, "
             f"|diff|={top1['mean_abs_diff']:.5f})")
lines.append("")
lines.append("VERTEX-LEVEL (Destrieux mPFC mask, %d vertices, %.1f%% of cortex):"
             % (n_mpfc, 100 * mpfc_share))
lines.append(f"  Mean |diff| inside mPFC:  {mpfc_mean_diff:.5f}")
lines.append(f"  Mean |diff| outside mPFC: {non_mpfc_mean_diff:.5f}")
lines.append(f"  Ratio in/out:            {mpfc_ratio:.3f}")
lines.append(f"  Diff-mass fraction in mPFC: {mpfc_fraction:.4f} "
             f"(enrichment {mpfc_enrichment:.2f}x)")
lines.append("")
lines.append("NULL TEST (vertex-level, 5000 random same-sized samples):")
lines.append(f"  Null mean |diff|:    {null_mean:.5f}  (p95: {null_p95:.5f})")
lines.append(f"  mPFC mean |diff|:    {mpfc_mean_diff:.5f}")
lines.append(f"  p-value (one-sided): {mpfc_pvalue:.4f}")
lines.append(f"  => mPFC {'IS' if mpfc_pvalue < 0.05 else 'is NOT'} "
             f"statistically enriched in differences (alpha=0.05)")
lines.append("")
lines.append("NULL TEST (ROI-level, Default network mean rank vs 10000 random subsets):")
lines.append(f"  Default mean rank:   {default_mean_rank:.1f}/{n_rois}")
lines.append(f"  Null mean rank:      {null_roi_ranks.mean():.1f}")
lines.append(f"  p-value (one-sided): {default_rank_pvalue:.4f}")
lines.append(f"  => Default network ROIs {'DO' if default_rank_pvalue < 0.05 else 'do NOT'} "
             f"rank significantly higher than chance")
lines.append("")

# Verdict
points = 0
reasons = []
if mpfc_pvalue < 0.05:
    points += 3; reasons.append(f"mPFC vertex-level enrichment is significant (p={mpfc_pvalue:.4f})")
elif mpfc_ratio > 1.2:
    points += 1; reasons.append(f"mPFC mean |diff| > outside ({mpfc_ratio:.2f}x)")
else:
    reasons.append(f"mPFC not enriched (ratio {mpfc_ratio:.2f}, p={mpfc_pvalue:.4f})")

if default_rank_pvalue < 0.05:
    points += 2; reasons.append(f"Default-network ROIs rank significantly high (p={default_rank_pvalue:.4f})")
else:
    reasons.append(f"Default-network ROIs not significantly high-ranked (p={default_rank_pvalue:.4f})")

if top1["network"] == "Default":
    points += 1; reasons.append("top-1 ROI is Default-network")

if points >= 5:
    verdict = "SUPPORTS the mPFC hypothesis"
elif points >= 3:
    verdict = "PARTIALLY supports the mPFC hypothesis"
else:
    verdict = "does NOT support the mPFC hypothesis"

lines.append(f"VERDICT: {verdict}")
for r in reasons:
    lines.append(f"   - {r}")
lines.append("")
lines.append("Top-20 ROIs by |mean difference|:")
lines.append(roi_table.head(20)[["rank", "name", "network", "mean_abs_diff",
                                  "mean_signed_diff"]].to_string(index=False))
lines.append("")
lines.append("Network-level summary (mean |diff| averaged across ROIs in each network):")
net_summary = roi_table.groupby("network")["mean_abs_diff"].agg(["mean", "count"]).sort_values("mean", ascending=False)
lines.append(net_summary.to_string())
lines.append("")

report = "\n".join(lines)
print(report)
(OUT_ROOT / "mpfc_report_neutral.txt").write_text(report)
print(f"\n[compare] saved mpfc_report_neutral.txt, schaefer_roi_table_neutral.csv, "
      f"difference_map_neutral.npy")
