"""Render only the STATISTICALLY SIGNIFICANT differences between the two clips.

Per-vertex two-sample t-test (bumbo TRs vs bumboTarol TRs) with FDR correction.
Renders thresholded maps where non-significant vertices are zeroed, so the
specific significant areas are clearly visible against the sulcal background.

Outputs (exp2/output/):
    plot_sig_diff_neutral_medial.png       FDR p<0.05, medial (mPFC)
    plot_sig_diff_neutral_lateral.png      FDR p<0.05, lateral
    plot_sig_diff_neural_strict_medial.png FDR p<0.01, medial (stricter)
    plot_sig_diff_audio_only_medial.png    audio-only, FDR p<0.05, medial
    sig_stats.txt                          per-network counts of sig. vertices
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from nilearn import datasets, surface
from tribev2.plotting import PlotBrain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "exp2" / "output"
AO = OUT / "_audio_only"
plotter = PlotBrain(mesh="fsaverage5")
N_VERT = 20484
HALF = N_VERT // 2


def fdr_mask(pvals, alpha):
    """Benjamini-Hochberg FDR mask (boolean, True = significant)."""
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    thresh = (ranks * alpha) / n
    passed = p[order] <= thresh
    # find largest rank that passes
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    sig = np.zeros(n, dtype=bool)
    sig[order[:kmax + 1]] = True
    return sig


def ttest_vertices(a, b):
    """Per-vertex Welch t-test. a: (n1, V), b: (n2, V)."""
    t, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=1.0)
    return t, p


def render(data, view_left, view_right, title, fname, cmap="hot", norm_percentile=99):
    try:
        d = data[np.newaxis]
        neuro = {"Left": d, "Right": d}
        views = {"Left": view_left, "Right": view_right}
        fig = plotter.plot_timesteps(neuro, views=views,
                                     norm_percentile=norm_percentile, cmap=cmap)
        fig.suptitle(title, y=1.02)
        fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[sig] saved {fname}")
    except Exception as e:
        print(f"[sig] {fname} skipped: {e!r}")


def network_labels():
    fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labels = np.zeros(N_VERT, dtype=int) - 1
    for i, hemi in enumerate(("left", "right")):
        proj = surface.vol_to_surf(sch.maps, surf_mesh=getattr(fsa, f"pial_{hemi}"),
                                    inner_mesh=getattr(fsa, f"white_{hemi}"), radius=0)
        labels[i * HALF:(i + 1) * HALF] = np.rint(proj).astype(int)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in sch.labels]

    def net_of(n):
        parts = n.split("_")
        return parts[2] if len(parts) >= 4 and parts[0].startswith("7Networks") else "?"
    net = np.array(["?"] * N_VERT, dtype=object)
    for rid in [int(l) for l in np.unique(labels) if l >= 0]:
        net[labels == rid] = net_of(names[rid])
    return net


net = network_labels()

# --- neutral video ---
a = np.load(OUT / "bumbo" / "predictions.npy").astype(np.float32)
b = np.load(OUT / "bumboTarol" / "predictions.npy").astype(np.float32)
t_n, p_n = ttest_vertices(a, b)
diff_n = np.abs(a.mean(0) - b.mean(0))

for alpha, suf in ((0.05, ""), (0.01, "_strict")):
    sig = fdr_mask(p_n, alpha)
    sig_map = np.where(sig, diff_n, 0.0)
    t_sig = np.where(sig, t_n, 0.0)
    n_sig = int(sig.sum())
    print(f"\n[neutral] FDR p<{alpha}: {n_sig} sig vertices "
          f"({100*n_sig/N_VERT:.1f}% of cortex)")
    if n_sig == 0:
        render(diff_n, "medial_left", "medial_right",
              f"Neutral |diff| — NO sig vertices at FDR p<{alpha} (showing raw)",
              f"plot_sig_diff_neutral{suf}_medial.png", cmap="hot", norm_percentile=95)
        continue
    # breakdown by network
    print("   by network (sig vertex count):")
    for nt in sorted(set(net[sig])):
        print(f"     {nt:12s} {int((net[sig]==nt).sum())}")
    render(sig_map, "medial_left", "medial_right",
          f"Neutral — SIGNIFICANT |diff| only (FDR p<{alpha}), medial",
          f"plot_sig_diff_neutral{suf}_medial.png", cmap="hot", norm_percentile=99)
    render(sig_map, "left", "right",
          f"Neutral — SIGNIFICANT |diff| only (FDR p<{alpha}), lateral",
          f"plot_sig_diff_neutral{suf}_lateral.png", cmap="hot", norm_percentile=99)

# --- audio only ---
aa = np.load(AO / "bumbo.npy").astype(np.float32)
bb = np.load(AO / "bumboTarol.npy").astype(np.float32)
t_ao, p_ao = ttest_vertices(aa, bb)
diff_ao = np.abs(aa.mean(0) - bb.mean(0))
sig_ao = fdr_mask(p_ao, 0.05)
sig_ao_map = np.where(sig_ao, diff_ao, 0.0)
print(f"\n[audio-only] FDR p<0.05: {int(sig_ao.sum())} sig vertices")
print("   by network:")
for nt in sorted(set(net[sig_ao])):
    print(f"     {nt:12s} {int((net[sig_ao]==nt).sum())}")
render(sig_ao_map, "medial_left", "medial_right",
      "Audio-only — SIGNIFICANT |diff| only (FDR p<0.05), medial",
      "plot_sig_diff_audio_only_medial.png", cmap="hot", norm_percentile=99)

# --- also: top-5% raw threshold for visual clarity (no stats, just biggest) ---
for name, dm in (("neutral", diff_n), ("audio_only", diff_ao)):
    thr = np.percentile(dm, 95)
    top5 = np.where(dm >= thr, dm, 0.0)
    render(top5, "medial_left", "medial_right",
          f"{name} — top 5% vertices by |diff| (medial)",
          f"plot_top5_{name}_medial.png", cmap="hot", norm_percentile=99)
    render(top5, "left", "right",
          f"{name} — top 5% vertices by |diff| (lateral)",
          f"plot_top5_{name}_lateral.png", cmap="hot", norm_percentile=99)

# --- save a small stats summary ---
with open(OUT / "sig_stats.txt", "w") as f:
    f.write("Per-vertex Welch t-test (bumbo TRs vs tarol TRs), FDR-corrected\n\n")
    for label, sig, p in (("neutral p<0.05", fdr_mask(p_n, 0.05), p_n),
                          ("neutral p<0.01", fdr_mask(p_n, 0.01), p_n),
                          ("audio-only p<0.05", sig_ao, p_ao)):
        f.write(f"{label}: {int(sig.sum())} / {N_VERT} sig vertices "
                f"({100*sig.sum()/N_VERT:.2f}%)\n")
        for nt in sorted(set(net[sig])):
            f.write(f"   {nt:12s} {int((net[sig]==nt).sum())}\n")
        f.write("\n")

print("\n[sig] done. See sig_stats.txt and plot_sig_diff_*.png")
