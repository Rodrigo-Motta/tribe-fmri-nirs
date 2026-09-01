#!/usr/bin/env python3
"""
Build predictions_with_coords.csv — vertex-centric wide format.

Each row = one fsaverage5 vertex, with its 3D coordinate columns
followed by its full 436-point predicted timecourse.

Layout (20484 rows × 441 columns):
    vertex_index, hemisphere, x_mm, y_mm, z_mm, t0, t1, ..., t435

This is the natural unit for spatial analysis: pick a vertex, you
immediately know where it is AND its predicted fMRI timecourse.

Why vertex-rows (not timestep-rows like predictions.csv)?
  - Coords are per-vertex (20484), not per-timestep (436), so they
    align naturally as columns on a vertex row.
  - Long/tidy format (8.9M rows) would be ~450 MB — exceeds GitHub's
    100 MB limit. This vertex-wide format is ~75 MB and fits.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
from nilearn import datasets
import nibabel as nib

OUT = Path(__file__).parent / "output"
TR_S = 1.0          # seconds per timestep (from run_info.json)
N_VERTICES = 20484

# ---- load predictions (436, 20484) -----------------------------------------
pred = np.load(OUT / "predictions.npy")        # (436, 20484) float32
n_t, n_v = pred.shape
assert n_v == N_VERTICES, f"expected {N_VERTICES} vertices, got {n_v}"
print(f"predictions: {pred.shape}  range [{pred.min():.3f}, {pred.max():.3f}]")

# ---- load vertex coords (20484, 3) — recompute from mesh (don't depend on
#      extract_vertex_coords.py having been run) -----------------------------
fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
coords_blocks, hemi_labels = [], []
for hemi, label in [("left", "L"), ("right", "R")]:
    p = fs[f"white_{hemi}"]
    path = p[0] if isinstance(p, (list, tuple)) else p
    c = nib.load(path).darrays[0].data.astype(np.float32)
    coords_blocks.append(c)
    hemi_labels.append(np.full(c.shape[0], label))
coords = np.vstack(coords_blocks)              # (20484, 3)
hemos  = np.concatenate(hemi_labels)            # (20484,)
assert coords.shape[0] == N_VERTICES

# ---- write CSV: vertex rows, coord cols + timecourse cols ------------------
out_csv = OUT / "predictions_with_coords.csv"
header = ["vertex_index", "hemisphere", "x_mm", "y_mm", "z_mm"] + \
         [f"t{t}" for t in range(n_t)]

print(f"writing {out_csv}  ({n_v} rows × {len(header)} cols)...")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for v in range(n_v):
        row = [v, hemos[v],
               f"{coords[v,0]:.4f}", f"{coords[v,1]:.4f}", f"{coords[v,2]:.4f}"]
        row.extend(f"{pred[t, v]:.6f}" for t in range(n_t))
        w.writerow(row)

size_mb = out_csv.stat().st_size / 1048576
print(f"\n✅ {out_csv.name}  {n_v} rows × {len(header)} cols  {size_mb:.1f} MB")
print(f"   columns: vertex_index, hemisphere, x_mm, y_mm, z_mm, t0..t{n_t-1}")
print(f"   each row = one vertex's location + full 436-point predicted timecourse")
