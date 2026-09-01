#!/usr/bin/env python3
"""
Extract the 3D coordinates of every fsaverage5 cortical vertex.

TRIBE v2 predictions live on 20484 vertices (10242 left + 10242 right
hemisphere). This script dumps the (20484, 3) coordinate matrix in
FreeSurfer RAS space (mm) so downstream code can map predictions to
any 3D point (fNIRS optodes, EEG electrodes, MEG sensors, ROI centers)
without needing the mesh files.

Outputs (in output/):
  vertex_coords.npy   (20484, 3) float32, FreeSurfer RAS mm
  vertex_coords.csv   long format: vertex_index, hemisphere, x, y, z
  vertex_coords_info.json

No optodes, no MRI, no participant data required — the fsaverage5 mesh
ships with nilearn and is the exact mesh TRIBE v2 uses.
"""
from __future__ import annotations
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import datasets

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ---- load fsaverage5 mesh (ships with nilearn, ~5MB) -----------------------
fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

# Vertex order MUST match TRIBE's prediction array: left hemisphere first,
# then right. This is the FreeSurfer/nilearn convention and what TRIBE v2
# uses internally (verified: 10242 + 10242 == 20484 == predictions.shape[1]).
coords_blocks = []
hemi_labels = []
for hemi, label in [("left", "L"), ("right", "R")]:
    p = fs[f"white_{hemi}"]
    path = p[0] if isinstance(p, (list, tuple)) else p
    gii = nib.load(path)
    c = gii.darrays[0].data.astype(np.float32)  # (10242, 3) in mm
    coords_blocks.append(c)
    hemi_labels.append(np.full(c.shape[0], label))
    print(f"{label}: {c.shape[0]} vertices  "
          f"x=[{c[:,0].min():.1f},{c[:,0].max():.1f}]  "
          f"y=[{c[:,1].min():.1f},{c[:,1].max():.1f}]  "
          f"z=[{c[:,2].min():.1f},{c[:,2].max():.1f}] mm")

coords = np.vstack(coords_blocks)            # (20484, 3)
hemos  = np.concatenate(hemi_labels)          # (20484,)
assert coords.shape[0] == 20484, coords.shape

# ---- sanity-check against predictions --------------------------------------
pred = np.load(OUT / "predictions.npy")
assert pred.shape[1] == coords.shape[0], \
    f"vertex count mismatch: pred {pred.shape[1]} vs coords {coords.shape[0]}"
print(f"\npredictions {pred.shape}  ==  coords {coords.shape}  ✅ aligned")

# ---- save .npy -------------------------------------------------------------
np.save(OUT / "vertex_coords.npy", coords)

# ---- save .csv (long, human-readable) --------------------------------------
import csv
with open(OUT / "vertex_coords.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["vertex_index", "hemisphere", "x_mm", "y_mm", "z_mm"])
    for i in range(coords.shape[0]):
        w.writerow([i, hemos[i],
                    f"{coords[i,0]:.4f}", f"{coords[i,1]:.4f}", f"{coords[i,2]:.4f}"])

# ---- info json -------------------------------------------------------------
info = {
    "mesh": "fsaverage5",
    "n_vertices": int(coords.shape[0]),
    "n_left": 10242,
    "n_right": 10242,
    "vertex_order": "left hemisphere [0:10242], then right [10242:20484]",
    "coordinate_system": "FreeSurfer RAS (x=L→R, y=P→A, z=I→S), millimeters",
    "source": "nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage5')",
    "surfaces_used": "white_left.gii.gz, white_right.gii.gz (white-matter surface)",
    "files": ["vertex_coords.npy", "vertex_coords.csv"],
    "matches_predictions": f"predictions.npy shape[1] == {coords.shape[0]} ✅",
}
with open(OUT / "vertex_coords_info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"\n✅ Saved:")
print(f"   {OUT/'vertex_coords.npy'}   {coords.shape} float32, ~{coords.nbytes//1024} KB")
print(f"   {OUT/'vertex_coords.csv'}   20485 rows (header + 20484 vertices)")
print(f"   {OUT/'vertex_coords_info.json'}")
