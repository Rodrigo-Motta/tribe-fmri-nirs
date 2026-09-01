"""Plot per-vertex standard deviation on the fsaverage5 cortical surface.

Reads std_brain/standard_deviations.csv (one row per vertex, vertex_0..vertex_20483)
and renders it on the inflated fsaverage5 mesh using tribev2's PlotBrain.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tribev2.plotting import PlotBrain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "std_brain" / "standard_deviations.csv"
OUT_PATH = PROJECT_ROOT / "std_brain" / "std_surface.png"


def load_vertex_std(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    # keep only vertex_* rows (drop timestep / start_s metadata rows)
    vert_rows = df.index.astype(str).str.startswith("vertex_")
    vals = df.loc[vert_rows, df.columns[0]].to_numpy(dtype=float)
    return vals


def main() -> None:
    data = load_vertex_std(CSV_PATH)
    print(f"Loaded {data.size} vertices; std range "
          f"[{data.min():.4f}, {data.max():.4f}], mean {data.mean():.4f}")

    plotter = PlotBrain(mesh="fsaverage5")

    views = ["left", "right", "ventral", "dorsal"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()

    plotter.plot_surf(
        data,
        axes=axes_flat.tolist(),
        views=views,
        cmap="hot",
        norm_percentile=95,
    )

    for ax, view in zip(axes_flat, views):
        ax.set_title(f"{view} view", fontsize=13)

    fig.suptitle("Per-vertex standard deviation (fsaverage5, 95th-pct normalized)",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
