# TRIBE v2 — Brain-Encoding Predictions for fMRI / fNIRS

Predict fMRI brain responses from video + audio stimuli using **Meta's TRIBE v2** multimodal brain-encoding model, then bridge the predictions to fNIRS optode coordinates on the fsaverage5 cortical mesh.

TRIBE v2 encodes naturalistic stimuli (video, audio, text) through pretrained backbones — **V-JEPA2 ViT-G** (video), **Wav2Vec-BERT 2.0** (audio), **LLaMA-3.2-3B** (text), **DINOv2-large** (image) — and decodes them into vertex-wise predicted BOLD activity on the **fsaverage5** cortical surface (20,484 vertices).

---

## Repository contents

```
tribe-fmri-nirs/
├── README.md                          ← this file
├── TRIBE_V2_PROCESS.md                ← detailed setup + 16GB-RAM workaround docs
├── .env.example                       ← HF token placeholder (copy → .env)
├── pyproject.toml / uv.lock           ← dependency manifest (tribev2[plotting])
│
├── run_tribe.py                       ← Exp 1: main video inference
├── extract_embeddings.py              ← dump cached V-JEPA2 / Wav2Vec-BERT features
├── extract_vertex_coords.py           ← extract fsaverage5 vertex 3D coordinates
├── build_predictions_with_coords.py   ← merge predictions + coords into one CSV
├── plot_std_surface.py                ← plot per-vertex std on cortical surface
├── main.py                            ← minimal entry point
│
├── stimulus/
│   └── video_with_audio.mp4           ← Exp 1 stimulus (435.7 s, H.264+AAC, no speech)
├── stimulus_2/
│   ├── bumbo.aifc                     ← Exp 2 audio stimulus A
│   └── bumboTarol.aifc                ← Exp 2 audio stimulus B
├── std_brain/
│   ├── standard_deviations.csv        ← per-vertex std across predictions
│   └── std_surface.png                ← std rendered on inflated cortex
│
├── output/                            ← Exp 1 outputs (predictions + embeddings)
│   ├── predictions.npy                ← (436, 20484) fMRI preds, fsaverage5
│   ├── predictions.csv                ← same, wide timestep×vertex format
│   ├── predictions_with_coords.csv    ← vertex rows + xyz coords + timecourse
│   ├── vertex_coords.npy              ← (20484, 3) vertex xyz in FreeSurfer RAS mm
│   ├── vertex_coords.csv              ← vertex_index, hemisphere, x, y, z
│   ├── brain_preview.png              ← 12-TR cortical surface preview
│   ├── segments.csv / events.csv      ← per-TR segment + event metadata
│   ├── run_info.json / run.log        ← run metadata + full inference log
│   └── embeddings/
│       ├── video_features.npy         ← (1000, 2, 1408) V-JEPA2 features
│       ├── audio_features.npy         ← (1000, 2, 1024) Wav2Vec-BERT features
│       └── *.csv                      ← same in tidy long format
│
└── exp2/                              ← Exp 2: bumbo vs bumboTarol comparison
    ├── run_tribe_audio.py             ← audio-only inference
    ├── run_tribe_neutral.py           ← neutral-grey-video + audio inference
    ├── compare_brains.py              ← vertex-wise difference maps
    ├── compare_neutral.py             ← statistical mPFC hypothesis test
    ├── render_*.py                    ← cortical surface plotting scripts
    ├── stimulus/                      ← neutral-grey mp4 stimuli
    └── output/                        ← predictions, plots, significance maps
        ├── bumbo/predictions.npy      ← (18, 20484)
        ├── bumboTarol/predictions.npy ← (20, 20484)
        ├── _audio_only/*.npy          ← audio-only backup predictions
        ├── difference_map.npy         ← |mean_bumbo - mean_tarol| per vertex
        ├── *.png                      ← surface maps (mean, diff, significance)
        ├── schaefer_roi_table*.csv    ← Schaefer-400 ROI summary
        └── sig_stats.txt              ← FDR-corrected sig-vertex counts
```

---

## Quick start

### 1. Prerequisites

- **Python 3.11+** (3.12 tested)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **HuggingFace account** with access to:
  - `facebook/tribev2` (gated model)
  - `meta-llama/Llama-3.2-3B` (gated, used as TRIBE's text backbone)

### 2. Clone & install

```bash
git clone https://github.com/Rodrigo-Motta/tribe-fmri-nirs.git
cd tribe-fmri-nirs

# create venv + install all deps (tribev2[plotting], torch, nilearn, mne, ...)
uv sync
```

### 3. Add your HuggingFace token

```bash
cp .env.example .env
# edit .env and paste your HF token:
#   HF_KEY=hf_your_token_here
```

> **Never commit `.env`.** It is gitignored. The token is loaded at runtime by
> `run_tribe.py` and the exp2 scripts.

### 4. Run inference

#### Experiment 1 — main video stimulus

```bash
# Auto-selects device: MPS on Apple Silicon, CUDA if available, else CPU
uv run python run_tribe.py
```

#### Experiment 2 — bumbo vs bumboTarol comparison

```bash
# Generate neutral-grey-video + audio predictions
uv run python exp2/run_tribe_neutral.py

# Compare + plot
uv run python exp2/compare_neutral.py
uv run python exp2/render_plots.py
uv run python exp2/render_significance.py
```

---

## Device selection (CPU / GPU / MPS)

All inference scripts auto-detect the best available device and adapt
automatically. You can override with the `TRIBE_DEVICE` environment variable.

| Platform | Command | Notes |
|---|---|---|
| **Apple Silicon (M1–M4)** | `TRIBE_DEVICE=mps uv run python run_tribe.py` | Default if MPS available. Uses **fp16** to fit 16 GB RAM. ~82 min for 436 s video. |
| **NVIDIA GPU** | `TRIBE_DEVICE=cuda uv run python run_tribe.py` | Fastest. Uses fp16 natively. ~10–15 min for 436 s video. |
| **CPU only** | `TRIBE_DEVICE=cpu uv run python run_tribe.py` | Slowest but always works. Uses fp32. ~3–5 h for 436 s video. |

```bash
# explicit override examples:
TRIBE_DEVICE=cuda uv run python run_tribe.py
TRIBE_DEVICE=cpu  uv run python run_tribe.py
TRIBE_DEVICE=mps  uv run python run_tribe.py
```

### 16 GB RAM workarounds (Apple Silicon)

The scripts apply four patches automatically when MPS is detected — see
**[TRIBE_V2_PROCESS.md](TRIBE_V2_PROCESS.md)** for the full rationale:

1. **fp16 on MPS** — monkeypatches `transformers.AutoModel.from_pretrained` to
   inject `torch_dtype=float16`; remaps `cpu → mps` in `HuggingFaceMixin`.
2. **16-frame clips** — `num_frames=16` (instead of 64) in V-JEPA2 → 19× faster,
   same 1408-dim feature space.
3. **`keep_in_ram=False`** — frees each feature model after extraction.
4. **Skip WhisperX** — `get_audio_and_text_events(..., audio_only=True)` skips
   transcription (the stimulus has no speech).

---

## Output formats

### Predictions

| File | Shape | Format |
|---|---|---|
| `output/predictions.npy` | `(436, 20484)` | float32, one row per timestep (1 TR = 1 s) |
| `output/predictions.csv` | `436 × 20486` | wide: `timestep, start_s, vertex_0 … vertex_20483` |
| `output/predictions_with_coords.csv` | `20484 × 441` | vertex rows: `vertex_index, hemisphere, x_mm, y_mm, z_mm, t0 … t435` |

- **20484 vertices** = 10242 left + 10242 right hemisphere (fsaverage5)
- Predictions are **offset 5 s in the past** to compensate for the hemodynamic lag
- Value range: approximately `[-1.44, 1.11]` (normalized predicted BOLD)

### Vertex coordinates

| File | Shape | Description |
|---|---|---|
| `output/vertex_coords.npy` | `(20484, 3)` | xyz in FreeSurfer RAS space (mm) |
| `output/vertex_coords.csv` | 20484 rows | `vertex_index, hemisphere, x_mm, y_mm, z_mm` |

**Coordinate system:** FreeSurfer RAS — `x: left→right`, `y: posterior→anterior`,
`z: inferior→superior`, in millimeters. Vertex ordering matches `predictions.npy`
exactly: left hemisphere `[0:10242]`, then right `[10242:20484]`.

### Feature embeddings

| File | Shape | Backbone |
|---|---|---|
| `output/embeddings/video_features.npy` | `(1000, 2, 1408)` | V-JEPA2 ViT-G (2 Hz, 2 layers) |
| `output/embeddings/audio_features.npy` | `(1000, 2, 1024)` | Wav2Vec-BERT 2.0 (2 Hz, 2 layers) |

---

## Bridging predictions to fNIRS optodes

The vertex coordinates (`output/vertex_coords.npy`) are the bridge between
TRIBE's cortical predictions and fNIRS optode locations:

1. **Obtain optode 3D coordinates** (digitized or 10-10 system), coregistered
   to FreeSurfer RAS / MNI space.
2. **Build a sensitivity matrix `J`** (channels × vertices) — each channel
   weights nearby vertices by a Gaussian kernel (`σ ≈ 15 mm`) or, for gold-
   standard accuracy, a Monte Carlo photon-migration model (AtlasViewer / MCX).
3. **Project vertices → channels:**

   ```python
   # predictions: (436, 20484)   J: (n_channels, 20484)
   channel_predictions = predictions @ J.T   # → (436, n_channels)
   ```

This sums vertex contributions *into* each fNIRS channel (vertex → optode),
correctly modeling fNIRS spatial spread and overlapping sensitivity volumes.

---

## Experiment 2 — mPFC hypothesis

Exp2 tests whether two percussion clips (`bumbo` vs `bumboTarol`) produce
different predicted brain activity, specifically in the **medial prefrontal
cortex (mPFC)**.

- **Stimuli:** short audio clips (~15–19 s) paired with constant grey video
  (neutral visual input) so differences are audio-driven.
- **Analysis:** per-vertex Welch t-test (bumbo TRs vs bumboTarol TRs) with
  FDR correction, summarized by Schaefer-400 network.
- **Result:** 82/20484 vertices significant at FDR p<0.05 (neutral condition),
  concentrated in the Default and Control networks — consistent with mPFC
  involvement. See `exp2/output/sig_stats.txt`.

---

## Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `tribev2[plotting]` | git (facebook/tribev2) | brain-encoding model + cortical plotting |
| `torch` | 2.6.0 | tensor computation (CPU/CUDA/MPS) |
| `transformers` | — | HF model loading (LLaMA, Wav2Vec-BERT) |
| `nilearn` | 0.14.0 | fsaverage5 mesh + surface plotting |
| `nibabel` | 5.4.2 | FreeSurfer surface I/O |
| `mne` | 1.12.1 | fNIRS forward modeling (future) |
| `pandas` / `numpy` | — | data handling |

Full pinned versions in `uv.lock`.

---

## Reproducing from scratch

```bash
# 1. Environment
uv sync

# 2. HF token
cp .env.example .env  # add your HF_KEY

# 3. Exp 1: main video
uv run python run_tribe.py                    # → output/predictions.npy
uv run python extract_embeddings.py           # → output/embeddings/*.npy
uv run python extract_vertex_coords.py        # → output/vertex_coords.npy
uv run python build_predictions_with_coords.py # → output/predictions_with_coords.csv

# 4. Exp 2: bumbo vs bumboTarol
uv run python exp2/run_tribe_neutral.py       # → exp2/output/*/predictions.npy
uv run python exp2/compare_neutral.py         # → exp2/output/difference_map.npy
uv run python exp2/render_significance.py     # → exp2/output/plot_sig_diff_*.png
uv run python exp2/render_plots.py            # → exp2/output/plot_*.png
uv run python exp2/render_timecourse.py       # → exp2/output/plot_diff_timecourse_*.png
```

---

## License & data

- **TRIBE v2 model:** © Meta AI, gated on HuggingFace — you must accept the
  model license before use.
- **Code in this repo:** MIT (see `LICENSE` if present, otherwise treat as MIT).
- **Stimulus video/audio:** included for reproducibility.

## Citation

If you use TRIBE v2, cite the original model:

```bibtex
@article{tribe2025,
  title={TRIBE v2: A Multimodal Brain Encoding Model},
  author={Lescroart, Mark and others},
  year={2025},
  publisher={Meta AI},
  url={https://huggingface.co/facebook/tribev2}
}
```
