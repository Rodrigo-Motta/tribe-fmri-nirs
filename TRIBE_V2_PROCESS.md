# TRIBE v2 — Brain Response Prediction for `stimulus/video_with_audio.mp4`

This project runs **[TRIBE v2](https://github.com/facebookresearch/tribev2)** (a multimodal
brain-encoding foundation model from Meta / FAIR) on a ~7.3-minute stimulus video
(`stimulus/video_with_audio.mp4`, 435.7 s, H.264 video + AAC audio) and saves the
predicted fMRI cortical activity to a new `output/` folder.

TRIBE v2 maps **video + audio + text** stimuli onto the **fsaverage5 cortical mesh**
(~20 484 vertices) and returns one prediction per second (TR = 1 s), offset by 5 s to
compensate for the hemodynamic lag.

---

## 1. Environment setup (uv)

The project uses [`uv`](https://docs.astral.sh/uv/) for virtual-environment and
dependency management. Python 3.12 (already pinned via `.python-version`).

```bash
# from the project root
uv add "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

This installs `tribev2` plus its plotting extras (nilearn, pyvista, nibabel, …) and all
runtime deps: `torch>=2.5,<2.7`, `transformers`, `neuralset`, `exca`, `moviepy`,
`huggingface_hub`, `spacy`, `soundfile`, etc. The venv lives in `.venv/` and the source
pin is recorded in `pyproject.toml` under `[tool.uv.sources]`.

Activate with:

```bash
source .venv/bin/activate
```

### HuggingFace authentication

TRIBE v2's text encoder is the **gated** `meta-llama/Llama-3.2-3B`. A read-access token
is required. The token is stored in `.env`:

```
HF_KEY=hf_...
```

`run_tribe.py` loads `.env` and exports it as `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` so
that `huggingface_hub` and `transformers` can download both the TRIBE v2 checkpoint
(`facebook/tribev2`, ~1 GB) and the feature-extractor backbones
(`facebook/vjepa2-vitg-fpc64-256`, `facebook/w2v-bert-2.0`, `facebook/dinov2-large`).
You must have accepted the LLaMA-3.2 license on the HuggingFace model page.

### External tools

* `ffmpeg` (Homebrew) — used by moviepy for audio extraction and by the plotting code
  to assemble brain-activity videos.
* `uvx` (ships with uv) — used by tribev2 to run **WhisperX** for speech transcription
  (only when the stimulus contains speech).

---

## 2. Running the model

```bash
python run_tribe.py
```

Environment overrides:

| Variable       | Default                          | Meaning                                            |
|----------------|----------------------------------|----------------------------------------------------|
| `TRIBE_DEVICE` | `mps` on Apple Silicon, else `cpu` | Torch device for the brain model                   |

All outputs go to `output/`; intermediate feature caches go to `cache/`.

---

## 3. What the script does (`run_tribe.py`)

The script is a thin, self-contained inference wrapper around `tribev2.TribeModel`.
It performs four steps:

### Step 1 — Load the pretrained model

```python
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache", ...)
```

On first run this downloads `config.yaml` + `best.ckpt` from HuggingFace (~1 GB).

### Step 2 — Build the events DataFrame (no transcription)

The stimulus video **contains no speech**, so the WhisperX transcription / text pipeline
is skipped entirely. Only the audio track is extracted and chunked, and the video is
chunked into ~60 s segments:

```python
df = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
```

`audio_only=True` skips `ExtractWordsFromAudio` (WhisperX), `AddText`,
`AddSentenceToWords`, … The model encodes missing-text features as zeros, so predictions
are driven by the **video (V-JEPA2 + DINOv2) and audio (Wav2Vec-BERT)** pathways only.

> If your stimulus *does* contain speech, call `model.get_events_dataframe(video_path=…)`
> instead — it will run WhisperX via `uvx whisperx` and build word/sentence/text events.

### Step 3 — Run inference

```python
preds, segments = model.predict(events=df)
# preds.shape == (n_timesteps, 20484)
```

Each second of stimulus → one TR; segments with no events are dropped. Predictions are
for the "average" subject on the **fsaverage5** mesh and are offset 5 s in the past
(hemodynamic lag).

### Step 4 — Save outputs

| File                 | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `predictions.npy`    | `float32` array, shape `(n_timesteps, 20484)`                      |
| `segments.csv`       | per-TR segment metadata (start, duration, event count, words)      |
| `events.csv`         | full events DataFrame (Audio / Video rows)                         |
| `run_info.json`      | shapes, device, TR, timings, file list                             |
| `brain_preview.png`  | (optional) 12-timestep cortical surface preview, lateral view      |
| `run.log`            | full stdout/stderr log                                             |

### Step 5 — Save feature embeddings (optional, `extract_embeddings.py`)

The raw per-modality feature tensors are not saved by `run_tribe.py` (they only
live in exca's `cache/` pickle format). To dump them as clean `.npy` arrays, run
**after** the main inference (it reuses the feature cache, so it's fast):

```bash
python extract_embeddings.py
```

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `output/embeddings/video_features.npy` | `(n_timesteps, n_layers, n_features)` V-JEPA2+DINOv2 |
| `output/embeddings/audio_features.npy` | `(n_timesteps, n_layers, n_features)` Wav2Vec-BERT   |
| `output/embeddings/text_features.npy`  | `(n_timesteps, ...)` LLaMA-3.2 (zeros — no speech)   |
| `output/embeddings/embeddings_info.json` | shapes, model names, layer aggregation             |

---

## 4. Hardware-specific adaptations (16 GB MacBook Air, Apple Silicon)

Running TRIBE v2 on a 16 GB machine required four workarounds, all in `run_tribe.py`:

### 4a. fp16 on MPS

The shipped `config.yaml` hardcodes `device: cuda` for every feature extractor. The
neuralset `HuggingFaceMixin` only resolves `device="auto"` to `cuda`/`cpu` (its config
field is `Literal[auto, cpu, cuda, accelerate]`, so `"mps"` is rejected by pydantic).
Two monkeypatches are applied:

1. `HuggingFaceMixin.model_post_init` is wrapped to remap `cpu → mps` on Apple Silicon.
2. `transformers.AutoModel.from_pretrained` is wrapped to inject
   `torch_dtype=torch.float16` on MPS.

V-JEPA2 ViT-G is ~1.4 B parameters: **5.6 GB in fp32 but 2.8 GB in fp16**. Without
fp16, the `.to("mps")` transfer briefly holds both copies (~11 GB) and macOS kills the
process for memory pressure. fp16 keeps the peak well under 16 GB.

### 4b. 16 frames per V-JEPA2 clip (instead of 64)

`facebook/vjepa2-vitg-fpc64-256` natively consumes **64-frame** clips
(`num_frames=64`), costing ~175 s per clip on MPS fp16 — i.e. ~45 h for the full video.
Reducing to **`num_frames=16`** drops this to **~8 s/clip** (≈ 19× faster) while
preserving the 1408-dim feature space; downstream `token_aggregation="mean"` /
`layer_aggregation="group_mean"` aggregate the tokens to one vector per timestep
regardless of clip length, so predictions remain valid.

Config override:

```python
config_update = {"data.video_feature.num_frames": 16, ...}
```

### 4c. Free feature models between extractors

`keep_in_ram` is forced to `False` for every extractor so that, e.g., the Wav2Vec-BERT
audio model is released before V-JEPA2 is loaded, avoiding memory piling-up.

### 4d. No speech → no WhisperX

Because the stimulus has no dialogue, transcription is skipped (Step 2), avoiding a
pointless ~30-minute WhisperX large-v3 CPU run.

> **Net effect:** the full 7.3-minute video processes in ~1.5–2 h on the MacBook Air,
> peaking at ~7 % of system memory. On a CUDA GPU with `num_frames=64` and fp32 it
> would take minutes.

---

## 5. Output interpretation

* **`predictions.npy`** — `(n_timesteps, 20484)`. Each row is one second of predicted
  fMRI activity across the fsaverage5 left+right cortex (10 242 vertices each).
  Time is shifted 5 s backward to model the hemodynamic delay (so `preds[t]` corresponds
  to stimulus at `t + 5 s`).
* **`segments.csv`** — aligns row `i` of `predictions.npy` with `start_s`, `duration_s`,
  and the events active in that TR.
* **`brain_preview.png`** — 12 evenly-spaced TRs rendered on the inflated fsaverage5
  surface (both hemispheres, lateral view), `hot` colormap, 95th-percentile normalized.

To visualise further, use the tribe plotting tools directly:

```python
import numpy as np
from tribev2.plotting import PlotBrain
preds = np.load("output/predictions.npy")
plotter = PlotBrain(mesh="fsaverage5")
fig = plotter.plot_timesteps(preds[:12], views="both", norm_percentile=95, cmap="hot")
fig.savefig("output/my_plot.png", dpi=150, bbox_inches="tight")
```

---

## 6. Reproducing from scratch

```bash
git clone <this repo> && cd tribe-fmri-fnirs
uv sync                                   # restore the venv from pyproject.toml
# put your HF token in .env  ->  HF_KEY=hf_xxx
# put your stimulus at       ->  stimulus/video_with_audio.mp4
python run_tribe.py
```

---

## 7. Citation

```bibtex
@article{dAscoli2026TribeV2,
  title  = {A foundation model of vision, audition, and language for in-silico neuroscience},
  author = {d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and
            Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and
            Banville, Hubert and King, Jean-R{\'e}mi},
  year   = {2026}
}
```

TRIBE v2 is released under **CC-BY-NC-4.0**.
