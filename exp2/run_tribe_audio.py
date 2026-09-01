"""Run TRIBE v2 on each audio stimulus in `stimulus_2/` and save predictions.

This is the experiment-2 companion to the project's `run_tribe.py`. It is
self-contained (does NOT modify the original scripts) and runs TRIBE v2 on the
two short percussion audio files:

    stimulus_2/bumbo.aifc        (~15.3 s)
    stimulus_2/bumboTarol.aifc   (~17.0 s)

Predictions are saved per-stimulus under `exp2/output/<stem>/`:

    predictions.npy   : (n_timesteps, 20484) predicted fsaverage5 fMRI activity
    segments.csv      : per-TR segment metadata
    events.csv        : events DataFrame
    run_info.json     : run metadata

Because the clips contain no speech, the WhisperX text pipeline is skipped
(`audio_only=True`) — only the audio (Wav2Vec-BERT) pathway drives predictions.

Usage:
    python exp2/run_tribe_audio.py            # run both audio files
    python exp2/run_tribe_audio.py bumbo.aifc  # run a single file
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- load .env (HF_KEY) into the environment ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

hf_token = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from tribev2.demo_utils import TribeModel, get_audio_and_text_events  # noqa: E402

# --- monkeypatch: MPS + fp16 on Apple Silicon (16GB RAM needs fp16) ----------
import neuralset.extractors.base as _ns_base  # noqa: E402

_USE_MPS = torch.backends.mps.is_available()
_DTYPE = torch.float16 if _USE_MPS else torch.float32

_orig_hf_post_init = _ns_base.HuggingFaceMixin.model_post_init


def _hf_post_init(self, log__: tp.Any) -> None:
    _orig_hf_post_init(self, log__)
    if _USE_MPS and self.device == "cpu":
        self.device = "mps"


_ns_base.HuggingFaceMixin.model_post_init = _hf_post_init

import transformers  # noqa: E402

_orig_auto_from_pretrained = transformers.AutoModel.from_pretrained.__func__


def _auto_from_pretrained_patched(*args, **kwargs):
    if _USE_MPS and "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = _DTYPE
    return _orig_auto_from_pretrained(*args, **kwargs)


transformers.AutoModel.from_pretrained = classmethod(_auto_from_pretrained_patched)
print(f"[exp2] Patched AutoModel.from_pretrained: torch_dtype={_DTYPE} on MPS={_USE_MPS}",
      flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exp2")


def build_segments_metadata(segments: list, tr: float) -> pd.DataFrame:
    rows = []
    for i, seg in enumerate(segments):
        start = float(getattr(seg, "start", float("nan")))
        duration = float(getattr(seg, "duration", float("nan")))
        ns_events = getattr(seg, "ns_events", None)
        n_events = len(ns_events) if ns_events is not None else None
        rows.append({
            "index": i,
            "start_s": start,
            "duration_s": duration,
            "n_events": n_events,
        })
    return pd.DataFrame(rows)


def run_one(model: TribeModel, audio_path: Path, out_dir: Path) -> Path:
    """Run TRIBE inference on a single audio file and save predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Running TRIBE on %s ===", audio_path.name)

    # Build an Audio event directly (no Video, so ExtractAudioFromVideo is a
    # no-op; the file is <60 s so ChunkEvents leaves it intact).
    import soundfile as sf
    info = sf.info(str(audio_path))
    event = {
        "type": "Audio",
        "filepath": str(audio_path),
        "start": 0,
        "duration": float(info.duration),
        "frequency": float(info.samplerate),
        "timeline": "default",
        "subject": "default",
    }
    df = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    log.info("Events built (%d rows)", len(df))
    df.to_csv(out_dir / "events.csv", index=False)

    t_pred = time.time()
    preds, segments = model.predict(events=df, verbose=True)
    log.info("Inference done in %.1fs — predictions %s",
             time.time() - t_pred, preds.shape)

    np.save(out_dir / "predictions.npy", preds.astype(np.float32))

    tr = float(getattr(model.data, "TR", 1.0))
    seg_df = build_segments_metadata(segments, tr)
    seg_df.to_csv(out_dir / "segments.csv", index=False)

    run_info = {
        "stimulus": str(audio_path),
        "stimulus_duration_s": float(info.duration),
        "model": "facebook/tribev2",
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "TR_s": tr,
        "hemodynamic_lag_s": 5,
        "mesh": "fsaverage5",
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log.info("Saved outputs to %s", out_dir)
    return out_dir / "predictions.npy"


def main() -> int:
    t0 = time.time()
    audio_dir = PROJECT_ROOT / "stimulus_2"
    out_root = PROJECT_ROOT / "exp2" / "output"
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # pick files
    if len(sys.argv) > 1:
        targets = [audio_dir / name for name in sys.argv[1:]]
    else:
        targets = sorted([p for p in audio_dir.iterdir() if p.suffix.lower() == ".aifc"])
    for p in targets:
        if not p.is_file():
            log.error("Audio file not found: %s", p)
            return 2
    log.info("Audio targets: %s", [p.name for p in targets])

    device = os.environ.get("TRIBE_DEVICE", "mps" if _USE_MPS else "cpu")
    log.info("Device: %s (MPS=%s, dtype=%s)", device, _USE_MPS, _DTYPE)

    # --- load model once, reuse for both stimuli ----------------------------
    t_load = time.time()
    config_update = {
        "data.text_feature.device": "auto",
        "data.image_feature.image.device": "auto",
        "data.image_feature.infra.keep_in_ram": False,
        "data.image_feature.image.infra.keep_in_ram": False,
        "data.audio_feature.device": "auto",
        "data.audio_feature.infra.keep_in_ram": False,
        "data.video_feature.image.device": "auto",
        "data.video_feature.infra.keep_in_ram": False,
        "data.video_feature.image.infra.keep_in_ram": False,
        "data.video_feature.num_frames": 16,
        "data.video_feature.frequency": 1.0,
        "data.image_feature.frequency": 1.0,
    }
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_dir,
        device=device,
        config_update=config_update,
    )
    log.info("Model loaded in %.1fs", time.time() - t_load)

    for audio_path in targets:
        stem = audio_path.stem
        out_dir = out_root / stem
        run_one(model, audio_path, out_dir)

    log.info("All done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
