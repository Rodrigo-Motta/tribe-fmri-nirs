"""Extract and save per-modality feature embeddings for the stimulus video.

Reuses the feature cache built by run_tribe.py, so this is fast (no model
inference — just loading cached V-JEPA2 / DINOv2 / Wav2Vec-BERT features and
stacking them into clean .npy arrays).

Outputs (output/embeddings/):
    video_features.npy   : (n_timesteps, n_layers, n_features)  V-JEPA2 + DINOv2
    audio_features.npy   : (n_timesteps, n_layers, n_features)  Wav2Vec-BERT
    text_features.npy    : (n_timesteps, ...)                   LLaMA-3.2 (zeros, no speech)
    embeddings_info.json : shapes, model names, layer aggregation

Usage:
    python extract_embeddings.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- load .env ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
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
import typing as tp  # noqa: E402
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
_orig_auto = transformers.AutoModel.from_pretrained.__func__


def _auto_patched(*args, **kwargs):
    if _USE_MPS and "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = _DTYPE
    return _orig_auto(*args, **kwargs)


transformers.AutoModel.from_pretrained = classmethod(_auto_patched)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_embeddings")


def main() -> int:
    t0 = time.time()
    video_path = PROJECT_ROOT / "stimulus" / "video_with_audio.mp4"
    out_dir = PROJECT_ROOT / "output" / "embeddings"
    cache_dir = PROJECT_ROOT / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = os.environ.get("TRIBE_DEVICE", "mps" if _USE_MPS else "cpu")
    # MUST match run_tribe.py config so cache keys align
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
    log.info("Loading model (features will load from cache)...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=cache_dir, device=device,
        config_update=config_update,
    )

    event = {"type": "Video", "filepath": str(video_path), "start": 0,
             "timeline": "default", "subject": "default"}
    df = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)

    # Force single-process dataloader (workers crash with MPS + pickling)
    model.data.num_workers = 0
    model.data.batch_size = 1
    loaders = model.data.get_loaders(events=df, split_to_build="all")
    loader = loaders["all"]
    log.info("Loader built. Iterating batches to stack cached features...")

    feats: dict[str, list[np.ndarray]] = {}
    n_batches = 0
    with torch.inference_mode():
        for batch in loader:
            n_batches += 1
            for modality in model.data.features_to_use:
                if modality not in batch.data:
                    continue
                tensor = batch.data[modality]  # (B, L, D, T) or (B, D, T)
                arr = tensor.detach().cpu().numpy().astype(np.float32)
                feats.setdefault(modality, []).append(arr)

    info = {"batches": n_batches, "modalities": {}, "device": device}
    for modality, chunks in feats.items():
        # Each chunk is (B, L, D, T) per the model's aggregate_features; but be
        # robust to (B, D, T) [ndim==3] layouts too. Concatenate along time and
        # flatten the batch dim into the time dimension (segments are sequential).
        stacked = np.concatenate(chunks, axis=-1)  # concat time dim
        if stacked.ndim == 4:        # (B, L, D, T)
            B, L, D, Ttot = stacked.shape
            flat = stacked.transpose(0, 3, 1, 2).reshape(Ttot, L, D)  # (T, L, D)
        elif stacked.ndim == 3:      # (B, D, T)
            B, D, Ttot = stacked.shape
            flat = stacked.transpose(0, 2, 1).reshape(Ttot, D)       # (T, D)
        else:
            flat = stacked.reshape(stacked.shape[-1], -1)
        np.save(out_dir / f"{modality}_features.npy", flat)
        info["modalities"][modality] = {
            "shape": list(flat.shape),
            "raw_shape": list(stacked.shape),
        }
        log.info("%s features: %s -> saved", modality, flat.shape)

    # capture model/extractor metadata
    try:
        info["extractors"] = {}
        for modality in model.data.features_to_use:
            ext = getattr(model.data, f"{modality}_feature", None)
            if ext is not None:
                info["extractors"][modality] = {
                    "name": getattr(ext, "name", "?"),
                    "frequency": getattr(ext, "frequency", None),
                    "layer_aggregation": getattr(ext, "layer_aggregation", None),
                    "image_model": getattr(getattr(ext, "image", None), "model_name", None),
                }
    except Exception as e:
        log.warning("extractor metadata capture failed: %r", e)

    info["timings_s"] = round(time.time() - t0, 1)
    (out_dir / "embeddings_info.json").write_text(json.dumps(info, indent=2))
    log.info("Done in %.1fs. Outputs in %s", time.time() - t0, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
