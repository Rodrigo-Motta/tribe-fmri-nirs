"""Run TRIBE v2 on the neutral (grey-video + percussion-audio) stimuli.

This is the in-distribution version of exp2. Each audio clip in `stimulus_2/`
is paired with a constant grey video (`exp2/stimulus/<stem>_neutral.mp4`) so
the video pathway (V-JEPA2 + DINOv2) receives a real, *constant* input. Any
difference in predicted brain activity between the two clips is therefore
driven by the **audio** (Wav2Vec-BERT), not by a zeroed-video artifact.

Outputs overwrite `exp2/output/<stem>/predictions.npy` (the earlier
audio-only predictions were out-of-distribution and are superseded here).

Usage:
    python exp2/run_tribe_neutral.py
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

# --- load .env (HF_KEY) ------------------------------------------------------
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

# --- monkeypatch: MPS + fp16 on Apple Silicon --------------------------------
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
print(f"[exp2-neutral] Patched AutoModel.from_pretrained: "
      f"torch_dtype={_DTYPE} on MPS={_USE_MPS}", flush=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("exp2-neutral")


def build_segments_metadata(segments: list, tr: float) -> pd.DataFrame:
    rows = []
    for i, seg in enumerate(segments):
        rows.append({
            "index": i,
            "start_s": float(getattr(seg, "start", float("nan"))),
            "duration_s": float(getattr(seg, "duration", float("nan"))),
            "n_events": len(getattr(seg, "ns_events", []) or []),
        })
    return pd.DataFrame(rows)


def run_one(model: TribeModel, video_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== TRIBE on %s ===", video_path.name)

    # Video event -> get_audio_and_text_events extracts the audio track and
    # chunks both Video and Audio into <=60s segments. audio_only=True skips
    # WhisperX (no speech in percussion). Crucially, the Video events are kept,
    # so V-JEPA2 + DINOv2 run on the (constant grey) frames.
    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    df = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    log.info("Events built (%d rows)", len(df))
    df.to_csv(out_dir / "events.csv", index=False)

    t = time.time()
    preds, segments = model.predict(events=df, verbose=True)
    log.info("Inference done in %.1fs — predictions %s", time.time() - t, preds.shape)

    np.save(out_dir / "predictions.npy", preds.astype(np.float32))
    tr = float(getattr(model.data, "TR", 1.0))
    build_segments_metadata(segments, tr).to_csv(out_dir / "segments.csv", index=False)

    run_info = {
        "stimulus": str(video_path),
        "stimulus_type": "neutral_grey_video + percussion_audio",
        "model": "facebook/tribev2",
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "TR_s": tr,
        "hemodynamic_lag_s": 5,
        "mesh": "fsaverage5",
        "video_pathway": "active (constant grey frames)",
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log.info("Saved outputs to %s", out_dir)


def main() -> int:
    t0 = time.time()
    stim_dir = PROJECT_ROOT / "exp2" / "stimulus"
    out_root = PROJECT_ROOT / "exp2" / "output"
    cache_dir = PROJECT_ROOT / "cache"

    targets = sorted(stim_dir.glob("*_neutral.mp4"))
    if not targets:
        log.error("No *_neutral.mp4 found in %s", stim_dir)
        return 2
    log.info("Neutral-video targets: %s", [p.name for p in targets])

    device = os.environ.get("TRIBE_DEVICE", "mps" if _USE_MPS else "cpu")
    log.info("Device: %s (MPS=%s, dtype=%s)", device, _USE_MPS, _DTYPE)

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
        "facebook/tribev2", cache_folder=cache_dir, device=device,
        config_update=config_update,
    )
    log.info("Model loaded in %.1fs", time.time() - t_load)

    for vp in targets:
        stem = vp.name.replace("_neutral.mp4", "")
        run_one(model, vp, out_root / stem)

    log.info("All done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
