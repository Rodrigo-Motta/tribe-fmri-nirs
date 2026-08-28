"""Run TRIBE v2 on the project stimulus video and save predictions.

Usage:
    # default device (auto -> CPU on this machine, MPS available)
    python run_tribe.py

    # force a device, e.g. Apple Silicon GPU
    TRIBE_DEVICE=mps python run_tribe.py

Outputs are written to ./output/:
    - predictions.npy            : (n_timesteps, n_vertices) brain predictions
    - segments.csv               : per-timestep segment metadata
    - events.csv                 : full events dataframe (audio/video/words/...)
    - run_info.json              : run metadata (shapes, device, timings, ...)
    - brain_preview.png          : (optional) cortical surface preview
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- load .env (HF_KEY) into the environment ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Make the HuggingFace token available to huggingface_hub / transformers.
hf_token = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from tribev2.demo_utils import TribeModel, get_audio_and_text_events  # noqa: E402
import typing as tp  # noqa: E402

# --- monkeypatch: MPS + fp16 on Apple Silicon (16GB RAM needs fp16) ----------
# neuralset's HuggingFaceMixin only resolves `device="auto"` to cuda/cpu. The
# config field is Literal[auto,cpu,cuda,accelerate] so we can't put "mps" in
# config; instead we set "auto" and remap to "mps" here.
import neuralset.extractors.base as _ns_base  # noqa: E402

_USE_MPS = torch.backends.mps.is_available()
_DTYPE = torch.float16 if _USE_MPS else torch.float32

_orig_hf_post_init = _ns_base.HuggingFaceMixin.model_post_init


def _hf_post_init(self, log__: tp.Any) -> None:
    _orig_hf_post_init(self, log__)
    if _USE_MPS and self.device == "cpu":
        self.device = "mps"


_ns_base.HuggingFaceMixin.model_post_init = _hf_post_init

# Patch transformers.AutoModel.from_pretrained to load in fp16 on MPS. This
# halves V-JEPA2 ViT-G (5.6GB -> 2.8GB) so it fits in 16GB unified memory during
# the .to(mps) transfer. Applies to DINOv2 / Wav2Vec-BERT / V-JEPA2 alike.
import transformers  # noqa: E402

_orig_auto_from_pretrained = transformers.AutoModel.from_pretrained.__func__


def _auto_from_pretrained_patched(*args, **kwargs):
    if _USE_MPS and "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = _DTYPE
    return _orig_auto_from_pretrained(*args, **kwargs)


transformers.AutoModel.from_pretrained = classmethod(_auto_from_pretrained_patched)
print(f"[run_tribe] Patched AutoModel.from_pretrained: torch_dtype={_DTYPE} on MPS={_USE_MPS}", flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_tribe")


# --- monkeypatch WhisperX to be CPU-friendly ---------------------------------
# tribev2 hardcodes compute_type="float16" which only works on CUDA. On CPU we
# switch to "int8" so transcription does not crash. Behaviour is unchanged on GPU.
from tribev2.eventstransforms import ExtractWordsFromAudio  # noqa: E402


def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
    import json
    import subprocess
    import tempfile

    language_codes = dict(
        english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
    )
    if language not in language_codes:
        raise ValueError(f"Language {language} not supported")

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    compute_type = "float16" if has_cuda else "int8"

    with tempfile.TemporaryDirectory() as output_dir:
        log.info("Running whisperx (device=%s, compute_type=%s)...", device, compute_type)
        cmd = [
            "uvx", "whisperx", str(wav_filename),
            "--model", "large-v3",
            "--language", language_codes[language],
            "--device", device,
            "--compute_type", compute_type,
            "--batch_size", "16",
            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
            "--output_dir", output_dir,
            "--output_format", "json",
        ]
        cmd = [c for c in cmd if c]
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"whisperx failed:\n{result.stderr}")
        json_path = Path(output_dir) / f"{wav_filename.stem}.json"
        transcript = json.loads(json_path.read_text())

    words = []
    for i, segment in enumerate(transcript["segments"]):
        sentence = segment["text"].replace('"', "")
        for word in segment["words"]:
            if "start" not in word:
                continue
            words.append({
                "text": word["word"].replace('"', ""),
                "start": word["start"],
                "duration": word["end"] - word["start"],
                "sequence_id": i,
                "sentence": sentence,
            })
    return pd.DataFrame(words)


ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_get_transcript_from_audio)


def build_segments_metadata(segments: list, tr: float) -> pd.DataFrame:
    rows = []
    for i, seg in enumerate(segments):
        start = float(getattr(seg, "start", float("nan")))
        duration = float(getattr(seg, "duration", float("nan")))
        ns_events = getattr(seg, "ns_events", None)
        n_events = len(ns_events) if ns_events is not None else None
        words = []
        events = getattr(seg, "events", None)
        if events is not None and isinstance(events, pd.DataFrame) and "text" in events:
            word_rows = events[events.get("type") == "Word"]
            words = word_rows["text"].astype(str).tolist()
        rows.append({
            "index": i,
            "start_s": start,
            "duration_s": duration,
            "n_events": n_events,
            "words": " ".join(words),
        })
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()

    video_path = PROJECT_ROOT / "stimulus" / "video_with_audio.mp4"
    if not video_path.is_file():
        log.error("Stimulus video not found: %s", video_path)
        return 2

    out_dir = PROJECT_ROOT / "output"
    cache_dir = PROJECT_ROOT / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = os.environ.get("TRIBE_DEVICE", "mps" if _USE_MPS else "cpu")
    log.info("Video: %s", video_path)
    log.info("Output dir: %s", out_dir)
    log.info("Cache dir: %s", cache_dir)
    log.info("Device: %s (CUDA=%s, MPS=%s, dtype=%s)",
             device, torch.cuda.is_available(), _USE_MPS, _DTYPE)

    # --- load model ----------------------------------------------------------
    t_load = time.time()
    # The shipped config.yaml hardcodes `device: cuda` for every feature
    # extractor; set them to "auto" and let our monkeypatch above resolve to
    # MPS on Apple Silicon (or CPU otherwise). The brain model itself is moved
    # to `device` via from_pretrained.
    # keep_in_ram=false frees each feature model after its extractor runs, so
    # V-JEPA2 (2.8GB fp16) does not pile on top of Wav2Vec-BERT etc.
    # video/image frequency reduced 2.0->1.0 Hz to halve the frame count (cuts
    # runtime ~2x with negligible effect on 1s-TR predictions).
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
        # V-JEPA2 fpc64 normally uses 64 frames/clip (~175s/clip on MPS). 16
        # frames drops it to ~9s/clip (19x faster) with the same 1408-dim
        # features; tokens are mean-aggregated downstream so quality stays good.
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
    # debug: confirm the extractor actually picked up our overrides
    try:
        vf = model.data.video_feature
        log.info("video_feature: frequency=%s num_frames=%s image.device=%s",
                 getattr(vf, "frequency", "?"), getattr(vf, "num_frames", "?"),
                 getattr(vf.image, "device", "?"))
    except Exception as e:
        log.warning("could not introspect video_feature: %r", e)

    # --- build events dataframe (audio + video only, NO transcription) ----------
    # The stimulus has no speech, so we skip WhisperX transcription and the text
    # pipeline entirely. Missing text features are encoded as zeros by the model.
    t_events = time.time()
    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    df = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    log.info("Events built in %.1fs (%d rows, no transcription)", time.time() - t_events, len(df))
    df.to_csv(out_dir / "events.csv", index=False)

    # --- run inference -------------------------------------------------------
    t_pred = time.time()
    preds, segments = model.predict(events=df, verbose=True)
    log.info("Inference done in %.1fs", time.time() - t_pred)
    log.info("Predictions shape: %s (n_timesteps, n_vertices)", preds.shape)

    # --- save outputs --------------------------------------------------------
    np.save(out_dir / "predictions.npy", preds.astype(np.float32))

    tr = float(getattr(model.data, "TR", 1.0))
    seg_df = build_segments_metadata(segments, tr)
    seg_df.to_csv(out_dir / "segments.csv", index=False)

    run_info = {
        "stimulus": str(video_path),
        "stimulus_duration_s": 435.699229,
        "model": "facebook/tribev2",
        "device": str(getattr(model, "_model", None).device if getattr(model, "_model", None) is not None else device),
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "TR_s": tr,
        "hemodynamic_lag_s": 5,
        "mesh": "fsaverage5",
        "note": "Predictions are offset 5s in the past to compensate for the hemodynamic lag.",
        "n_events_rows": int(len(df)),
        "timings_s": {
            "load": round(time.time() - t_load, 1),
            "events": round(time.time() - t_events, 1),
            "predict": round(time.time() - t_pred, 1),
            "total": round(time.time() - t0, 1),
        },
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log.info("Saved run_info.json")

    # --- optional brain-surface preview --------------------------------------
    try:
        from tribev2.plotting import PlotBrain
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plotter = PlotBrain(mesh="fsaverage5")
        # preview ~12 evenly spaced timesteps
        n = preds.shape[0]
        k = min(12, n)
        idx = np.linspace(0, n - 1, k).astype(int)
        sub_preds = preds[idx]
        sub_segs = [segments[i] for i in idx]
        timestamps = [int(round(segments[i].start)) for i in idx]
        fig = plotter.plot_timesteps(
            sub_preds,
            segments=sub_segs,
            views="left",
            timestamps=timestamps,
            norm_percentile=95,
            cmap="hot",
            show_stimuli=False,
        )
        fig.savefig(out_dir / "brain_preview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved brain_preview.png")
    except Exception as e:  # visualization is optional
        log.warning("Brain visualization skipped: %r", e)

    log.info("All done in %.1fs. Outputs in %s", time.time() - t0, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
