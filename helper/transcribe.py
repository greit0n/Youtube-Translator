"""Whisper transcription for the YouTube translator helper.

Two modes:

- `transcribe()`  -> task="translate": any language -> English directly with
  Whisper. Used as the fallback engine (engine == "whisper").
- `transcribe_source()` -> task="transcribe": faithful SOURCE-language text,
  which a local Ollama model then translates to English (engine == "ollama").
  This two-stage path preserves profanity / slang / names far better than
  Whisper's built-in translate.

Loads faster-whisper large-v3 on the GPU (CUDA/float16) when available and
falls back to CPU/int8 otherwise.

CLI:
    python transcribe.py <audio_file> [language]
"""

from __future__ import annotations

import sys
from typing import Iterator, Optional, Tuple

# large-v3 is used (NOT large-v3-turbo): turbo is noticeably weaker at the
# translate task, and translation quality is the whole point of this tool.
# large-v3 stays the preferred/default tier; it is no longer the ONLY option —
# resolve_model() can pick a smaller tier when VRAM is tight or the user asks.
MODEL_NAME = "large-v3"

# Cached singletons so the (expensive) model load happens at most once.
_model = None
_model_name: Optional[str] = None  # name of the currently loaded model
_device: Optional[str] = None
_compute_type: Optional[str] = None


def _detect_vram_gb() -> float:
    """Total VRAM of GPU 0 in GiB, or 0.0 on CPU / no-CUDA / any error."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def resolve_model(quality: str) -> str:
    """Map a Quality setting to a concrete Whisper model name.

    Explicit tiers:
        "max"      -> "large-v3"
        "balanced" -> "medium"
        "lite"     -> "small"
    "auto" (or anything unknown) picks by available VRAM, keeping large-v3
    quality whenever it fits (the whole point of the tool):
        >= 10 GB -> "large-v3"
        >=  5 GB -> "medium"
        >= 2.5 GB -> "small"
        else      -> "base"

    NOTE: never returns a distil-* model — those are English-only and break the
    translate task this tool relies on.
    """
    q = (quality or "auto").lower()
    if q == "max":
        return "large-v3"
    if q == "balanced":
        return "medium"
    if q == "lite":
        return "small"
    # "auto" / unknown -> size by VRAM.
    vram = _detect_vram_gb()
    if vram >= 10.0:
        return "large-v3"
    if vram >= 5.0:
        return "medium"
    if vram >= 2.5:
        return "small"
    return "base"


def get_model_name() -> str:
    """Name of the loaded model, or the auto-resolved choice if none loaded."""
    if _model_name is not None:
        return _model_name
    return resolve_model("auto")


def _detect_device() -> Tuple[str, str]:
    """Return (device, compute_type), preferring CUDA/float16."""
    try:
        import torch  # noqa: F401  (only used to probe CUDA availability)

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        # torch may not be installed; faster-whisper can still use CUDA via
        # ctranslate2. Probe ctranslate2 directly as a fallback.
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
    return "cpu", "int8"


def load_model(name: Optional[str] = None):
    """Load a named faster-whisper model (singleton) and cache it.

    `name` selects the Whisper model tier; when None it is auto-resolved from
    available VRAM via resolve_model("auto"). The model is a NAMED singleton:
    re-requesting the already-loaded model is a no-op, but switching to a
    different tier frees the old model (and the CUDA cache) before loading.

    Returns the loaded model. Sets module globals describing the loaded model
    name, real device, and compute type so the server can report them.
    """
    global _model, _model_name, _device, _compute_type

    if name is None:
        name = resolve_model("auto")

    # Already loaded AND it's the requested model -> reuse.
    if _model is not None and _model_name == name:
        return _model

    # A DIFFERENT model is loaded -> free it before loading the new one so we
    # don't hold two large models in VRAM at once.
    if _model is not None:
        _model = None
        _model_name = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    from faster_whisper import WhisperModel

    device, compute_type = _detect_device()
    try:
        _model = WhisperModel(name, device=device, compute_type=compute_type)
    except Exception:
        # CUDA may be advertised but unusable (driver/cuDNN mismatch). Fall
        # back to CPU so the tool still works, and report the real device.
        if device != "cpu":
            device, compute_type = "cpu", "int8"
            _model = WhisperModel(name, device=device, compute_type=compute_type)
        else:
            raise

    _model_name = name
    _device = device
    _compute_type = compute_type
    return _model


def get_device() -> str:
    """Return the device the model is (or would be) loaded on: 'cuda'|'cpu'."""
    if _device is not None:
        return _device
    return _detect_device()[0]


def is_cuda() -> bool:
    return get_device() == "cuda"


def is_loaded() -> bool:
    return _model is not None


def transcribe(
    audio_path: str,
    language: Optional[str] = None,
    time_offset: float = 0.0,
    hotwords: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    beam_size: int = 5,
) -> Iterator[Tuple[float, float, str]]:
    """Translate `audio_path` to English, yielding (start, end, text) segments.

    Args:
        audio_path: path to a decodable audio file (faster-whisper uses PyAV).
        language: source language ISO code (e.g. "cs") or None to auto-detect.
        time_offset: seconds to ADD to every timestamp. Used when the audio was
            clipped / windowed so timestamps stay absolute to the original video.
        hotwords: optional space-separated hint words biasing recognition
            (names, game terms) — passed through to faster-whisper.
        initial_prompt: optional context prompt passed through to faster-whisper.

    Yields segments lazily as Whisper produces them.
    """
    model = load_model()

    kwargs = dict(
        task="translate",  # any language -> English
        language=language,  # None => auto-detect
        beam_size=beam_size,  # lower => faster first caption, slightly less accurate
        vad_filter=True,  # skip long silences for speed/quality
    )
    if hotwords:
        kwargs["hotwords"] = hotwords
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    segments, _info = model.transcribe(audio_path, **kwargs)

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        yield (seg.start + time_offset, seg.end + time_offset, text)


def transcribe_source(
    audio_path: str,
    language: Optional[str] = None,
    time_offset: float = 0.0,
    hotwords: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> Iterator[Tuple[float, float, str, str]]:
    """Transcribe `audio_path` in its SOURCE language (no translation).

    Same shape as `transcribe()` but task="transcribe", and each yielded tuple
    additionally carries the detected source language code so the server can
    build a faithful LLM translation prompt:

        (start, end, text, lang)

    `vad_filter=True` is always on. `hotwords` / `initial_prompt` are passed
    through to faster-whisper when provided.
    """
    model = load_model()

    kwargs = dict(
        task="transcribe",  # faithful source-language text
        language=language,  # None => auto-detect
        beam_size=5,
        vad_filter=True,
    )
    if hotwords:
        kwargs["hotwords"] = hotwords
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    segments, info = model.transcribe(audio_path, **kwargs)

    # info.language is the detected (or forced) source language code.
    detected = getattr(info, "language", None) or language or "unknown"

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        yield (seg.start + time_offset, seg.end + time_offset, text, detected)


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python transcribe.py <audio_file> [language]", file=sys.stderr)
        return 2

    audio_path = argv[1]
    language = argv[2] if len(argv) > 2 else None

    print("Loading model (auto tier) ...", file=sys.stderr)
    load_model()
    print(
        f"Using model={get_model_name()} device={get_device()} compute={_compute_type}",
        file=sys.stderr,
    )

    for start, end, text in transcribe(audio_path, language=language):
        print(f"[{_format_ts(start)} --> {_format_ts(end)}] {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
