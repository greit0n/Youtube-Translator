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
MODEL_NAME = "large-v3"

# Cached singletons so the (expensive) model load happens at most once.
_model = None
_device: Optional[str] = None
_compute_type: Optional[str] = None


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


def load_model():
    """Load the faster-whisper model once and cache it.

    Returns the loaded model. Sets module globals describing the real device
    and compute type so the server can report them via /health.
    """
    global _model, _device, _compute_type

    if _model is not None:
        return _model

    from faster_whisper import WhisperModel

    device, compute_type = _detect_device()
    try:
        _model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    except Exception:
        # CUDA may be advertised but unusable (driver/cuDNN mismatch). Fall
        # back to CPU so the tool still works, and report the real device.
        if device != "cpu":
            device, compute_type = "cpu", "int8"
            _model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
        else:
            raise

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

    print(f"Loading model {MODEL_NAME} ...", file=sys.stderr)
    load_model()
    print(f"Using device={get_device()} compute={_compute_type}", file=sys.stderr)

    for start, end, text in transcribe(audio_path, language=language):
        print(f"[{_format_ts(start)} --> {_format_ts(end)}] {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
