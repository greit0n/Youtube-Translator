"""Whisper transcription for the YouTube translator helper.

Now powered by **WhisperX** (`whisperx`), which wraps faster-whisper with
batched VAD inference and integrates pyannote diarization (wired in a later
step). The public surface here is unchanged so the rest of the server barely
changes.

Two modes:

- `transcribe()`  -> task="translate": any language -> English directly with
  Whisper. Used as the fallback engine (engine == "whisper").
- `transcribe_source()` -> task="transcribe": faithful SOURCE-language text,
  which a local Ollama model then translates to English (engine == "ollama").
  This two-stage path preserves profanity / slang / names far better than
  Whisper's built-in translate.

Loads WhisperX large-v3 on the GPU (CUDA/float16) when available and falls back
to CPU/int8 otherwise.

KEY WhisperX DIFFERENCE: the `task` ("translate" vs "transcribe") is baked into
the model at `whisperx.load_model(...)`, NOT passed per-call like
faster-whisper. Our two engines need different tasks, so the model singleton is
keyed by **(name, task)** and reloaded when either changes.

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

# Max length (seconds) of a single VAD-merged chunk -> roughly the length of one
# on-screen caption. WhisperX defaults this to 30s, which merges ~20s of speech
# into ONE giant segment that fills the whole screen. faster-whisper used to emit
# sentence-sized segments; a small chunk_size restores that bite-sized feel.
CHUNK_SIZE = 6.0

# Cached singletons so the (expensive) model load happens at most once.
# The model is keyed by (name, task): WhisperX bakes the task into the loaded
# model, so a task change forces a reload just like a model-name change does.
_model = None
_model_name: Optional[str] = None  # name of the currently loaded model
_model_task: Optional[str] = None  # task baked into the currently loaded model
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


def get_model_task() -> Optional[str]:
    """Task baked into the currently loaded model, or None if none loaded."""
    return _model_task


def _detect_device() -> Tuple[str, str]:
    """Return (device, compute_type), preferring CUDA/float16."""
    try:
        import torch  # noqa: F401  (only used to probe CUDA availability)

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        # torch may not be installed; whisperx/faster-whisper can still use CUDA
        # via ctranslate2. Probe ctranslate2 directly as a fallback.
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
    return "cpu", "int8"


def _asr_options(
    beam_size: int = 5,
    hotwords: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> dict:
    """Build the WhisperX `asr_options` dict.

    WhisperX bakes decoding options into the model at load time (unlike
    faster-whisper, which takes them per `transcribe()` call). beam_size,
    hotwords and initial_prompt are all valid asr_options keys, so we apply them
    here. We keep this minimal/robust: only set keys that differ from defaults.
    """
    opts: dict = {"beam_size": beam_size}
    if hotwords:
        opts["hotwords"] = hotwords
    if initial_prompt:
        opts["initial_prompt"] = initial_prompt
    return opts


def load_model(
    name: Optional[str] = None,
    task: str = "translate",
    beam_size: int = 5,
    hotwords: Optional[str] = None,
    initial_prompt: Optional[str] = None,
):
    """Load a WhisperX model (singleton, keyed by (name, task)) and cache it.

    `name` selects the Whisper model tier; when None it is auto-resolved from
    available VRAM via resolve_model("auto"). `task` is baked into the WhisperX
    model ("translate" -> English; "transcribe" -> faithful source language).

    The model is a singleton keyed by (name, task): re-requesting the same
    (name, task) is a no-op, but switching EITHER frees the old model (and the
    CUDA cache) before loading the new one so we don't hold two large models in
    VRAM at once.

    Returns the loaded model and sets module globals describing the loaded model
    name, task, real device, and compute type so the server can report them.
    """
    global _model, _model_name, _model_task, _device, _compute_type

    name = name or resolve_model("auto")

    # Already loaded AND it's the requested (name, task) -> reuse.
    if _model is not None and _model_name == name and _model_task == task:
        return _model

    # A DIFFERENT (name, task) is loaded -> free it before loading the new one.
    if _model is not None:
        _model = None
        _model_name = None
        _model_task = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # Lazy import so `python -m py_compile` (and CPU-only/no-whisperx envs that
    # never call this) keep working without the package installed.
    import whisperx

    device, compute_type = _detect_device()
    asr_options = _asr_options(beam_size, hotwords, initial_prompt)
    try:
        _model = whisperx.load_model(
            name,
            device=device,
            compute_type=compute_type,
            task=task,
            asr_options=asr_options,
        )
    except Exception:
        # CUDA may be advertised but unusable (driver/cuDNN mismatch). Fall back
        # to CPU so the tool still works, and report the real device.
        if device != "cpu":
            device, compute_type = "cpu", "int8"
            _model = whisperx.load_model(
                name,
                device=device,
                compute_type=compute_type,
                task=task,
                asr_options=asr_options,
            )
        else:
            raise

    _model_name = name
    _model_task = task
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
    model_name: Optional[str] = None,
) -> Iterator[Tuple[float, float, str, str]]:
    """Translate `audio_path` to English, yielding (start, end, text, detected)
    segments where `detected` is the auto-detected SOURCE language code (so the
    caller can filter windows by spoken language).

    Args:
        audio_path: path to a decodable audio file (WhisperX uses ffmpeg).
        language: kept for signature compat; NOT forced (see note below) — the
            caller filters by the detected language instead.
        time_offset: seconds to ADD to every timestamp. Used when the audio was
            clipped / windowed so timestamps stay absolute to the original video.
        hotwords: optional space-separated hint words biasing recognition
            (names, game terms).
        initial_prompt: optional context prompt.
        beam_size: beam width; lower => faster first caption, slightly less
            accurate.
        model_name: explicit model tier; None auto-resolves from VRAM.

    Yields segments lazily from the WhisperX result.

    NOTE: WhisperX bakes `task`, `beam_size`, `hotwords` and `initial_prompt`
    into the model at LOAD time (via asr_options), not per `transcribe()` call.
    `hotwords`/`initial_prompt` come from the glossary and are constant for a
    session, so baking them at first load is correct. `beam_size`, however, the
    server varies per window (1 at the playhead, 5 ahead). Honoring that here
    would be ORDER-DEPENDENT — whichever window loads first would bake its beam
    for the whole session (and the first window is usually the playhead, beam=1,
    which would silently cap quality everywhere). WhisperX's batched VAD makes
    the beam=1 latency trick unnecessary, so we always load at the quality beam
    (5) and ignore the per-call `beam_size`.
    """
    import whisperx

    model = load_model(
        model_name,
        task="translate",  # any language -> English
        beam_size=5,  # always quality beam; see NOTE (per-call beam ignored)
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )

    audio = whisperx.load_audio(audio_path)
    # CRITICAL: pass task on the transcribe() call, not just at load. whisperx's
    # FasterWhisperPipeline.transcribe defaults task to "transcribe" when omitted
    # (asr.py: `task = task or "transcribe"`), IGNORING the load-time task — which
    # silently produced source-language (e.g. Czech) output instead of English.
    #
    # `language` is intentionally NOT forced here even when provided: the caller
    # uses the detected language (4th tuple element) to FILTER windows by spoken
    # language (e.g. subtitle Czech, skip English game audio). Forcing would make
    # whisper transcribe English-as-Czech garbage instead of detecting "en".
    kwargs = dict(batch_size=16, task="translate", chunk_size=CHUNK_SIZE)
    result = model.transcribe(audio, **kwargs)

    detected = result.get("language") or language or "unknown"
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        yield (seg["start"] + time_offset, seg["end"] + time_offset, text, detected)


def transcribe_source(
    audio_path: str,
    language: Optional[str] = None,
    time_offset: float = 0.0,
    hotwords: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Iterator[Tuple[float, float, str, str]]:
    """Transcribe `audio_path` in its SOURCE language (no translation).

    Same shape as `transcribe()` but task="transcribe", and each yielded tuple
    additionally carries the detected source language code so the server can
    build a faithful LLM translation prompt:

        (start, end, text, lang)

    `hotwords` / `initial_prompt` are applied via load-time asr_options (see the
    note in `transcribe()`).
    """
    import whisperx

    model = load_model(
        model_name,
        task="transcribe",  # faithful source-language text
        beam_size=5,
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )

    audio = whisperx.load_audio(audio_path)
    # Pass task explicitly (see note in transcribe()): omitting it makes whisperx
    # default to "transcribe" — which is what we want here, but be explicit so the
    # behavior can't silently change.
    kwargs = dict(batch_size=16, task="transcribe", chunk_size=CHUNK_SIZE)
    if language is not None:
        kwargs["language"] = language
    result = model.transcribe(audio, **kwargs)

    # result["language"] is the detected (or forced) source language code.
    detected = result.get("language") or language or "unknown"

    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        yield (seg["start"] + time_offset, seg["end"] + time_offset, text, detected)


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
    load_model(task="translate")
    print(
        f"Using model={get_model_name()} task={get_model_task()} "
        f"device={get_device()} compute={_compute_type}",
        file=sys.stderr,
    )

    for start, end, text, detected in transcribe(audio_path, language=language):
        print(f"[{_format_ts(start)} --> {_format_ts(end)}] ({detected}) {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
