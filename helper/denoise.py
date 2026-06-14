"""Optional audio-cleanup preprocessing for the YouTube translator helper.

Applies noise-reduction / source-separation to each fetched window WAV
(16 kHz mono) BEFORE it is passed to Whisper.  The step is *completely
optional*: if the required library is not installed, or anything fails at
runtime, ``clean()`` logs once and returns the original path unchanged so
the pipeline never crashes.

Install the optional backends:

    pip install deepfilternet          # enables mode="light"  (DeepFilterNet)
    pip install demucs                 # enables mode="music"  (Demucs htdemucs)

Usage from server code::

    from denoise import clean
    wav_path = clean(wav_path, mode)   # mode from user settings: "off"|"light"|"music"

CLI for manual testing::

    python denoise.py <path/to/input.wav> <mode>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

# ---------------------------------------------------------------------------
# Shared helpers (mirror audio.py style)
# ---------------------------------------------------------------------------


def _log(*parts) -> None:
    """Match audio.py / server.py lightweight stdout logging."""
    print("[ytx]", *parts, flush=True)


def _temp_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "yt_translator_audio")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Module-level singletons — loaded at most once per process
# ---------------------------------------------------------------------------

# DeepFilterNet (light mode)
_df_model = None
_df_state = None
_df_sr: Optional[int] = None  # native sample rate reported by init_df

# Demucs (music mode)
_demucs_model = None

# "logged missing dependency" flags — warn only once per mode
_warned_missing: set[str] = set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ffmpeg_to_16k_mono(src: str, dst: str) -> bool:
    """Re-encode *src* to 16 kHz mono WAV at *dst* via ffmpeg.

    Returns True on success, False if ffmpeg is unavailable or fails.
    Mirrors the ``shutil.which("ffmpeg")`` pattern from audio.py.
    """
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i", src,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        dst,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except OSError:
        return False


def _make_temp_wav(prefix: str) -> tuple[int, str]:
    """Create a named temp WAV file in the shared temp dir."""
    return tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=_temp_dir())


# ---------------------------------------------------------------------------
# DeepFilterNet — mode="light"
# ---------------------------------------------------------------------------


def _load_df():
    """Lazy-load and cache the DeepFilterNet model+state singleton."""
    global _df_model, _df_state, _df_sr

    if _df_model is not None:
        return _df_model, _df_state, _df_sr

    from df.enhance import init_df  # type: ignore[import]

    _log("denoise: loading DeepFilterNet model (first use) …")
    _df_model, _df_state, _df_sr = init_df()
    _log(f"denoise: DeepFilterNet ready (native SR={_df_sr})")
    return _df_model, _df_state, _df_sr


def _clean_light(wav_path: str) -> str:
    """Run DeepFilterNet on *wav_path*; return path to cleaned 16kHz mono WAV."""
    from df.enhance import enhance, load_audio, save_audio  # type: ignore[import]

    model, state, df_sr = _load_df()

    # DeepFilterNet's load_audio/save_audio handle resampling internally.
    audio, _ = load_audio(wav_path, sr=df_sr)
    enhanced = enhance(model, state, audio)

    # Write the DeepFilterNet output to a temp file (may be 48 kHz).
    fd_df, df_out = _make_temp_wav("ydn_df_")
    os.close(fd_df)
    save_audio(df_out, enhanced, df_sr)

    # Always down-sample to 16 kHz mono (Whisper's native rate) via ffmpeg.
    fd_out, out_path = _make_temp_wav("ydn_light_")
    os.close(fd_out)

    if _ffmpeg_to_16k_mono(df_out, out_path):
        # Clean up the intermediate DeepFilterNet file.
        try:
            os.remove(df_out)
        except OSError:
            pass
        return out_path

    # ffmpeg unavailable or failed — try using the DeepFilterNet file directly
    # (it may already be 16kHz if the model happens to use that rate).
    _log("denoise: ffmpeg resample unavailable; using DeepFilterNet output as-is")
    try:
        os.remove(out_path)
    except OSError:
        pass
    return df_out


# ---------------------------------------------------------------------------
# Demucs — mode="music"
# ---------------------------------------------------------------------------


def _load_demucs():
    """Lazy-load and cache the Demucs htdemucs model singleton."""
    global _demucs_model

    if _demucs_model is not None:
        return _demucs_model

    from demucs.pretrained import get_model  # type: ignore[import]

    _log("denoise: loading Demucs htdemucs model (first use) …")
    _demucs_model = get_model("htdemucs")
    _log("denoise: Demucs htdemucs ready")
    return _demucs_model


def _clean_music(wav_path: str) -> str:
    """Run Demucs htdemucs on *wav_path*; return path to vocals 16kHz mono WAV."""
    import torch  # type: ignore[import]
    from demucs.apply import apply_model  # type: ignore[import]

    model = _load_demucs()

    # Choose device: CUDA when available, CPU otherwise.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load the window WAV.  Demucs expects a tensor of shape (batch, channels, samples).
    # We use torchaudio here because it's already a transitive dependency of demucs.
    import torchaudio  # type: ignore[import]

    waveform, sr = torchaudio.load(wav_path)  # (C, T)

    # Resample to whatever Demucs expects (htdemucs defaults to 44100 Hz).
    model_sr: int = getattr(model, "samplerate", 44100)
    if sr != model_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
        waveform = resampler(waveform)

    # Ensure stereo — Demucs models expect (batch, 2, T).
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    mixture = waveform.unsqueeze(0).to(device)  # (1, 2, T)

    with torch.no_grad():
        sources = apply_model(model, mixture, device=device)  # (1, stems, 2, T)

    # Find the vocals stem by name.
    stems: list[str] = list(model.sources)
    if "vocals" not in stems:
        raise RuntimeError(
            f"Demucs htdemucs did not expose a 'vocals' stem; found: {stems}"
        )
    vocals_idx = stems.index("vocals")
    vocals = sources[0, vocals_idx]  # (2, T)

    # Downmix to mono.
    vocals_mono = vocals.mean(dim=0, keepdim=True)  # (1, T)

    # Resample to 16 kHz.
    if model_sr != 16000:
        resampler_out = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=16000)
        vocals_mono = resampler_out(vocals_mono)

    # Save to temp WAV.
    fd_out, out_path = _make_temp_wav("ydn_music_")
    os.close(fd_out)
    torchaudio.save(out_path, vocals_mono.cpu(), 16000, format="wav")
    return out_path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def clean(wav_path: str, mode: str) -> str:
    """Return a path to a cleaned 16 kHz mono WAV, or *wav_path* unchanged.

    ``wav_path`` must be an existing 16 kHz mono WAV (as produced by
    ``audio.fetch_window``).  ``mode`` controls which backend is used:

    * ``"off"`` (or any unknown / empty value) — pass-through immediately.
    * ``"light"`` — DeepFilterNet noise suppression.
    * ``"music"`` — Demucs ``htdemucs`` vocal stem extraction (good for
      music-heavy content).

    Graceful degradation guarantee
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Any ``ImportError`` (library not installed), ``RuntimeError``, or
    unexpected exception is caught, logged **once per mode**, and the
    original *wav_path* is returned so the pipeline continues without
    interruption.
    """
    # Fast path: feature disabled.
    if not mode or mode not in ("light", "music"):
        return wav_path

    try:
        if mode == "light":
            return _clean_light(wav_path)
        else:  # mode == "music"
            return _clean_music(wav_path)

    except ImportError as exc:
        if mode not in _warned_missing:
            _warned_missing.add(mode)
            if mode == "light":
                _log(
                    f"denoise: DeepFilterNet not installed (mode={mode!r}); "
                    f"skipping — install with: pip install deepfilternet  [{exc}]"
                )
            else:
                _log(
                    f"denoise: Demucs not installed (mode={mode!r}); "
                    f"skipping — install with: pip install demucs  [{exc}]"
                )
        return wav_path

    except Exception as exc:
        if mode not in _warned_missing:
            _warned_missing.add(mode)
            _log(f"denoise: mode={mode!r} failed, disabling for this session: {exc}")
        return wav_path


# ---------------------------------------------------------------------------
# __main__ CLI — manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: python denoise.py <wav_path> <mode>\n"
            "  modes: off | light | music",
            file=sys.stderr,
        )
        sys.exit(2)

    in_path, in_mode = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"error: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out = clean(in_path, in_mode)
    print(out)
