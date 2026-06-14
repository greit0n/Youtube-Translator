"""Optional speaker diarization for the YouTube translator helper.

This module is OPTIONAL — the rest of the helper works without it. It assigns
stable "Speaker 1", "Speaker 2", … labels to Whisper segments, keeping those
labels consistent across multiple windowed fetches for the same physical voice.

Dependencies (not required at import time — lazily imported on first use):
    pip install whisperx pyannote.audio

HuggingFace token requirement:
    Diarization uses pyannote models that require accepting the model terms on
    https://huggingface.co/pyannote/speaker-diarization-3.1 and
    https://huggingface.co/pyannote/embedding, then supplying your HF access token
    by one of:
      1. Environment variable:  HF_TOKEN=hf_...  (or HUGGINGFACE_TOKEN=hf_...)
      2. File: helper/hf_token.txt  (place your token there, one line, no quotes)

    Without a token the module logs once and all label_segments() calls return
    the input segments unchanged (no 'speaker' key added).

Cross-window stability:
    pyannote/WhisperX assigns per-call LOCAL speaker labels (SPEAKER_00, …) that
    reset each window. SpeakerTracker maintains a session-scoped registry of
    GLOBAL speakers, each represented by a running-mean voice embedding centroid.
    On each window it:
      1. Computes a voice embedding (pyannote/embedding) cropped to that speaker's
         turns within the WAV.
      2. Cosine-compares against every known global centroid.
      3. If similarity >= SIMILARITY_THRESHOLD: reuses that global label and
         updates its centroid with a running mean.
      4. Otherwise: registers a new global speaker ("Speaker N").
    This gives labels that remain stable across arbitrarily many windows.
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Logging — same _log / [ytx] convention used throughout audio.py / server.py
# ---------------------------------------------------------------------------

def _log(*parts) -> None:
    print("[ytx]", *parts, flush=True)


# ---------------------------------------------------------------------------
# HuggingFace token resolution
# ---------------------------------------------------------------------------

def _resolve_hf_token() -> Optional[str]:
    """Return the HF token from env or helper/hf_token.txt, or None."""
    for env_var in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_token.txt")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as fh:
                val = fh.read().strip()
            if val:
                return val
        except OSError:
            pass

    return None


# ---------------------------------------------------------------------------
# Module-level "warn once" flags — printed at most once per process lifetime
# ---------------------------------------------------------------------------

_warned_no_token: bool = False
_warned_no_libs: bool = False

# Cosine-similarity threshold for matching a new local speaker to an existing
# global centroid. 0.0 = always new speaker, 1.0 = perfect match required.
# 0.65 is a sensible middle ground for pyannote/embedding in typical content.
SIMILARITY_THRESHOLD: float = 0.65

# --- Voice enrollment ------------------------------------------------------
# Drop an audio clip of a single person (e.g. 1-3 min of just them talking) into
# helper/enroll/ and that voice gets a FIXED label (the filename, sans extension)
# and is flagged `enrolled` so the extension can paint it a chosen colour — even
# among several speakers. No training: we compute one pyannote/embedding
# voiceprint from the clip and cosine-match every detected speaker against it.
ENROLL_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enroll")
ENROLL_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm")
# Enrolled references are clean, single-speaker recordings, while in-video turns
# are short and noisy, so the cross-recording cosine runs lower than the
# window-to-window number. A slightly looser threshold catches the enrolled
# voice without (in practice) grabbing other speakers, who sit well below ~0.45.
ENROLL_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Tiny numpy-free cosine similarity (avoids a hard numpy import at module level)
# ---------------------------------------------------------------------------

def enrolled_names() -> list[str]:
    """Names of voices enrolled in ENROLL_DIR (filenames, sans extension).

    Cheap directory listing — no model load — so /health can confirm the user's
    clip was found without spinning up diarization.
    """
    try:
        if not os.path.isdir(ENROLL_DIR):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(ENROLL_DIR)
            if os.path.splitext(f)[1].lower() in ENROLL_EXTS
        )
    except OSError:
        return []


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two 1-D numeric sequences. Returns float in [-1, 1]."""
    import math
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) ** 2 for x in a))
    norm_b = math.sqrt(sum(float(x) ** 2 for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SpeakerTracker — the public interface
# ---------------------------------------------------------------------------

class SpeakerTracker:
    """Per-session tracker that assigns stable global speaker labels.

    Lifecycle:
        tracker = SpeakerTracker(device="cuda")   # one per WS session
        segs = tracker.label_segments(wav_path, segments, window_start)

    If diarization is unavailable (no token, missing libs, or any runtime
    error) label_segments() returns the original segments unchanged and never
    raises.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device

        # Global speaker registry: list of {"label": "Speaker N", "centroid": list[float], "count": int}
        self._speakers: list[dict] = []

        # Lazy-loaded pipeline and embedding model (None until first use)
        self._pipeline = None       # WhisperX / pyannote diarization pipeline
        self._embedding = None      # pyannote.audio Inference for speaker embeddings

        # Whether we've already attempted (and potentially failed) to load
        self._load_attempted: bool = False
        self._load_ok: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """Return True if diarization is ready (token + libs both present)."""
        global _warned_no_token, _warned_no_libs

        # Quick check without paying the full load cost.
        if _resolve_hf_token() is None:
            return False

        # Probe importability of the two heavy deps — no real import side-effects.
        # NB: `import importlib` alone does NOT bind importlib.util; import the
        # submodule explicitly or find_spec raises AttributeError (-> False).
        try:
            import importlib.util
            if importlib.util.find_spec("whisperx") is None:
                return False
            if importlib.util.find_spec("pyannote.audio") is None:
                return False
        except Exception:
            return False

        return True

    def label_segments(
        self,
        wav_path: str,
        segments: list[dict],
        window_start: float = 0.0,
    ) -> list[dict]:
        """Assign stable global speaker labels to *segments*.

        Args:
            wav_path:      Path to the window WAV (16kHz mono, matches the window).
            segments:      List of segment dicts with at least 'start' and 'end'
                           as ABSOLUTE video timestamps (seconds).
            window_start:  Absolute video timestamp of the WAV's t=0 (seconds).
                           Used to convert pyannote's relative turn times to absolute
                           before overlap-matching against segment boundaries.

        Returns:
            The same segment dicts (shallow-copied) with a 'speaker' key added,
            e.g. "Speaker 1".  On any failure the ORIGINAL list is returned
            unchanged (no 'speaker' key, no exception raised).
        """
        if not segments:
            return segments

        try:
            return self._label_segments_inner(wav_path, segments, window_start)
        except Exception as exc:
            _log(f"diarize: label_segments failed (returning unlabelled): {exc}")
            return segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load the pipeline and embedding model. Returns True on success."""
        global _warned_no_token, _warned_no_libs

        if self._load_attempted:
            return self._load_ok

        self._load_attempted = True

        # --- token check -------------------------------------------------
        token = _resolve_hf_token()
        if token is None:
            if not _warned_no_token:
                _log(
                    "diarize: disabled — no HF token found "
                    "(set HF_TOKEN env var or create helper/hf_token.txt; "
                    "see how-to.html for setup details)"
                )
                _warned_no_token = True
            self._load_ok = False
            return False

        # --- library check -----------------------------------------------
        try:
            import whisperx  # noqa: F401
            from pyannote.audio import Pipeline, Inference, Model
        except ImportError as exc:
            if not _warned_no_libs:
                _log(
                    f"diarize: disabled — required libraries not installed "
                    f"({exc}). Run: pip install whisperx pyannote.audio"
                )
                _warned_no_libs = True
            self._load_ok = False
            return False

        # --- load diarization pipeline -----------------------------------
        try:
            _log("diarize: loading diarization pipeline (first use) …")
            # whisperx 3.8.x exposes the class under whisperx.diarize, not at the
            # top level (older versions had whisperx.DiarizationPipeline).
            from whisperx.diarize import DiarizationPipeline

            # whisperx 3.8.x signature: DiarizationPipeline(model_name=None,
            # token=None, device=..., cache_dir=None) — the auth kwarg is `token`
            # (older versions used `use_auth_token`).
            # Use speaker-diarization-community-1 (pyannote's newer/better model
            # and whisperx's default). It's self-contained — needs only its own
            # gated-repo acceptance. (NB: under pyannote.audio 4.x even the older
            # 3.1 pipeline pulls a PLDA asset from community-1, so accepting
            # community-1 is required regardless.) The cross-window voice
            # fingerprint still uses pyannote/embedding (loaded separately below).
            self._pipeline = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-community-1",
                token=token,
                device=self._device,
            )
            _log(f"diarize: pipeline loaded on {self._device}")
        except Exception as exc:
            _log(f"diarize: failed to load diarization pipeline: {exc}")
            self._load_ok = False
            return False

        # --- load embedding model ----------------------------------------
        try:
            _log("diarize: loading speaker embedding model (first use) …")
            from pyannote.audio import Inference, Model

            # pyannote.audio 4.x renamed the auth kwarg use_auth_token -> token
            # (passing use_auth_token silently drops it -> 401 on the gated repo).
            emb_model = Model.from_pretrained(
                "pyannote/embedding", token=token
            )
            self._embedding = Inference(
                emb_model,
                window="whole",
            )
            _log("diarize: embedding model loaded")
        except Exception as exc:
            _log(
                f"diarize: failed to load embedding model — "
                f"speaker tracking will use label-order only: {exc}"
            )
            # Diarization still works, just without cross-window stability;
            # keep self._embedding = None as the signal.

        # --- load enrolled voiceprints (optional) ------------------------
        # Needs the embedding model; silently skipped if it failed to load or
        # the enroll/ dir is empty/absent. Pre-seeds the speaker registry so the
        # enrolled voice is matched (and flagged) from the very first window.
        try:
            self._load_enrollments()
        except Exception as exc:
            _log(f"diarize: enrollment load failed (continuing without): {exc}")

        self._load_ok = True
        return True

    def _load_enrollments(self) -> None:
        """Scan ENROLL_DIR and pre-seed the registry with locked reference voices.

        Each audio file becomes one enrolled speaker whose label is the filename
        (without extension). Its centroid is a whole-file voiceprint and is
        LOCKED (never drifts via the running-mean update) so the reference stays
        clean across the session.
        """
        if self._embedding is None:
            return
        if not os.path.isdir(ENROLL_DIR):
            return

        files = sorted(
            f for f in os.listdir(ENROLL_DIR)
            if os.path.splitext(f)[1].lower() in ENROLL_EXTS
        )
        if not files:
            return

        for fname in files:
            name = os.path.splitext(fname)[0]
            path = os.path.join(ENROLL_DIR, fname)
            emb = self._embed_whole(path)
            if emb is None:
                _log(f"diarize: enrollment '{name}' skipped (no embedding)")
                continue
            self._speakers.append({
                "label": name,
                "centroid": emb,
                "count": 1,
                "enrolled": True,   # flag segments + paint a chosen colour
                "locked": True,     # don't drift the clean reference
            })
            _log(f"diarize: enrolled voice '{name}' from {fname}")

    def _embed_whole(self, audio_path: str):
        """Whole-file voiceprint (list[float]) for an enrollment clip, or None."""
        if self._embedding is None:
            return None
        try:
            # window="whole" -> a single embedding vector over the entire file.
            emb_array = self._embedding({"uri": audio_path, "audio": audio_path})
            return list(float(x) for x in emb_array.flatten())
        except Exception as exc:
            _log(f"diarize: whole-file embedding failed for {audio_path}: {exc}")
            return None

    def _label_segments_inner(
        self,
        wav_path: str,
        segments: list[dict],
        window_start: float,
    ) -> list[dict]:
        """Core implementation — may raise; wrapped by label_segments()."""
        if not self._ensure_loaded():
            return segments  # unavailable — return unchanged, no exception

        # --- 1. Run diarization on the window WAV ------------------------
        diar_result = self._pipeline(wav_path)

        # WhisperX returns a DataFrame with columns [start, end, speaker] or
        # an Annotation-like object — normalise to a list of dicts.
        turns: list[dict] = _extract_turns(diar_result)

        if not turns:
            return segments  # no speech detected in this window

        # --- 2. Collect unique local speaker labels ----------------------
        local_labels: list[str] = list(dict.fromkeys(t["speaker"] for t in turns))

        # --- 3. Map each local label -> global stable label --------------
        local_to_global: dict[str, str] = {}
        for local_label in local_labels:
            # Turns belonging to this local speaker (relative to window WAV)
            speaker_turns = [t for t in turns if t["speaker"] == local_label]

            # Compute voice embedding (or None if embedding model unavailable)
            emb = self._compute_embedding(wav_path, speaker_turns)

            if emb is not None and self._speakers:
                # Find best cosine match among known global speakers
                best_idx, best_sim = _best_match(emb, self._speakers)
                matched = self._speakers[best_idx]
                # Enrolled references match on a looser threshold (clean clip vs
                # noisy in-video turn); cross-window speakers use the strict one.
                thr = ENROLL_THRESHOLD if matched.get("enrolled") else SIMILARITY_THRESHOLD
                if best_sim >= thr:
                    global_label = matched["label"]
                    # Don't drift a locked enrolled reference; update others.
                    if not matched.get("locked"):
                        _update_centroid(matched, emb)
                    local_to_global[local_label] = global_label
                    continue

            # No match (or no embedding / no speakers yet) -> new global speaker.
            # Number only the AUTO speakers so enrolled names don't shift the
            # "Speaker N" sequence.
            n = sum(1 for s in self._speakers if not s.get("enrolled")) + 1
            global_label = f"Speaker {n}"
            entry: dict = {
                "label": global_label,
                "centroid": list(emb) if emb is not None else [],
                "count": 1,
            }
            self._speakers.append(entry)
            local_to_global[local_label] = global_label

        # --- 4. Convert turn times to absolute and build overlap index ---
        abs_turns: list[dict] = []
        for t in turns:
            abs_turns.append(
                {
                    "start": t["start"] + window_start,
                    "end": t["end"] + window_start,
                    "global": local_to_global[t["speaker"]],
                }
            )

        # --- 5. Assign each segment the label with max temporal overlap --
        enrolled_labels = {s["label"] for s in self._speakers if s.get("enrolled")}
        out: list[dict] = []
        for seg in segments:
            label = _assign_label(seg, abs_turns)
            new_seg = dict(seg)
            if label is not None:
                new_seg["speaker"] = label
                if label in enrolled_labels:
                    new_seg["enrolled"] = True
            out.append(new_seg)

        return out

    def _compute_embedding(
        self,
        wav_path: str,
        speaker_turns: list[dict],
    ):
        """Return a voice embedding (list[float]) for the speaker's turns, or None."""
        if self._embedding is None or not speaker_turns:
            return None

        try:
            # pyannote Inference on a cropped segment (use the longest turn for
            # best signal-to-noise; fall back to the first if equal length).
            longest = max(speaker_turns, key=lambda t: t["end"] - t["start"])
            crop_start = longest["start"]
            crop_end = longest["end"]

            # The embedding model needs a minimum amount of audio (its conv
            # kernel spans ~7 frames); a sub-second turn raises "Kernel size
            # can't be greater than input". Pad short crops up to ~1s; if even
            # the whole turn is tiny, skip (too little signal to ID anyway).
            MIN_DUR = 1.0
            if crop_end - crop_start < MIN_DUR:
                pad = (MIN_DUR - (crop_end - crop_start)) / 2.0
                crop_start = max(0.0, crop_start - pad)
                crop_end = crop_end + pad
            if crop_end - crop_start < 0.4:
                return None

            # For a window="whole" Inference, the per-excerpt embedding is taken
            # via .crop(file, Segment) — calling the model directly with an
            # `excerpt=` kwarg is NOT supported (raises TypeError) and silently
            # disabled cross-window tracking. .crop returns one vector per chunk.
            from pyannote.core import Segment

            emb_array = self._embedding.crop(
                {"uri": wav_path, "audio": wav_path},
                Segment(crop_start, crop_end),
            )

            # Result is a numpy array; convert to plain list for storage.
            import numpy as _np
            return list(float(x) for x in _np.asarray(emb_array).flatten())
        except Exception as exc:
            _log(f"diarize: embedding computation failed: {exc}")
            return None

    def reset(self) -> None:
        """Clear AUTO-detected speakers; keep enrolled reference voiceprints."""
        self._speakers = [s for s in self._speakers if s.get("enrolled")]


# ---------------------------------------------------------------------------
# Free helper functions (no access to self)
# ---------------------------------------------------------------------------

def _extract_turns(diar_result) -> list[dict]:
    """Normalise WhisperX/pyannote diarization output to list[{start,end,speaker}]."""
    turns: list[dict] = []

    # WhisperX returns a pandas DataFrame with columns [start, end, speaker]
    try:
        import pandas as pd

        if isinstance(diar_result, pd.DataFrame):
            for _, row in diar_result.iterrows():
                turns.append(
                    {
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "speaker": str(row["speaker"]),
                    }
                )
            return turns
    except ImportError:
        pass

    # pyannote Annotation object (iterable of (segment, _, label) triples)
    try:
        for segment, _, label in diar_result.itertracks(yield_label=True):
            turns.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(label),
                }
            )
        return turns
    except AttributeError:
        pass

    # Last resort: try iterating as list/tuple of (start, end, speaker)
    try:
        for item in diar_result:
            turns.append(
                {
                    "start": float(item[0]),
                    "end": float(item[1]),
                    "speaker": str(item[2]),
                }
            )
        return turns
    except (TypeError, IndexError, KeyError):
        pass

    return []


def _best_match(emb: list[float], speakers: list[dict]) -> tuple[int, float]:
    """Return (index, similarity) of the best-matching global speaker."""
    best_idx = 0
    best_sim = -1.0
    for i, sp in enumerate(speakers):
        if not sp["centroid"]:
            continue
        sim = _cosine_similarity(emb, sp["centroid"])
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_idx, best_sim


def _update_centroid(speaker_entry: dict, new_emb: list[float]) -> None:
    """Update the centroid with a running mean incorporating new_emb."""
    count = speaker_entry["count"]
    centroid = speaker_entry["centroid"]
    if not centroid:
        speaker_entry["centroid"] = list(new_emb)
        speaker_entry["count"] = 1
        return
    n = count
    # Running mean: new_centroid = (n * old + new) / (n + 1)
    speaker_entry["centroid"] = [
        (n * c + v) / (n + 1) for c, v in zip(centroid, new_emb)
    ]
    speaker_entry["count"] = n + 1


def _overlap_duration(seg_start: float, seg_end: float, turn_start: float, turn_end: float) -> float:
    """Temporal overlap in seconds between two intervals."""
    lo = max(seg_start, turn_start)
    hi = min(seg_end, turn_end)
    return max(0.0, hi - lo)


def _assign_label(seg: dict, abs_turns: list[dict]) -> Optional[str]:
    """Return the global speaker label with the most overlap with seg."""
    seg_start = float(seg.get("start", 0.0))
    seg_end = float(seg.get("end", seg_start))

    best_label: Optional[str] = None
    best_overlap: float = 0.0

    for turn in abs_turns:
        ov = _overlap_duration(seg_start, seg_end, turn["start"], turn["end"])
        if ov > best_overlap:
            best_overlap = ov
            best_label = turn["global"]

    return best_label  # None if no overlap at all
