"""Disk cache for transcripts (full OR partial), with interval coverage.

Cached as JSON under cache/, keyed by videoId + language + task + engine + model.
Each file stores the segment list plus `covered` — a list of [start, end]
absolute-time intervals that have actually been transcribed. Tracking real
intervals (instead of a single "covered_until") means a session that starts in
the MIDDLE of a video (YouTube resumed at 0:37, or the user seeked) does NOT
falsely claim the beginning is done — the [0, 0:37] gap stays uncovered and gets
filled when the user goes there.

The server uses `covered` to:
  - serve cached segments for already-covered regions instantly,
  - skip re-transcribing covered regions, and
  - fill uncovered gaps around the playback head.

Writes are atomic (write tmp + os.replace). Appends merge new segments and
union the newly-covered interval into `covered`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

Interval = List[float]  # [start, end]


def _ensure_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _key(
    video_id: str,
    language: Optional[str],
    task: str,
    engine: str,
    model: Optional[str],
    variant: str = "",
) -> str:
    lang = language or "auto"
    mdl = (model or "default").replace(os.sep, "_").replace(":", "-")
    vr = (variant or "none").replace(os.sep, "_").replace(":", "-").replace("|", "-")
    return f"{video_id}__{lang}__{task}__{engine}__{mdl}__{vr}"


def _path(
    video_id: str,
    language: Optional[str],
    task: str,
    engine: str,
    model: Optional[str],
    variant: str = "",
) -> str:
    return os.path.join(
        CACHE_DIR, _key(video_id, language, task, engine, model, variant) + ".json"
    )


# --- Interval helpers ------------------------------------------------------


def merge_intervals(intervals: List[Interval], eps: float = 0.25) -> List[Interval]:
    """Return sorted, merged (non-overlapping) intervals. Adjacent within eps merge."""
    ivs = sorted([list(iv) for iv in intervals if iv and iv[1] > iv[0]])
    if not ivs:
        return []
    out = [ivs[0]]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered_end_at(covered: List[Interval], t: float) -> Optional[float]:
    """If `t` falls inside a covered interval, return that interval's end, else None."""
    for s, e in covered:
        if s - 0.25 <= t < e:
            return e
    return None


def next_covered_start_after(covered: List[Interval], t: float) -> Optional[float]:
    """Smallest covered-interval start strictly greater than `t`, or None."""
    starts = [s for s, e in covered if s > t]
    return min(starts) if starts else None


# --- Load / save -----------------------------------------------------------


def load(
    video_id: str,
    language: Optional[str] = None,
    task: str = "translate",
    engine: str = "whisper",
    model: Optional[str] = None,
    variant: str = "",
) -> Optional[Dict]:
    """Return the cached record, or None if not cached / not in the current format.

    Record shape:
        {"segments": [{"start","end","text"}, ...], "covered": [[s,e], ...]}

    Files in the legacy "covered_until"-only format are intentionally treated as
    a cache MISS (return None) so they get re-transcribed with correct interval
    coverage — this auto-heals old/poisoned caches.
    """
    path = _path(video_id, language, task, engine, model, variant)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None

    segments = data.get("segments")
    covered = data.get("covered")
    if not isinstance(segments, list) or not isinstance(covered, list):
        return None  # legacy/unknown format -> miss, re-transcribe cleanly

    norm = [[float(iv[0]), float(iv[1])] for iv in covered if isinstance(iv, list) and len(iv) == 2]
    return {"segments": segments, "covered": merge_intervals(norm)}


def save(
    video_id: str,
    segments: List[Dict],
    covered: List[Interval],
    language: Optional[str] = None,
    task: str = "translate",
    engine: str = "whisper",
    model: Optional[str] = None,
    variant: str = "",
) -> None:
    """Persist (overwrite) the segment list and covered intervals atomically."""
    _ensure_dir()
    path = _path(video_id, language, task, engine, model, variant)

    payload = {
        "videoId": video_id,
        "language": language,
        "task": task,
        "engine": engine,
        "model": model,
        "covered": merge_intervals(covered),
        "segments": segments,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def append(
    video_id: str,
    new_segments: List[Dict],
    covered_interval: Interval,
    language: Optional[str] = None,
    task: str = "translate",
    engine: str = "whisper",
    model: Optional[str] = None,
    variant: str = "",
) -> None:
    """Merge `new_segments` + union `covered_interval` into the existing cache.

    Segments already represented (same rounded start) are dropped to avoid
    duplicates; the result is kept sorted by start. `covered` unions the new
    interval. Safe to call repeatedly as windows complete.
    """
    existing = load(video_id, language, task, engine, model, variant)
    if existing is None:
        merged = list(new_segments)
        covered = [list(covered_interval)]
    else:
        merged = list(existing["segments"])
        seen = {round(s.get("start", 0.0), 2) for s in merged}
        for s in new_segments:
            if round(s.get("start", 0.0), 2) not in seen:
                merged.append(s)
                seen.add(round(s.get("start", 0.0), 2))
        covered = existing["covered"] + [list(covered_interval)]

    merged.sort(key=lambda s: s.get("start", 0.0))
    save(
        video_id,
        merged,
        merge_intervals(covered),
        language=language,
        task=task,
        engine=engine,
        model=model,
        variant=variant,
    )


def clear(video_id: str) -> int:
    """Delete ALL cached variants (every language/task/engine/model) for a video.

    Returns the number of files removed. Powers the extension's
    "Re-translate this video" button.
    """
    _ensure_dir()
    removed = 0
    prefix = f"{video_id}__"
    for name in os.listdir(CACHE_DIR):
        if name.startswith(prefix) and name.endswith(".json"):
            try:
                os.remove(os.path.join(CACHE_DIR, name))
                removed += 1
            except OSError:
                pass
    return removed


def filter_from(segments: List[Dict], start_time: float) -> List[Dict]:
    """Return segments whose end is at/after start_time (current position first)."""
    if start_time <= 0:
        return segments
    return [s for s in segments if s.get("end", 0.0) >= start_time]
