"""Audio acquisition for the YouTube translator helper.

Two acquisition strategies live here:

1. WINDOWED / LOW-LATENCY (preferred, used by the live WS loop):
   - `get_audio_url(video_id)` resolves the direct (deciphered) audio stream URL
     via yt-dlp WITHOUT downloading anything.
   - `fetch_window(url, start, duration)` range-seeks into that remote stream with
     ffmpeg and decodes only the requested window to a 16kHz mono WAV. This lets
     transcription start almost immediately instead of waiting for a multi-minute
     full download on long VODs.

2. WHOLE-FILE (legacy fallback):
   - `download_audio(video_id)` downloads the whole bestaudio track, and
     `clip_from()` clips it from an offset. Kept for environments where the
     windowed path fails.

yt-dlp deciphers the throttling `n` parameter during extraction, so the resolved
URL is directly fetchable by ffmpeg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import yt_dlp


def _log(*parts) -> None:
    """Match server.py's lightweight stdout logging so timings are greppable."""
    print("[ytx]", *parts, flush=True)


def _temp_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "yt_translator_audio")
    os.makedirs(d, exist_ok=True)
    return d


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# --- Authentication (cookies) ----------------------------------------------
#
# Most videos resolve fine with no auth. Age-restricted videos (and, sometimes,
# YouTube's "confirm you're not a bot" checks) require the user's logged-in
# cookies. On Windows, yt-dlp cannot read Chrome's cookies directly while Chrome
# is open (the DB is locked and, since Chrome 127, App-Bound-Encrypted), so the
# reliable path is a Netscape `cookies.txt` exported by a browser extension.


def _cookie_opts() -> dict:
    """Return yt-dlp cookie options if the user configured authentication.

    Resolution order (first match wins):
      1. ``YTDLP_COOKIES_FILE`` env var  -> a Netscape cookies.txt path.
      2. ``cookies.txt`` sitting next to this module (helper/cookies.txt).
      3. ``YTDLP_COOKIES_FROM_BROWSER`` env var, e.g. "firefox", "edge",
         "chrome", or "chrome:Profile 1" -> read cookies straight from a browser
         (works for Firefox/closed-Chrome; usually fails for an open Chrome).

    Returns ``{}`` when nothing is configured, so normal videos still work.
    """
    path = os.environ.get("YTDLP_COOKIES_FILE")
    if path and os.path.exists(path):
        return {"cookiefile": path}

    default_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cookies.txt"
    )
    if os.path.exists(default_file):
        return {"cookiefile": default_file}

    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
    if browser:
        name, _, profile = browser.partition(":")
        return {"cookiesfrombrowser": (name.strip(), profile.strip() or None, None, None)}

    return {}


def cookies_source() -> Optional[str]:
    """Human-readable description of the active cookie source, or None."""
    opts = _cookie_opts()
    if "cookiefile" in opts:
        return f"file:{opts['cookiefile']}"
    if "cookiesfrombrowser" in opts:
        return f"browser:{opts['cookiesfrombrowser'][0]}"
    return None


# --- Windowed / low-latency path ------------------------------------------


def _do_extract(video_id: str, with_cookies: bool) -> dict:
    opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if with_cookies:
        opts.update(_cookie_opts())
    # NOTE: by default keep the PO token (do NOT pass fetch_pot=never) — without
    # it YouTube throttles the audio download to a crawl for SOME videos.
    # A/B switch: set YTX_NO_POT=1 to skip PO-token fetching and measure whether
    # it actually speeds up resolution for the video you're testing. If resolves
    # get faster AND windowed seeks still work, the token wasn't needed for it.
    no_pot = bool(os.environ.get("YTX_NO_POT"))
    if no_pot:
        opts["extractor_args"] = {"youtube": {"fetch_pot": ["never"]}}
    _log(f"_do_extract cookies={with_cookies} pot={'off' if no_pot else 'on'}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(_watch_url(video_id), download=False)


def extract_info(video_id: str) -> dict:
    """Resolve full yt-dlp info for `video_id` WITHOUT downloading.

    Prefer the COOKIE (authenticated) path when a cookie source exists. This is
    counter-intuitive for speed, but it is REQUIRED for correctness: the
    anonymous (no-cookie) audio URL is frequently served *sequential-only* —
    YouTube throttles ranged HTTP seeks on it to zero bytes, so any windowed
    `ffmpeg -ss` past the start of the stream hangs forever. The authenticated
    URL supports deep range-seeks (the whole basis of our low-latency windowed
    fetch). We fall back to the anonymous path only if the cookie path fails or
    no cookies are configured. (Empirically verified 2026-06-14: anonymous URL
    times out at every offset > 0; cookie URL seeks to 60s/600s in ~1.8s.)
    """
    if cookies_source() is not None:
        try:
            _t0 = time.time()
            info = _do_extract(video_id, with_cookies=True)
            _log(f"extract_info cookie-path OK in {time.time() - _t0:.2f}s")
            return info
        except Exception as exc:
            _log(f"extract_info cookie-path FAILED, falling back to anon: {str(exc)[:120]}")
            _t1 = time.time()
            info = _do_extract(video_id, with_cookies=False)
            _log(f"extract_info anon-fallback OK in {time.time() - _t1:.2f}s")
            return info
    _t2 = time.time()
    info = _do_extract(video_id, with_cookies=False)
    _log(f"extract_info anon-path OK in {time.time() - _t2:.2f}s")
    return info


def _best_audio_url(info: dict) -> str:
    """Pick the best fetchable audio URL from a yt-dlp info dict.

    Prefers the already-selected format's `url`. Otherwise scans `formats` for
    audio-only entries (acodec != none, vcodec == none) and picks the highest
    average bitrate.
    """
    url = info.get("url")
    if url:
        return url

    formats = info.get("formats") or []

    def _abr(f: dict) -> float:
        return f.get("abr") or f.get("tbr") or 0.0

    audio_only = [
        f
        for f in formats
        if f.get("acodec") not in (None, "none")
        and f.get("vcodec") in (None, "none")
        and f.get("url")
    ]
    if audio_only:
        best = max(audio_only, key=_abr)
        return best["url"]

    # Last resort: any format with a url and an audio codec.
    with_audio = [
        f for f in formats if f.get("acodec") not in (None, "none") and f.get("url")
    ]
    if with_audio:
        return max(with_audio, key=_abr)["url"]

    raise RuntimeError("no audio format with a resolvable URL found")


def get_audio_url(video_id: str) -> str:
    """Resolve the direct (deciphered) audio stream URL for `video_id`.

    No file is downloaded. The returned URL is short-lived (expires after a few
    hours), so callers should be ready to re-extract on fetch failure.
    """
    return _best_audio_url(extract_info(video_id))


def get_url_and_duration(video_id: str) -> tuple[str, Optional[float]]:
    """Resolve (audio_url, duration_seconds). duration may be None if unknown."""
    info = extract_info(video_id)
    duration = info.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    return _best_audio_url(info), duration


# Hard ceiling on how long a single window fetch may take. Generous on purpose:
# when YouTube throttles a stream, we'd rather wait up to this long and actually
# produce captions than give up. The timeout only exists to stop a truly dead
# connection from freezing the loop forever.
FETCH_TIMEOUT = 90.0


def fetch_window(
    url: str, start: float, duration: float, on_proc=None, timeout: float = FETCH_TIMEOUT
) -> str:
    """Decode the [start, start+duration] window of a remote audio stream to WAV.

    Uses ffmpeg to range-seek into the remote URL (`-ss` before `-i` so only the
    requested window is fetched/decoded) and produces a 16kHz mono WAV — the
    native input format faster-whisper prefers.

    `on_proc`, if given, is called with the live ffmpeg Popen so the caller can
    KILL it when the session goes away (otherwise a reload leaves the download
    running and hammering YouTube). `timeout` caps this single fetch — callers
    pass a short value at the playhead so a throttled (0-byte) stream surfaces
    quickly instead of hanging the full default. Raises on failure/timeout/kill.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (required for windowed fetch)")

    fd, wav_path = tempfile.mkstemp(
        prefix="ytw_", suffix=".wav", dir=_temp_dir()
    )
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", url,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        wav_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if on_proc is not None:
        try:
            on_proc(proc)
        except Exception:
            pass
    try:
        _out, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        cleanup(wav_path)
        raise RuntimeError(
            f"window fetch timed out after {timeout:.0f}s (throttled)"
        )

    if proc.returncode != 0:
        cleanup(wav_path)
        # Negative return code => killed (e.g. the session was torn down).
        detail = (stderr.decode("utf-8", "replace")[-400:] if stderr else "") or (
            f"exit {proc.returncode}"
        )
        raise RuntimeError(f"ffmpeg window fetch failed: {detail}")

    if not (os.path.exists(wav_path) and os.path.getsize(wav_path) > 0):
        cleanup(wav_path)
        raise RuntimeError("ffmpeg produced an empty window")

    return wav_path


# --- Whole-file fallback path ----------------------------------------------


def download_audio(video_id: str) -> str:
    """Download the bestaudio track for `video_id`. Returns the local path.

    yt-dlp picks the file extension based on the chosen format, so we let it
    fill in the extension via an %(ext)s template and read back the resolved
    path from the result.
    """
    out_template = os.path.join(_temp_dir(), f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Don't fetch the whole playlist if the id resolves to one.
        "noplaylist": True,
        "overwrites": True,
        **_cookie_opts(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(_watch_url(video_id), download=True)
        # prepare_filename reflects the actual outtmpl resolution.
        path = ydl.prepare_filename(info)

    if not os.path.exists(path):
        # Fallback: some postprocessors change the extension; find the file.
        base = os.path.join(_temp_dir(), video_id)
        for ext in ("webm", "m4a", "mp3", "opus", "ogg", "wav", "aac"):
            candidate = f"{base}.{ext}"
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"yt-dlp did not produce expected file: {path}")

    return path


def clip_from(audio_path: str, start_time: float) -> str:
    """Clip `audio_path` from `start_time` seconds forward using ffmpeg -ss.

    Returns the path to the clipped file. If `start_time` <= 0 or ffmpeg is
    unavailable, returns the original path unchanged (the caller adds the
    offset back to timestamps).
    """
    if start_time <= 0:
        return audio_path

    if shutil.which("ffmpeg") is None:
        # No ffmpeg: fall back to transcribing the whole file (offset = 0).
        return audio_path

    root, ext = os.path.splitext(audio_path)
    if not ext:
        ext = ".m4a"
    clipped = f"{root}.clip{ext}"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.3f}",
        "-i", audio_path,
        "-vn",
        "-c", "copy",
        clipped,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        # Stream copy can fail for some containers; retry with re-encode.
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time:.3f}",
                    "-i", audio_path,
                    "-vn",
                    clipped,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, OSError):
            return audio_path

    if os.path.exists(clipped) and os.path.getsize(clipped) > 0:
        return clipped
    return audio_path


def cleanup(*paths: Optional[str]) -> None:
    """Best-effort removal of temp audio files."""
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
