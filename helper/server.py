"""FastAPI server for the YouTube -> English subtitle helper.

Endpoints (shared contract with the Chrome extension):
    GET  /health     -> {"status","model_loaded","cuda","device","ollama"}
    GET  /models     -> {"models":[<ollama chat model names>]}
    WS   /transcribe -> streams status/segment/progress/done/error JSON messages

The WS loop transcribes in ROLLING WINDOWS that follow the client's playback
position (lead-following + pre-buffer) instead of downloading the whole VOD up
front, so subtitles start within a couple seconds even on long streams.

Run:
    python server.py      # uvicorn on 127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import audio
import cache
import denoise
from diarize import SpeakerTracker
import transcribe
import translate_llm

HOST = "127.0.0.1"
PORT = 8765

# Stay this many seconds AHEAD of the client's current playback position.
LEAD = 90.0
# Size of each fetch+transcribe window, in seconds.
WINDOW = 45.0
# Smaller window when fetching right AT the playback head, so the first caption
# appears fast even when YouTube throttles the stream (age-restricted videos).
# Larger WINDOW is used once we're comfortably pre-buffering ahead.
FIRST_WINDOW = 12.0

def _log(*parts) -> None:
    """Lightweight stdout logging so the rolling-window loop isn't a black box."""
    print("[ytx]", *parts, flush=True)


app = FastAPI(title="YouTube Translator Helper")

# Localhost-only tool; permissive CORS so the chrome-extension origin connects.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Background model load -------------------------------------------------

_load_lock = threading.Lock()
_load_started = False
_load_error: str | None = None


def _load_model_blocking() -> None:
    global _load_error
    try:
        transcribe.load_model()
    except Exception as exc:  # surfaced to clients via the WS error message
        _load_error = str(exc)


def _start_model_load() -> None:
    global _load_started
    with _load_lock:
        if _load_started:
            return
        _load_started = True
    threading.Thread(target=_load_model_blocking, daemon=True).start()


@app.on_event("startup")
async def _on_startup() -> None:
    # Kick off the (slow) model load in the background; /health reflects state.
    _start_model_load()


@app.get("/health")
async def health() -> dict:
    # Ollama liveness is a quick network probe -> run off the event loop.
    ollama_up = await asyncio.to_thread(translate_llm.is_up)
    return {
        "status": "ok",
        "model_loaded": transcribe.is_loaded(),
        "whisper_model": transcribe.get_model_name(),
        "vram_gb": round(transcribe._detect_vram_gb(), 1),
        "cuda": transcribe.is_cuda(),
        "device": transcribe.get_device(),
        "ollama": ollama_up,
        "cookies": audio.cookies_source() is not None,
    }


@app.get("/models")
async def models() -> dict:
    chat_models = await asyncio.to_thread(translate_llm.list_chat_models)
    return {"models": chat_models}


@app.post("/reset")
async def reset(payload: dict) -> dict:
    """Clear ALL cached transcripts for a video so it re-transcribes fresh.

    Powers the extension's "Re-translate this video" button — useful when a
    cache is wrong/partial (e.g. poisoned by a session that started mid-video).
    """
    video_id = (payload or {}).get("videoId")
    if not video_id:
        return {"cleared": 0}
    n = await asyncio.to_thread(cache.clear, video_id)
    _log(f"reset cache video={video_id} cleared={n}")
    return {"cleared": n}


# --- Shared session state --------------------------------------------------


@dataclass
class Session:
    """State shared between the transcription loop and the position reader."""

    current_time: float = 0.0
    # Set True once the client has sent at least one position update.
    saw_position: bool = False
    closed: bool = False
    # Live ffmpeg fetch process, so we can kill it the instant the client goes
    # away (a reload otherwise leaves the download running and hammering YouTube).
    current_proc: object = None
    # Rolling (source, english) context fed to the LLM for continuity.
    llm_context: list = field(default_factory=list)
    # Lazily-created speaker diarization tracker (one per session) — keeps
    # stable global "Speaker N" labels across windows. None until first use.
    speaker_tracker: object = None


def _kill_proc(proc) -> None:
    """Best-effort kill of an in-flight ffmpeg fetch."""
    try:
        if proc is not None and proc.poll() is None:
            proc.kill()
    except Exception:
        pass


async def _position_reader(ws: WebSocket, sess: Session) -> None:
    """Continuously read client messages and update shared state.

    Runs as its own asyncio task so the transcription loop never blocks on the
    socket. Tolerates the socket closing (sets sess.closed and returns).
    """
    try:
        while True:
            msg = await ws.receive_json()
            if isinstance(msg, dict) and msg.get("type") == "position":
                ct = msg.get("currentTime")
                if isinstance(ct, (int, float)):
                    sess.current_time = float(ct)
                    sess.saw_position = True
    except (WebSocketDisconnect, RuntimeError, ValueError, KeyError):
        # Socket closed or sent something undecodable -> stop following AND kill
        # any in-flight fetch so a reload doesn't leave a download running.
        sess.closed = True
        _kill_proc(sess.current_proc)


async def _wait_for_model(ws: WebSocket) -> bool:
    """Ensure the model is loaded, streaming a status while we wait.

    Returns True if the model is ready, False if loading failed (an error
    message has already been sent to the client).
    """
    _start_model_load()

    if not transcribe.is_loaded():
        await ws.send_json({"type": "status", "message": "loading model"})

    while not transcribe.is_loaded():
        if _load_error is not None:
            await ws.send_json(
                {"type": "error", "message": f"model load failed: {_load_error}"}
            )
            return False
        await asyncio.sleep(0.5)

    return True


# --- Transcription helpers (run off the event loop) ------------------------


def _drain(gen) -> list:
    """Fully consume a generator in a worker thread, returning a list.

    Each window is short (<= WINDOW seconds), so draining the whole window's
    segments in one thread hop is fine and far simpler than stepping the
    generator one item at a time across the executor boundary.
    """
    return list(gen)


def _fetch_window_with_retry(
    video_id: str, url_box: list, start: float, duration: float, on_proc=None,
    timeout: float = audio.FETCH_TIMEOUT,
) -> str:
    """Fetch a window, re-extracting the (possibly expired) URL once on failure.

    `url_box` is a single-element list holding the current audio URL so we can
    update it in place when we re-extract. `on_proc` is forwarded so the caller
    can kill the live ffmpeg process on disconnect. `timeout` caps each attempt.
    """
    try:
        return audio.fetch_window(url_box[0], start, duration, on_proc=on_proc, timeout=timeout)
    except Exception:
        # URLs expire after a few hours — re-extract and retry once.
        url_box[0] = audio.get_audio_url(video_id)
        return audio.fetch_window(url_box[0], start, duration, on_proc=on_proc, timeout=timeout)


def _next_uncovered(covered: list, t: float, duration: float | None) -> float | None:
    """First uncovered absolute time at/after `t`, or None if covered to the end.

    If `t` sits inside a covered interval, the next thing to transcribe is that
    interval's end (a genuine gap, since stored intervals are non-adjacent).
    """
    if duration is not None and t >= duration:
        return None
    end = cache.covered_end_at(covered, t)
    if end is None:
        return t
    if duration is not None and end >= duration:
        return None
    return end


# --- Transcription WebSocket ----------------------------------------------


@app.websocket("/transcribe")
async def transcribe_ws(ws: WebSocket) -> None:
    await ws.accept()

    sess = Session()
    reader_task: asyncio.Task | None = None

    try:
        # --- Init message --------------------------------------------------
        req = await ws.receive_json()
        video_id = req.get("videoId")
        start_time = float(req.get("startTime") or 0.0)
        language = req.get("language")  # None => auto-detect
        engine = (req.get("engine") or "whisper").lower()
        model = req.get("model") or translate_llm.DEFAULT_MODEL
        pre_buffer = bool(req.get("preBuffer"))
        hotwords = req.get("hotwords")
        quality = req.get("quality") or "auto"
        clean_audio = req.get("cleanAudio") or "off"
        diarize = bool(req.get("diarize"))
        glossary = req.get("glossary") or []  # [{term, preferred}, ...]

        if not video_id:
            await ws.send_json({"type": "error", "message": "missing videoId"})
            return

        if engine not in ("ollama", "whisper"):
            engine = "whisper"

        # task is fixed by engine: whisper translates directly; ollama keeps the
        # faithful source transcript and translates it with the LLM.
        task = "translate" if engine == "whisper" else "transcribe"

        # Resolve the Whisper model tier from the Quality setting (VRAM-adaptive
        # for "auto"). This feeds both the cache variant and the model load.
        whisper_model = transcribe.resolve_model(quality)

        # Cache variant captures output-affecting dims so different quality/clean/diarize
        # combos get separate cache files.
        variant = f"{whisper_model}|{clean_audio}|{'diar' if diarize else 'mono'}"

        _log(
            f"session video={video_id} start={start_time:.1f} engine={engine} "
            f"preBuffer={pre_buffer} quality={quality} whisper_model={whisper_model} "
            f"clean={clean_audio} diarize={diarize} cookies={audio.cookies_source()}"
        )

        sess.current_time = start_time

        # Start following the client's playback position immediately.
        reader_task = asyncio.create_task(_position_reader(ws, sess))

        # --- Serve cached segments + load interval coverage -----------------
        cached = cache.load(video_id, language, task, engine, model, variant=variant)
        covered: list = []

        if cached is not None:
            covered = cached["covered"]
            await ws.send_json({"type": "status", "message": "cached"})
            # Tell the client which regions are already transcribed (incl. silent
            # ones) so its playback gate doesn't hold on an already-done spot.
            for iv in covered:
                await ws.send_json(
                    {"type": "progress", "start": iv[0], "until": iv[1]}
                )
            for seg in cache.filter_from(cached["segments"], start_time):
                await ws.send_json(
                    {
                        "type": "segment",
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    }
                )

        # --- Make sure the Whisper model is ready --------------------------
        if not await _wait_for_model(ws):
            return

        # The background load picks the auto tier; if this session needs a
        # different model (explicit Quality override, or auto on a machine the
        # background load mis-sized), switch it now. With WhisperX the `task` is
        # baked into the model, so we ALSO reload when the loaded task differs
        # from this session's task. Reloads are rare.
        if (
            transcribe.get_model_name() != whisper_model
            or transcribe.get_model_task() != task
        ):
            await ws.send_json({"type": "status", "message": "switching model"})
            try:
                await asyncio.to_thread(transcribe.load_model, whisper_model, task)
            except Exception as exc:
                await ws.send_json(
                    {"type": "error", "message": f"model load failed: {exc}"}
                )
                return

        # --- Resolve audio URL + duration (no download) --------------------
        await ws.send_json({"type": "status", "message": "resolving stream"})
        _t_resolve = time.time()
        try:
            url, duration = await asyncio.to_thread(
                audio.get_url_and_duration, video_id
            )
            _log(f"resolve total={time.time() - _t_resolve:.2f}s video={video_id}")
        except Exception as exc:
            emsg = str(exc)
            low = emsg.lower()
            if "sign in to confirm your age" in low or "inappropriate" in low:
                hint = (
                    "🔒 Age-restricted video needs your YouTube login. "
                    "Add a cookies.txt (see how-to.html). "
                )
            elif "not a bot" in low or "sign in to confirm you" in low:
                hint = (
                    "🔒 YouTube wants a login to confirm you're not a bot. "
                    "Add a cookies.txt (see how-to.html). "
                )
            elif "cookie" in low:
                hint = "🍪 Cookie problem — re-export cookies.txt. "
            else:
                hint = ""
            _log(f"resolve FAILED video={video_id} cookies={audio.cookies_source()}: {emsg[:300]}")
            await ws.send_json(
                {"type": "error", "message": f"{hint}could not resolve audio: {emsg[:200]}"}
            )
            return
        url_box = [url]

        await ws.send_json({"type": "status", "message": "transcribing"})

        # Effective engine may downgrade to whisper if Ollama dies mid-session.
        eff_engine = engine
        # Whether we've already warned the client about the Ollama fallback.
        warned_ollama = False
        # Whether we've already warned the client that diarization is unavailable
        # (no HF token / missing libs) — sent at most once per session.
        warned_diarize = False

        # eff_task tracks the task actually written to cache (whisper fallback
        # writes English under "translate", not the source transcript).
        eff_task = "translate" if eff_engine == "whisper" else "transcribe"
        fetch_fails = 0  # consecutive throttled/failed fetches

        # --- Position-driven coverage loop ---------------------------------
        # Keep the region around the playback head covered. We always look for
        # the nearest UNCOVERED point at/after where the user currently is, fill
        # it, and idle when the playhead's surroundings are already done. This
        # handles forward AND backward seeks (jump back into an un-transcribed
        # gap -> we fill it) and never falsely treats the start as "done". The
        # socket stays open for the life of the video so later seeks get filled.
        while not sess.closed:
            # If pre-buffer is off, wait for the client to actually start playing.
            if not pre_buffer and not sess.saw_position:
                await asyncio.sleep(0.3)
                continue

            target = sess.current_time
            cursor = _next_uncovered(covered, target, duration)

            # Nothing to do: covered from the playhead out to LEAD (or past end).
            if cursor is None or cursor > target + LEAD:
                await asyncio.sleep(0.4)
                continue

            # Size the window: small right at the playhead (fast first caption,
            # even on throttled age-restricted streams), full WINDOW once we're
            # buffering ahead. Also never fill into the next covered region/EOF.
            at_playhead = cursor <= sess.current_time + 5.0
            window_dur = FIRST_WINDOW if at_playhead else WINDOW
            if duration is not None:
                window_dur = min(window_dur, duration - cursor)
            ncs = cache.next_covered_start_after(covered, cursor)
            if ncs is not None:
                window_dur = min(window_dur, ncs - cursor)
            if window_dur <= 0.05:
                covered = cache.merge_intervals(covered + [[cursor, cursor + 0.1]])
                continue
            window_end = cursor + window_dur

            # Fetch the window (re-extract URL + retry once on failure). At the
            # playhead use a SHORT timeout so a throttled (0-byte) stream is
            # reported within seconds instead of hanging the full default; when
            # buffering ahead we can afford to wait the generous default.
            fetch_timeout = 25.0 if at_playhead else audio.FETCH_TIMEOUT
            _t0 = time.time()
            try:
                wav_path = await asyncio.to_thread(
                    _fetch_window_with_retry, video_id, url_box, cursor, window_dur,
                    lambda p: setattr(sess, "current_proc", p),
                    fetch_timeout,
                )
            except Exception as exc:
                emsg = str(exc)
                # Unknown-length stream that ran past its end -> finished.
                if duration is None and "empty" in emsg.lower():
                    break
                fetch_fails += 1
                _log(f"window fetch failed @ {cursor:.1f} (#{fetch_fails}): {emsg[:90]}")
                if fetch_fails >= 6:
                    await ws.send_json({
                        "type": "error",
                        "message": "YouTube is heavily throttling the audio download "
                        "(likely IP rate-limited). Try again later or via a VPN / "
                        "different network.",
                    })
                    break
                # Throttled / hung fetch: notify, back off, and retry. Because the
                # loop re-targets the current playhead each pass, it chases where
                # you actually are instead of freezing on one spot.
                await ws.send_json({
                    "type": "status",
                    "message": "⏳ YouTube is throttling the download — buffering…",
                })
                await asyncio.sleep(2.0)
                continue
            _t_fetch = time.time() - _t0
            fetch_fails = 0

            # Optional noise/music suppression BEFORE transcription. denoise is
            # lazy + graceful: it returns wav_path unchanged on off/missing-lib/
            # failure, and can be heavy -> run on a worker thread.
            clean_wav = await asyncio.to_thread(denoise.clean, wav_path, clean_audio)

            try:
                if eff_engine == "whisper":
                    # Fast beam at the playhead (latency is felt here); full beam
                    # for pre-buffered windows you haven't reached yet (quality).
                    beam = 1 if at_playhead else 5
                    # transcribe() now yields (s, e, t, detected_lang) and does
                    # NOT force the language (so we can filter by what's actually
                    # spoken). detected is used by the language filter below.
                    segs = await asyncio.to_thread(
                        _drain,
                        transcribe.transcribe(
                            clean_wav, time_offset=cursor,
                            hotwords=hotwords, beam_size=beam,
                            model_name=whisper_model,
                        ),
                    )
                else:
                    # language=None -> auto-detect so the filter below can work.
                    segs = await asyncio.to_thread(
                        _drain,
                        transcribe.transcribe_source(
                            clean_wav, language=None, time_offset=cursor,
                            hotwords=hotwords, model_name=whisper_model,
                        ),
                    )

                # Language filter: when the user picked a specific spoken language
                # (not "auto"), only keep windows actually DETECTED as that
                # language — e.g. subtitle the Czech speaker but show nothing while
                # the (English) game audio plays. segs carry the detected language
                # as the 4th tuple element; detection is per-window.
                if language and segs:
                    detected = segs[0][3]
                    if detected and detected != language:
                        _log(
                            f"window [{cursor:.1f},{window_end:.1f}] "
                            f"detected={detected} != filter={language} -> skip"
                        )
                        segs = []

                # Optional speaker diarization: tag each segment with a stable
                # global "Speaker N" label. Must run INSIDE this try (before the
                # finally below deletes clean_wav) because it needs the audio
                # file. Heavy -> run on a worker thread. speakers[i] aligns with
                # segs[i]; on any failure / missing token/libs it stays None.
                speakers = None
                if diarize and segs:
                    if sess.speaker_tracker is None:
                        sess.speaker_tracker = SpeakerTracker(
                            device=transcribe.get_device()
                        )
                        if not sess.speaker_tracker.available() and not warned_diarize:
                            warned_diarize = True
                            await ws.send_json({
                                "type": "status",
                                "message": "diarization unavailable — add a "
                                "HuggingFace token (see how-to.html)",
                            })
                    seg_dicts = [{"start": s, "end": e} for (s, e, t, sl) in segs]
                    labeled = await asyncio.to_thread(
                        sess.speaker_tracker.label_segments, clean_wav, seg_dicts, cursor
                    )
                    speakers = [d.get("speaker") for d in labeled]
            finally:
                # Remove the fetched window WAV; also remove the denoised file,
                # but only when it's a distinct temp file (denoise returns the
                # original path when it didn't process -> don't double-clean).
                audio.cleanup(wav_path)
                if clean_wav != wav_path:
                    audio.cleanup(clean_wav)

            _log(
                f"window [{cursor:.1f},{window_end:.1f}] engine={eff_engine} "
                f"fetch={_t_fetch:.2f}s transcribe={time.time() - _t0 - _t_fetch:.2f}s "
                f"segs={len(segs)}"
            )

            # Stream (and, for ollama, translate) the window's segments.
            produced: list[dict] = []
            redo = False
            for i, (start, end, text, src_lang) in enumerate(segs):
                out_text = text
                if eff_engine == "ollama":
                    try:
                        translated = await asyncio.to_thread(
                            translate_llm.translate,
                            text, src_lang, model, list(sess.llm_context),
                            glossary=glossary,
                        )
                        if translated:
                            out_text = translated
                            sess.llm_context.append((text, translated))
                            sess.llm_context[:] = sess.llm_context[-2:]
                    except Exception:
                        # Ollama died: warn once, fall back to Whisper translate,
                        # and redo this window so nothing leaks out untranslated.
                        if not warned_ollama:
                            warned_ollama = True
                            await ws.send_json({
                                "type": "status",
                                "message": "Ollama unavailable — using Whisper translate",
                            })
                        eff_engine = "whisper"
                        eff_task = "translate"
                        redo = True
                        break
                seg = {"start": start, "end": end, "text": out_text}
                sp = speakers[i] if speakers else None
                if sp:
                    seg["speaker"] = sp
                produced.append(seg)
                await ws.send_json({"type": "segment", **seg})

            if redo:
                sess.llm_context.clear()
                continue

            # Mark the WHOLE window covered (even if silent) so we never reprocess
            # it, and persist incrementally.
            covered = cache.merge_intervals(covered + [[cursor, window_end]])
            # `start` lets the client rebuild coverage intervals for its gate.
            await ws.send_json(
                {"type": "progress", "start": cursor, "until": window_end}
            )
            await asyncio.to_thread(
                cache.append,
                video_id, produced, [cursor, window_end],
                language, eff_task, eff_engine, model,
                variant,
            )

        await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        sess.closed = True
        _kill_proc(sess.current_proc)  # don't leave a download running
        if reader_task is not None:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
