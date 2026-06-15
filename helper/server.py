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
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import audio
import cache
import denoise
import diarize as diarize_mod
from diarize import SpeakerTracker
import transcribe
import translate_llm

HOST = "127.0.0.1"
PORT = 8765

# Build identifier. Printed loudly at startup and returned by /health so you can
# confirm the RUNNING process is current — restarting the helper is the #1 cause
# of "I fixed it but nothing changed" (stale code keeps serving). Bump in lockstep
# with extension/manifest.json when shipping behaviour changes.
HELPER_VERSION = "1.10.6"

# Cache/runtime behavior version. Bumping this bypasses old poisoned caches
# without deleting them.
PIPELINE_CACHE_VERSION = "accurate-cs-audio-v9"

# Language filtering is intentionally tolerant for Agraelus/Czech streams:
# only drop high-confidence non-target speech and keep uncertain segments.
LANGUAGE_DROP_CONFIDENCE = 0.80

def _parse_static_glossary(text: str) -> list[dict]:
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        term, preferred = line.split("=", 1)
        entries.append({"term": term.strip(), "preferred": preferred.strip()})
    return entries


SERVER_GLOSSARY_HINTS = _parse_static_glossary("""blbec = idiot, moron
idiot = idiot
debil = moron, dumbass
magor = psycho, nutcase
kretén = cretin, idiot
vůl = idiot, dumb ox
pitomec = fool, idiot
trouba = dummy, fool
dement = moron, idiot
retard = retard

kokot = asshole, dickhead
píča = cunt, dumb bitch
čurák = dickhead, prick
zmrd = scumbag, motherfucker
mrdka = piece of shit
sráč = asshole, coward
hajzl = asshole, bastard
kunda = cunt
buzna = faggot
šulina = dickhead, little dick

do prdele = for fuck's sake, damn it
kurva = fuck
kurva fix = fucking hell
ty vole = dude, bro, holy shit
doprdele práce = for fuck's sake
ježiši kriste = Jesus Christ
sakra = damn
do hajzlu = go to hell, fuck this
no ty píčo = holy fuck
kurva drát = fucking hell

co to je za kokotinu = what is this bullshit
to si děláš prdel = are you kidding me
běž do prdele = fuck off
ses posral ne = are you out of your mind
to je úplně v píči = this is completely fucked
tak tohle je mrdka = this is garbage
co je to za bullshit = what is this bullshit
ty vole neee = dude noooo
kurvaaa = fuuuck

cringe = cringe
npc = npc
brainrot = brainrot
autista = autist (insult)
lobotom = lobotomite, brain-dead person
schizo = schizo
opice = monkey, ape
klaun = clown
klauníček = little clown
copium = copium
cope = cope

kkt = asshole, dickhead
p*ča = cunt
pica = cunt
pyčo = fuck, holy shit
pyčo vole = holy fuck dude
čůrák = dickhead
curak = dickhead
kokutek = little dickhead
kokůtek = little dickhead
krva = fuck
kruci = darn, dang
doprkna = damn it

kámo = bro, mate
brácho = brother, bro
no tak = come on
počkej = wait
ježišmarja = oh my god
no do píči = holy fuck
cože = what
jak jako = what do you mean
to nemyslíš vážně = you can't be serious
do píči = holy fuck, fuck
do pici = holy fuck, fuck
chápu = I understand, I get it
ch?pu = I understand, I get it
vyhul = suck it
vykuř mi ho = suck my dick
vykur mi ho = suck my dick
fakt mi ho vykuř = seriously suck my dick
fakt mi ho vykur = seriously suck my dick
fakt mi ho vykuš = seriously suck my dick
fakt mi ho vykus = seriously suck my dick
fuck mi ho = suck my dick
vykuš mi ho = suck my dick
vykus mi ho = suck my dick
vykuš ty piču = suck it, you bitch
vykuš ty píču = suck it, you bitch
vykus ty picu = suck it, you bitch
dej mi ho = give it to me
dej mi ho dej mi ho = give it to me, give it to me
pochcal = pissed myself
nepochcal = didn't piss myself
pochcat = piss myself
oklepávám = shake my dick off after peeing
Jsou prostě Bíčí varlata = They're just bull's balls
Bíčí varlata = bull's balls
býčí varlata = bull's balls
varlata = balls, testicles
koule = balls
Můžu ty dveře zavřít hubu = Can those doors shut up
zavřít hubu = shut up
zavři hubu = shut up
drž hubu = shut up
sas = suspicious
čum = look
sus = suspicious
pískání = whistling
kecáš = you are joking
časák = magazine""")

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
        "version": HELPER_VERSION,
        "model_loaded": transcribe.is_loaded(),
        "whisper_model": transcribe.get_model_name(),
        "vram_gb": round(transcribe._detect_vram_gb(), 1),
        "cuda": transcribe.is_cuda(),
        "device": transcribe.get_device(),
        "ollama": ollama_up,
        "cookies": audio.cookies_source() is not None,
        "enrolled": diarize_mod.enrolled_names(),
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


def _short_hash(value) -> str:
    """Stable short hash for cache-profile dimensions."""
    if value in (None, "", [], {}):
        return "none"
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _with_server_glossary_hints(glossary: list | None) -> list:
    """Append critical built-in hints without overwriting user choices."""
    out = []
    seen = set()
    for entry in glossary or []:
        term = (entry.get("term") if isinstance(entry, dict) else "") or ""
        key = term.strip().lower()
        if not key:
            continue
        out.append(entry)
        seen.add(key)
    for entry in SERVER_GLOSSARY_HINTS:
        key = entry["term"].lower()
        if key not in seen:
            out.append(entry)
            seen.add(key)
    return out


def _cache_model_for(engine: str, model: str | None) -> str:
    """Only Ollama's selected chat model affects output."""
    return model if engine == "ollama" else "whisper"


def _cache_variant(
    whisper_model: str,
    clean_audio: str,
    diarize: bool,
    enrolled_only: bool,
    language: str | None,
    hotwords: str | None,
    glossary: list | None,
    source_profile: str = "asr",
) -> str:
    return "|".join(
        [
            PIPELINE_CACHE_VERSION,
            whisper_model,
            f"source-{source_profile}",
            clean_audio,
            "diar" if diarize else "mono",
            "eo" if enrolled_only else "all",
            f"lang-{language or 'auto'}",
            f"hot-{_short_hash(hotwords)}",
            f"gloss-{_short_hash(glossary)}",
        ]
    )


def _filter_segments_by_language(segs: list, language: str | None) -> tuple[list, bool, int]:
    """Drop only high-confidence non-target segments.

    Returns (kept_segments, filtered_empty, dropped_count). `filtered_empty`
    identifies windows that should not be persisted as permanently covered.
    """
    if not language or not segs:
        return segs, False, 0

    kept = []
    dropped = 0
    for seg in segs:
        lang = seg[3] if len(seg) > 3 else None
        try:
            confidence = float(seg[4]) if len(seg) > 4 else 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        if (
            lang
            and lang != "unknown"
            and lang != language
            and confidence >= LANGUAGE_DROP_CONFIDENCE
        ):
            dropped += 1
            continue
        kept.append(seg)

    return kept, bool(dropped and not kept), dropped


def _should_persist_window(
    produced: list[dict],
    filtered_empty: bool,
    eo_filter_failed: bool,
    window_had_signal: bool,
) -> bool:
    if eo_filter_failed or filtered_empty:
        return False
    # Audible windows with no captions are exactly the class of cache poison we
    # want to avoid. Keep them covered for this session, but let a future run
    # retry with better position/model/settings.
    if not produced and window_had_signal:
        return False
    return True


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
        engine = (req.get("engine") or "ollama").lower()
        model = req.get("model") or translate_llm.DEFAULT_MODEL
        pre_buffer = bool(req.get("preBuffer"))
        client_hotwords = req.get("hotwords")
        quality = req.get("quality") or "auto"
        clean_audio = req.get("cleanAudio") or "off"
        diarize = bool(req.get("diarize"))
        enrolled_only = bool(req.get("enrolledOnly"))
        # Enrolled-only needs speaker tags to filter on, so force diarization on.
        diarize = diarize or enrolled_only
        glossary = _with_server_glossary_hints(
            req.get("glossary") or []
        )  # [{term, preferred}, ...]

        if not video_id:
            await ws.send_json({"type": "error", "message": "missing videoId"})
            return

        if engine not in ("ollama", "whisper"):
            engine = "ollama"

        # task is fixed by engine: whisper translates directly; ollama keeps the
        # faithful source transcript and translates it with the LLM.
        task = "translate" if engine == "whisper" else "transcribe"
        # Accurate source-first mode must not feed slang/profanity glossary terms
        # into Whisper. In testing, hotwords corrupted Czech ASR ("ch?pu", etc.).
        asr_hotwords = client_hotwords if engine == "whisper" else None

        # Resolve the Whisper model tier from the Quality setting (VRAM-adaptive
        # for "auto"). This feeds both the cache variant and the model load.
        whisper_model = transcribe.resolve_model(quality)

        sess.current_time = start_time

        # Start following the client's playback position immediately.
        reader_task = asyncio.create_task(_position_reader(ws, sess))

        source_profile = "asr"

        # Cache variant captures output-affecting dims and a pipeline version so
        # older poisoned empty/Ollama caches are bypassed automatically.
        variant = _cache_variant(
            whisper_model,
            clean_audio,
            diarize,
            enrolled_only,
            language,
            asr_hotwords,
            glossary if engine == "ollama" else None,
            source_profile,
        )
        cache_model = _cache_model_for(engine, model)

        _log(
            f"session video={video_id} start={start_time:.1f} engine={engine} "
            f"source={source_profile} preBuffer={pre_buffer} quality={quality} "
            f"whisper_model={whisper_model} clean={clean_audio} diarize={diarize} "
            f"enrolled_only={enrolled_only} cookies={audio.cookies_source()}"
        )

        # --- Serve cached segments + load interval coverage -----------------
        cached = cache.load(
            video_id, language, task, engine, cache_model, variant=variant
        )
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
                frame = {
                    "type": "segment",
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                if seg.get("speaker"):
                    frame["speaker"] = seg["speaker"]
                if seg.get("enrolled"):
                    frame["enrolled"] = True
                if "source" in seg:
                    frame["source"] = seg["source"]
                if "sourceLang" in seg:
                    frame["sourceLang"] = seg["sourceLang"]
                if "sourceKind" in seg:
                    frame["sourceKind"] = seg["sourceKind"]
                await ws.send_json(frame)

        # --- Make sure the Whisper model is ready --------------------------
        if not await _wait_for_model(ws):
            return

        # The background load picks the auto tier; if this session needs a
        # different model, switch it now. The translate/transcribe task is passed
        # per call and must not force a reload.
        if transcribe.get_model_name() != whisper_model:
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

        # Effective engine stays fixed. Accurate source-first mode must not
        # silently downgrade to direct Whisper when Ollama fails, because that
        # caches softened/inaccurate translations as if they were good.
        eff_engine = engine
        # Whether we've already warned the client that diarization is unavailable
        # (no HF token / missing libs) — sent at most once per session.
        warned_diarize = False
        # Whether we've already warned the client that enrolled-only can't filter
        # (no enrolled voice / diarization unavailable) — sent at most once.
        warned_eo = False

        # eff_task tracks the task written to cache.
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
            window_had_signal = await asyncio.to_thread(
                transcribe.audio_has_signal_file, clean_wav
            )

            try:
                if eff_engine == "whisper":
                    # Fast beam at the playhead (latency is felt here); full beam
                    # for pre-buffered windows you haven't reached yet (quality).
                    beam = 1 if at_playhead else 5
                    # transcribe() now yields (s, e, t, detected_lang, confidence) and does
                    # NOT force the language (so we can filter by what's actually
                    # spoken). detected is used by the language filter below.
                    segs = await asyncio.to_thread(
                        _drain,
                        transcribe.transcribe(
                            clean_wav, time_offset=cursor,
                            hotwords=asr_hotwords, beam_size=beam,
                            model_name=whisper_model,
                            detect_per_segment=bool(language),
                        ),
                    )
                else:
                    # Accurate source-first mode: force the selected source
                    # language (default "cs") and do not run per-segment language
                    # filtering. This preserves Czech profanity/slang instead of
                    # dropping uncertain windows or translating directly.
                    source_language = language or None
                    segs = await asyncio.to_thread(
                        _drain,
                        transcribe.transcribe_source(
                            clean_wav, language=source_language, time_offset=cursor,
                            hotwords=None, model_name=whisper_model,
                            detect_per_segment=False,
                        ),
                    )

                # Language filter: when the user picked a specific spoken language
                # (not "auto"), keep only the SEGMENTS actually detected as that
                # language — e.g. subtitle the Czech streamer but drop the English
                # game audio EVEN WITHIN THE SAME WINDOW. Each seg carries its own
                # detected language (4th tuple element) via per-segment detection.
                if eff_engine == "whisper" and language and segs and not enrolled_only:
                    before = len(segs)
                    segs, filtered_empty, dropped = _filter_segments_by_language(
                        segs, language
                    )
                    if dropped:
                        _log(
                            f"window [{cursor:.1f},{window_end:.1f}] "
                            f"lang-filter={language}: kept {len(segs)}/{before} "
                            f"(dropped high-confidence non-target={dropped})"
                        )
                else:
                    filtered_empty = False

                # Optional speaker diarization: tag each segment with a stable
                # global "Speaker N" label. Must run INSIDE this try (before the
                # finally below deletes clean_wav) because it needs the audio
                # file. Heavy -> run on a worker thread. speakers[i] aligns with
                # segs[i]; on any failure / missing token/libs it stays None.
                speakers = None
                enrolled_flags = None
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
                    seg_dicts = [
                        {"start": s, "end": e}
                        for (s, e, t, sl, _lc) in segs
                    ]
                    labeled = await asyncio.to_thread(
                        sess.speaker_tracker.label_segments, clean_wav, seg_dicts, cursor
                    )
                    speakers = [d.get("speaker") for d in labeled]
                    # Parallel list: True where the speaker is an enrolled voice
                    # (lets the client paint "your voice" a fixed colour).
                    enrolled_flags = [bool(d.get("enrolled")) for d in labeled]
            except Exception as exc:
                _log(
                    f"window transcription failed @ {cursor:.1f}: "
                    f"{type(exc).__name__}: {str(exc)[:180]}"
                )
                await ws.send_json({
                    "type": "status",
                    "message": "skipping one unreadable audio window",
                })
                # Do not persist this as cache coverage, but do mark it covered
                # for this live session so playback does not get stuck forever.
                covered = cache.merge_intervals(covered + [[cursor, window_end]])
                await ws.send_json(
                    {"type": "progress", "start": cursor, "until": window_end}
                )
                continue
            finally:
                # Remove the fetched window WAV; also remove the denoised file,
                # but only when it's a distinct temp file (denoise returns the
                # original path when it didn't process -> don't double-clean).
                audio.cleanup(wav_path)
                if clean_wav != wav_path:
                    audio.cleanup(clean_wav)

            # Enrolled-only: keep ONLY segments spoken by an enrolled voice. This
            # replaces the language filter (which would drop his speech under
            # music) with speaker identity. Filter segs/speakers/enrolled_flags
            # together so the streaming loop, Ollama translate, and cache all see
            # just the kept lines.
            # When True, enrolled-only was requested but could NOT be honored this
            # window (no usable voiceprint / diarization down) -> we showed all
            # speakers, so we must NOT persist (else the 'eo' cache is poisoned
            # with unfiltered segments and replays everyone next session).
            eo_filter_failed = False
            if enrolled_only and segs:
                tracker = sess.speaker_tracker
                if enrolled_flags is None or tracker is None or not tracker.has_enrolled():
                    # No USABLE enrolled voiceprint (missing clip / embedding model
                    # failed / diarization unavailable) -> filtering would blank the
                    # screen. Warn once and show everyone. NB: a window that simply
                    # has no enrolled SPEECH is NOT this case (has_enrolled() is
                    # True), and the else-branch correctly empties it.
                    eo_filter_failed = True
                    if not warned_eo:
                        warned_eo = True
                        await ws.send_json({
                            "type": "status",
                            "message": "enrolled-only needs a voice clip in "
                            "helper/enroll/ + a HuggingFace token — showing all "
                            "speakers",
                        })
                else:
                    keep = [i for i in range(len(segs)) if enrolled_flags[i]]
                    segs = [segs[i] for i in keep]
                    speakers = [speakers[i] for i in keep] if speakers else None
                    enrolled_flags = [enrolled_flags[i] for i in keep]

            _log(
                f"window [{cursor:.1f},{window_end:.1f}] engine={eff_engine} "
                f"fetch={_t_fetch:.2f}s transcribe={time.time() - _t0 - _t_fetch:.2f}s "
                f"segs={len(segs)}"
            )

            # Stream (and, for ollama, translate) the window's segments.
            produced: list[dict] = []
            translations = None
            if eff_engine == "ollama" and segs:
                await ws.send_json({
                    "type": "status",
                    "message": f"translating with {model}",
                })
                translations = []
                try:
                    for _start, _end, text, src_lang, _lc in segs:
                        try:
                            out = await asyncio.to_thread(
                                translate_llm.translate,
                                text,
                                src_lang or language,
                                model,
                                list(sess.llm_context),
                                glossary,
                            )
                        except ValueError as exc:
                            translations.append(None)
                            _log(
                                f"invalid translation skipped @ {_start:.1f}: "
                                f"{str(exc)[:120]} source={text[:80]!r}"
                            )
                            continue
                        translations.append(out)
                        sess.llm_context.append((text, out))
                        sess.llm_context[:] = sess.llm_context[-2:]
                except Exception as exc:
                    msg = (
                        f"Accurate Czech translation failed with {model}: {exc}. "
                        "Start Ollama and pull gemma2:9b, or switch the popup to "
                        "Fast Whisper. This window was not cached."
                    )
                    _log(f"ollama translation FAILED video={video_id} @ {cursor:.1f}: {exc}")
                    await ws.send_json({"type": "error", "message": msg[:300]})
                    break

            for i, (start, end, text, src_lang, _lang_conf) in enumerate(segs):
                out_text = text
                if translations is not None:
                    out_text = translations[i]
                    if not out_text:
                        continue
                seg = {"start": start, "end": end, "text": out_text}
                if translations is not None:
                    seg["source"] = text
                    seg["sourceLang"] = src_lang or language or "unknown"
                    seg["sourceKind"] = "whisperx-asr"
                sp = speakers[i] if speakers else None
                if sp:
                    seg["speaker"] = sp
                    if enrolled_flags and enrolled_flags[i]:
                        seg["enrolled"] = True
                produced.append(seg)
                await ws.send_json({"type": "segment", **seg})

            # Mark the WHOLE window covered (even if silent) so we never reprocess
            # it, and persist incrementally.
            covered = cache.merge_intervals(covered + [[cursor, window_end]])
            # `start` lets the client rebuild coverage intervals for its gate.
            await ws.send_json(
                {"type": "progress", "start": cursor, "until": window_end}
            )
            # Skip persistence when the result is likely to poison cache:
            # enrolled-only couldn't be honored, language filtering blanked a
            # non-empty window, or an audible window produced no captions.
            if _should_persist_window(
                produced, filtered_empty, eo_filter_failed, window_had_signal
            ):
                await asyncio.to_thread(
                    cache.append,
                    video_id, produced, [cursor, window_end],
                    language, eff_task, eff_engine, _cache_model_for(eff_engine, model),
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
    banner = f"  YouTube Translator Helper  v{HELPER_VERSION}  "
    print("=" * len(banner), flush=True)
    print(banner, flush=True)
    print("  per-line glossary · num_predict cap · runaway guard", flush=True)
    print("  merged-clip enrollment · enrolled-only filter", flush=True)
    print("=" * len(banner), flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
