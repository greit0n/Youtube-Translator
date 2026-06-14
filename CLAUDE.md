# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **live YouTube → English subtitle translator** that runs 100% free and locally on the user's GPU (RTX 5070 Ti, Blackwell/sm_120). Two components that talk over a localhost WebSocket:

- **`extension/`** — Chrome MV3 extension. Content script overlays captions on the YouTube player; popup is settings + helper status; background SW seeds defaults.
- **`helper/`** — Python FastAPI server on `127.0.0.1:8765`. Resolves audio with `yt-dlp`, runs **faster-whisper `large-v3`** on CUDA, optionally post-translates via **Ollama**, streams subtitle segments back.

There is **no build step, no test framework, no linter, no CI, no git repo**. "Testing" means running the helper and exercising it (see below).

## Commands

```powershell
# Run the helper (from helper/). First run downloads large-v3 (~3GB); model loads
# in a background thread, so /health reports model_loaded:false until ready.
cd helper
python server.py                      # uvicorn on 127.0.0.1:8765

python -m py_compile audio.py server.py cache.py transcribe.py translate_llm.py   # syntax check (closest thing to a build)

# CLI transcribe test — no extension/browser needed (uses direct Whisper translate):
python transcribe.py path\to\audio.m4a        # auto-detect source
python transcribe.py path\to\audio.m4a cs      # force Czech

# Load the extension: chrome://extensions → Developer mode → Load unpacked → select extension/
# After editing extension code, BUMP extension/manifest.json "version" so the user can update,
# then reload the extension AND hard-reload the YouTube tab (Ctrl+Shift+R).
```

**There is no unit-test suite.** To verify the helper end-to-end without a browser, drive the WebSocket directly with the `websockets` library: connect to `ws://127.0.0.1:8765/transcribe`, send a start message then a `{"type":"position","currentTime":N}` frame, and print incoming `segment`/`status` messages. This is the canonical way to reproduce and confirm fetch/transcription behavior.

## Architecture: the windowed lead-following loop (the core idea)

The whole design exists to make captions appear in **seconds on multi-hour VODs** without downloading the whole file. Understanding it requires reading `server.py`, `audio.py`, and `cache.py` together:

1. **`audio.get_url_and_duration(video_id)`** (`audio.py`) resolves the direct deciphered audio URL via `yt-dlp` — no download.
2. The **`WS /transcribe`** loop in `server.py` keeps the region around the playhead covered. Each pass: read the client's latest `currentTime`, find the nearest **uncovered** point at/after it (`_next_uncovered` + interval coverage), and `fetch_window` only that slice.
3. **`audio.fetch_window(url, start, dur)`** runs `ffmpeg -ss <start> -t <dur> -i <url> ... 16kHz mono WAV` — an HTTP **range-seek** into the remote stream, decoding only that window.
4. Transcribe the window → (optionally Ollama-translate) → stream `segment` frames with **absolute** video timestamps → mark the interval covered in `cache`.

Key constants (top of `server.py`): `LEAD=90.0` (stay this far ahead of playback), `WINDOW=45.0` (normal fetch size), `FIRST_WINDOW=12.0` (small window right at the playhead for a fast first caption). `audio.FETCH_TIMEOUT=90.0`; the loop passes a **shorter 25s timeout at the playhead** so a throttled stream surfaces a status message in seconds instead of hanging.

The loop re-targets the **current** playhead every pass, so forward/backward seeks and pre-buffer-while-paused all fall out naturally. Coverage is tracked as **intervals** (`covered = [[start,end],...]`), not a single `covered_until` — a session that starts mid-video must not poison the start as "done."

## Critical, non-obvious gotchas

- **Cookies are effectively REQUIRED, not just for age-restricted videos.** The anonymous (no-cookie) audio URL is frequently served **sequential-only** — YouTube throttles ranged `ffmpeg -ss` seeks on it to **zero bytes**, so offset 0 plays but every window past the start hangs (symptom: captions start, then freeze on "transcribing"). The authenticated (cookie) URL seeks deep instantly. `audio.extract_info` therefore **prefers the cookie path** when `cookies.txt` exists, falling back to anonymous only on failure. Do not "optimize" this back to no-cookie-first. See `memory/ytdlp-anonymous-url-no-seek.md`.

- **Kill in-flight ffmpeg on disconnect.** Reloads/navigations otherwise leave `ffmpeg` downloads running and piling up, hammering YouTube. `fetch_window` takes an `on_proc` callback; `server.py` stores the live process on `Session.current_proc` and `_kill_proc`s it on disconnect and in `finally`. When debugging "stuck" issues, first check for orphaned `ffmpeg` processes.

- **Orphaned content scripts.** Reloading the unpacked extension severs the `chrome.*` bridge to the already-injected content script but leaves its overlay on the page; the new script isn't injected into open tabs. `content.js` has an orphan watchdog (checks `chrome.runtime.id` each 1s tick) that self-removes the stale overlay. A switch toggle must be a `<label>`, not a `<span>`, or clicks don't reach the checkbox.

- **Whisper model is now hardware-adaptive via the `quality` setting** (`auto|max|balanced|lite`). `auto` probes VRAM and picks `large-v3` (≥10 GB) / `medium` (≥5 GB) / `small` (≥2.5 GB) / `base` (else); `max`=large-v3, `balanced`=medium, `lite`=small. **`large-v3` is still the default/`max` on purpose** (not `large-v3-turbo` — turbo is weaker and quality is the point). Changing `quality` can trigger a brief model reload ("switching model"). GPU works on Blackwell via ctranslate2 4.8.0 with no special cuDNN setup (`memory/blackwell-ctranslate2-cuda.md`).

- **`ctranslate2==4.8.0` is PINNED in `requirements.txt` — do not let it drift.** The transcription core is now **WhisperX**, which will happily pull a newer ctranslate2 that **breaks the Blackwell/sm_120 (RTX 5070 Ti) GPU** (falls back to CPU or errors). The pin is the #1 gotcha after the WhisperX migration. After any reinstall, confirm `/health` still reports `cuda:true` and that a video actually transcribes on GPU.

## Transcription core (WhisperX) + two translation engines

The transcription core was migrated from raw faster-whisper to **WhisperX** (batched VAD + integrated pyannote diarization). Two optional helper modules sit on top, both **lazy, optional, and graceful** (never raise, pass through if deps/token missing):
- **`denoise.py`** — `clean(wav, mode)` cleans a window before transcription. `light` = DeepFilterNet, `music` = Demucs vocal isolation. Driven by the `cleanAudio` setting (`off|light|music`). Extra deps in `requirements-optional.txt`.
- **`diarize.py`** — `SpeakerTracker`, WhisperX/pyannote diarization with **stable cross-window "Speaker N" labels** via voice-fingerprint embeddings. Driven by the `diarize` setting; needs a free HuggingFace token (`HF_TOKEN` env var or `helper/hf_token.txt`). No extra pip install beyond core whisperx.

Translation engines:
- **`whisper`** (current default in code): Whisper `task="translate"` → English directly. Faster, single-step.
- **`ollama`**: Whisper `task="transcribe"` (faithful source language) → local Ollama chat model translates, preserving profanity/slang/names. The `glossary` terms are injected into the prompt to keep translations consistent. Falls back to `whisper` automatically if Ollama is unreachable. **Known issue:** `qwen2.5:7b` sometimes emits Chinese characters, which is why the default was set to `whisper`.

Settings (`engine`, `model`, `language`, `preBuffer`, `quality`, `cleanAudio`, `diarize`, `glossary`) live in `chrome.storage.sync` (defaults in `extension/background.js`), are sent in the WS start message, and changing any of them triggers a full session reinit (reconnect) in `content.js`. The `glossary` is a multiline editor (one entry per line, `term` or `term = preferred`); terms also bias Whisper recognition as hotwords.

## WS protocol (shared contract — keep `content.js` and `server.py` in lockstep)

- **Client → server, first message:** `{videoId, startTime, language(null=auto), engine, model, preBuffer, quality, cleanAudio, diarize, hotwords, glossary}` — `hotwords` derived from the glossary; `glossary` is an array of `{term, preferred}`.
- **Client → server, ongoing:** `{type:"position", currentTime}` every ~4s and on seek (drives lead-following).
- **Server → client:** `{type:"status",message}`, `{type:"segment",start,end,text,speaker?}` (absolute timestamps; optional `speaker` field when diarization is on), `{type:"progress",start,until}` (a covered interval; `start` lets the client rebuild coverage for its playback gate — on a cache hit the server replays one progress frame per cached interval before streaming segments), `{type:"done"}`, `{type:"error",message}`.
- **Client-side playback gate (`content.js`):** when the `autoPause` setting is on, the content script mirrors the `progress` intervals and pauses the `<video>` whenever the current moment isn't covered yet, resuming once coverage reaches `currentTime + ~1.5s`. It distinguishes transcribed-but-silent (play) from not-yet-transcribed (hold), never fights a manual pause/play, and on a genuine stuck fetch stays paused with a "waiting for subtitles" message (manual play overrides).
- **`GET /health`** → `{status,model_loaded,cuda,device,ollama,cookies,whisper_model,vram_gb}` (`whisper_model` = the model the helper actually loaded after VRAM probing; `vram_gb` = probed VRAM). **`GET /models`** → installed Ollama chat models (embeddings filtered). **`POST /reset` `{videoId}`** → `cache.clear` for that video (powers the popup's "Re-translate this video" button).

## Cache

JSON under `helper/cache/`, keyed by `videoId + language + task + engine + model + variant` (the `variant` folds in the resolved whisper model | clean-audio mode | diar-vs-mono, so different combos cache separately). Stores `covered` intervals + `segments`. Re-opening a video replays the covered range instantly and resumes only the uncovered tail. Legacy `covered_until`-only files (and old keys without `variant`) auto-heal (treated as a miss). Delete files or use `POST /reset` to force re-transcription.

## Age-restricted ("needs_auth") video stack

Normal videos need only `cookies.txt`. Age-restricted videos additionally require a **PO token** (Docker container `bgutil-pot` on `127.0.0.1:4416`, plus the `bgutil-ytdlp-pot-provider` pip plugin) and a **solved JS challenge** (`yt-dlp-ejs` plugin + **Deno** runtime — yt-dlp prefers sandboxed Deno over Node). All documented in `how-to.html`. Cookies live at `helper/cookies.txt` (auto-detected) or via `YTDLP_COOKIES_FILE`. They expire/rotate; re-export from an Incognito window then close it for stability.

## User-facing docs

`how-to.html` is the end-user setup guide (ffmpeg, CUDA, Ollama, cookies, the age-restricted stack). Keep it in sync when changing setup requirements or the engine/settings UI. `helper/README.md` is the developer-facing helper doc.
