# AGENTS.md

This file gives maintainers and coding agents the repo-specific context needed
to work safely in this project.

## Project Shape

This is a live YouTube to English subtitle translator that runs locally:

- `extension/`: Chrome MV3 extension. The content script overlays captions on
  the YouTube player. The popup controls settings and helper health.
- `helper/`: Python FastAPI server on `127.0.0.1:8765`. It resolves YouTube
  audio with `yt-dlp`, fetches short windows with `ffmpeg`, transcribes with
  WhisperX, optionally translates with Ollama, and streams subtitle segments over
  WebSocket.

Windows with an NVIDIA GPU is the primary tested environment. The RTX 5070 Ti /
Blackwell path depends on the pinned `ctranslate2==4.8.0` package.

## Commands

```powershell
cd helper
python server.py

python -m py_compile audio.py server.py cache.py transcribe.py translate_llm.py denoise.py diarize.py

python transcribe.py path\to\audio.m4a
python transcribe.py path\to\audio.m4a cs

python smoke_ws.py VIDEO_ID --start 0 --language cs --engine whisper
```

Load the extension through `chrome://extensions`, enable Developer mode, and
select `extension/` with Load unpacked.

After editing extension behavior, bump `extension/manifest.json` `version`, then
reload the extension and hard-reload the YouTube tab.

## Architecture

The core design is the windowed lead-following loop:

1. `audio.get_url_and_duration(video_id)` resolves a direct audio URL.
2. `server.py` tracks the client's latest playback position and finds the next
   uncovered interval around the playhead.
3. `audio.fetch_window(url, start, dur)` uses `ffmpeg -ss` and `-t` to range-seek
   only that audio slice.
4. The helper transcribes the window, optionally translates it, streams absolute
   timestamped `segment` frames, and records coverage in `cache`.

Key constants in `server.py`:

- `LEAD = 90.0`
- `WINDOW = 45.0`
- `FIRST_WINDOW = 12.0`

Coverage is stored as intervals, not a single `covered_until`, so mid-video
starts and seeks do not poison earlier ranges.

## Gotchas

- Cookie-backed YouTube audio URLs often seek much more reliably than anonymous
  URLs. Do not change `audio.extract_info` back to anonymous-first.
- Kill in-flight `ffmpeg` on disconnect. `fetch_window` exposes `on_proc`, and
  `server.py` stores the live process on the session for cleanup.
- Reloading an unpacked extension can leave orphaned content scripts in existing
  tabs. Keep the content-script watchdog intact.
- Keep `ctranslate2==4.8.0` pinned unless GPU transcription has been verified on
  the tested Blackwell path and any newly claimed target hardware.
- Optional `denoise.py` and `diarize.py` paths must stay lazy and graceful. If
  dependencies or tokens are absent, captions should still work.

## WebSocket Contract

Client first message:

```json
{
  "videoId": "abc123",
  "startTime": 0,
  "language": null,
  "engine": "ollama",
  "model": "gemma2:9b",
  "preBuffer": true,
  "quality": "auto",
  "cleanAudio": "off",
  "diarize": false,
  "enrolledOnly": false,
  "hotwords": null,
  "glossary": []
}
```

Client ongoing message:

```json
{"type":"position","currentTime":123.45}
```

Server messages:

- `{"type":"status","message":"..."}`
- `{"type":"segment","start":1.0,"end":2.0,"text":"...","speaker":"Speaker 1"}`
- `{"type":"progress","start":0.0,"until":45.0}`
- `{"type":"done"}`
- `{"type":"error","message":"..."}`

Keep `content.js`, `server.py`, `helper/README.md`, and this file in lockstep
when changing protocol fields.

## Cache

Cache files live under `helper/cache/` and are ignored by Git. The cache key
includes video ID, source language, task, engine, model, and pipeline variant.
The variant folds in resolved Whisper model, clean-audio mode, diarization mode,
enrolled-only mode, prompt/glossary profile, and pipeline version.

## Public Release Hygiene

Never commit:

- `helper/cookies.txt`
- `helper/hf_token.txt`
- `helper/cache/`
- `helper/*.log`
- `helper/enroll/*` except `.gitkeep`
- model downloads, temp audio, or `__pycache__/`

Update `README.md`, `how-to.html`, and `helper/README.md` whenever setup,
settings, API behavior, or user-visible workflows change.
