# YouTube → English Subtitle Helper (local)

Local Python helper for the YouTube translator Chrome extension. It resolves a
video's audio stream with `yt-dlp`, runs **faster-whisper `large-v3`** on your
GPU, and streams English subtitle segments back to the extension over a
WebSocket. Transcripts (full or partial) are cached to disk per video.

Two translation engines:

- **`ollama` (default, recommended):** Whisper transcribes the **source
  language faithfully** (`task="transcribe"`), then a local **Ollama** chat
  model translates each line to natural spoken English while **preserving
  profanity, slang, gaming terms, and names**. This is much more faithful than
  Whisper's built-in translate, which tends to censor and drop names.
- **`whisper` (fallback):** Whisper translates directly (`task="translate"`).
  Used automatically if Ollama is unreachable.

Subtitles are produced in **rolling windows** that follow your playback
position (lead-following + pre-buffer), so they start within a couple seconds
even on multi-hour VODs — no full-file download.

Runs on Windows 11. Built for an RTX 5070 Ti 16GB (CUDA / float16), with an
automatic CPU/int8 fallback if CUDA isn't usable.

> `large-v3` is used **on purpose** (not `large-v3-turbo`): turbo is noticeably
> weaker, and transcription/translation quality is the whole point.

## System dependencies

### 1. ffmpeg (REQUIRED)
ffmpeg now does the live windowed fetch (`ffmpeg -ss <start> -t <W> -i <url>`),
range-seeking into the remote audio stream and decoding only the current window
to 16kHz mono WAV. The helper **cannot run the low-latency path without it.**

```powershell
winget install Gyan.FFmpeg
```

Or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its `bin`
folder to your `PATH`. Verify:

```powershell
ffmpeg -version
```

### 2. CUDA-enabled runtime
faster-whisper runs on CTranslate2, which needs the CUDA 12 + cuDNN runtime
libraries on your `PATH` for GPU inference. The simplest way to get them on
Windows is to install a CUDA-enabled PyTorch build, which bundles the needed
DLLs:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If CUDA isn't available, the helper automatically falls back to CPU (`int8`)
and reports the real device in `/health`.

### 3. Ollama (for the default `ollama` engine)
Install Ollama (https://ollama.com) and pull a chat model. The default is
`qwen2.5:7b`:

```powershell
ollama pull qwen2.5:7b
```

Ollama should be running at `http://127.0.0.1:11434`. The popup model dropdown
is populated from `GET /models` (embedding models are filtered out). If Ollama
is down, the helper transparently falls back to the `whisper` engine.

## Install

```powershell
cd helper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python server.py
```

The server listens on `http://127.0.0.1:8765`.

**First run downloads the `large-v3` model (~3 GB).** The model loads in the
background, so `GET /health` will report `"model_loaded": false` until the
download + load finishes. After that it flips to `true`. Example:

```json
{"status":"ok","model_loaded":true,"cuda":true,"device":"cuda","ollama":true}
```

## API (shared contract with the extension)

- `GET /health` →
  `{"status":"ok","model_loaded":bool,"cuda":bool,"device":"cuda"|"cpu","ollama":bool}`
  (`ollama` = is Ollama reachable).
- `GET /models` → `{"models":[<chat model names>]}` — Ollama models with any
  "embed" model filtered out. Populates the popup dropdown.
- `WS /transcribe`:
  - First client message:
    `{"videoId":str,"startTime":float,"language":null|str,"engine":"ollama"|"whisper","model":str,"preBuffer":bool,"hotwords":null|str}`
    (`language: null` = auto-detect).
  - Ongoing client messages while watching:
    `{"type":"position","currentTime":float}` — sent every few seconds and on
    seek. Read concurrently by the server; drives lead-following/pre-buffer.
  - Server streams:
    - `{"type":"status","message":"..."}` — e.g. `resolving stream`,
      `loading model`, `transcribing`, `cached`, or the Ollama-fallback notice.
    - `{"type":"segment","start":<float>,"end":<float>,"text":"<english>"}` —
      absolute video timestamps.
    - `{"type":"progress","start":<float>,"until":<float>}` — a covered
      interval. `start` lets the client rebuild coverage for its playback gate;
      on a cache hit the server replays one frame per cached interval first.
    - `{"type":"done"}`
    - `{"type":"error","message":"..."}`

### Lead-following / pre-buffer
The server stays ~`LEAD` (90s) ahead of `currentTime`, fetching `WINDOW` (45s)
chunks. When far enough ahead (e.g. while paused) it idles and re-checks. With
`preBuffer:false` it waits for the first position update before starting.

## CLI test (no extension needed)

Transcribe/translate any local audio file straight to the console (uses the
direct Whisper translate path):

```powershell
python transcribe.py path\to\audio.m4a          # auto-detect source language
python transcribe.py path\to\audio.m4a cs        # force Czech source
```

## Cache

Transcripts are stored as JSON under `cache/`, keyed by
`videoId + language + task + engine + model`, with a `covered_until` marker.
Partial transcripts are persisted as windows complete; re-opening a video
streams the covered range instantly and resumes transcribing only the
uncovered tail. Delete files in `cache/` to force re-transcription.
