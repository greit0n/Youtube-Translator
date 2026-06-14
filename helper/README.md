# YouTube → English Subtitle Helper (local)

Local Python helper for the YouTube translator Chrome extension. It resolves a
video's audio stream with `yt-dlp`, transcribes on your GPU with **WhisperX**
(batched VAD + integrated pyannote diarization), and streams English subtitle
segments back to the extension over a WebSocket. Transcripts (full or partial)
are cached to disk per video.

> **Transcription core migrated to WhisperX.** It replaces the raw
> faster-whisper calls but still runs the same CTranslate2 backend, so the
> Blackwell/sm_120 GPU path is unchanged — **as long as `ctranslate2` stays
> pinned at `4.8.0`** (see the warning under [Install](#install)).

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

### Whisper model is hardware-adaptive (`quality` setting)

The loaded model is now chosen from the `quality` setting (`auto|max|balanced|lite`):

- `auto` (default) probes VRAM and picks `large-v3` (≥10 GB) / `medium`
  (≥5 GB) / `small` (≥2.5 GB) / `base` (else).
- `max` = `large-v3`, `balanced` = `medium`, `lite` = `small`.

Changing `quality` mid-session can trigger a brief model reload
("switching model"). The resolved model is reported as `whisper_model` in
`/health`.

> `large-v3` (the default/`max`) is used **on purpose** (not `large-v3-turbo`):
> turbo is noticeably weaker, and transcription/translation quality is the
> whole point.

### Optional features (denoise, diarization)

Two optional modules sit on top of the transcription core. Both are **lazy,
optional, and graceful** — if their dependencies (or the HuggingFace token)
aren't present they simply pass through and never raise:

- **`denoise.py`** — `clean(wav, mode)` cleans each audio window before
  transcription. Driven by the `cleanAudio` setting: `light` = DeepFilterNet,
  `music` = Demucs vocal isolation, `off` = no-op. Needs the optional deps (see
  [Install](#install)).
- **`diarize.py`** — `SpeakerTracker` runs WhisperX/pyannote diarization and
  assigns **stable "Speaker N" labels across windows** via voice-fingerprint
  embeddings. Driven by the `diarize` setting (bool). Needs a free HuggingFace
  token (no extra pip install beyond core WhisperX). When on, segments may carry
  a `speaker` field.

The `enrolledOnly` setting ("only my voice") narrows output to **just the
enrolled speaker** — it drops game audio and every other speaker, keeping only
lines matched to the voice clip(s) in `helper/enroll/`. It **implies
diarization** (turning it on automatically enables the diarize path), so it has
the same requirements as Speaker labels: an enroll clip in `helper/enroll/` plus
a HuggingFace token. If either is missing it **degrades gracefully** — the helper
shows all speakers and emits a `status` warning instead of erroring. Like the
other feature toggles it folds into the cache `variant`, so flipping it caches
separately.

The `glossary` setting (array of `{term, preferred}`) is used two ways: the
terms bias Whisper recognition as **hotwords**, and they're injected into the
Ollama prompt to keep names/terms consistent in the translation.

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
`gemma2:9b`; `aya-expanse:8b` is the recommended alternative — both are strong
multilingual models that avoid the `qwen2.5:7b` Chinese-character leak:

```powershell
ollama pull gemma2:9b
ollama pull aya-expanse:8b
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

`requirements.txt` now pulls **WhisperX** and **pins `ctranslate2==4.8.0`**.

> ⚠️ **Do not let `ctranslate2` drift off 4.8.0.** WhisperX (and other deps)
> will happily upgrade it to a version that **breaks the Blackwell/sm_120
> (RTX 5070 Ti) GPU** — transcription silently drops to CPU or errors. The pin
> is the #1 gotcha of the WhisperX migration. After installing, verify the GPU
> still works: check `GET /health` reports `"cuda": true` and confirm a real
> video actually transcribes on GPU (watch the startup logs for `cuda`, not
> `cpu`).

### Optional extras

**Clean audio (Light/Music).** DeepFilterNet + Demucs live in a separate file
so the core install stays lean:

```powershell
pip install -r requirements-optional.txt
```

Without these the `cleanAudio` setting safely passes audio through unchanged.

**Speaker labels (diarization).** No extra pip install is needed beyond core
WhisperX, but pyannote's models are gated, so you need a **free HuggingFace
token**:

1. Create a free account at https://huggingface.co.
2. Accept the model terms for **`pyannote/speaker-diarization-3.1`** and
   **`pyannote/segmentation`** (visit each model page while logged in and accept).
3. Create an access token (Settings → Access Tokens) and provide it either via
   the `HF_TOKEN` environment variable **or** a `helper/hf_token.txt` file next
   to the module.

Without a token, `diarize` is a graceful no-op (captions just have no speaker
labels).

## Run

```powershell
python server.py
```

The server listens on `http://127.0.0.1:8765`.

**First run downloads the `large-v3` model (~3 GB).** The model loads in the
background, so `GET /health` will report `"model_loaded": false` until the
download + load finishes. After that it flips to `true`. Example:

```json
{"status":"ok","model_loaded":true,"cuda":true,"device":"cuda","ollama":true,"whisper_model":"large-v3","vram_gb":16.0}
```

## API (shared contract with the extension)

- `GET /health` →
  `{"status":"ok","model_loaded":bool,"cuda":bool,"device":"cuda"|"cpu","ollama":bool,"cookies":bool,"whisper_model":str,"vram_gb":float}`
  (`ollama` = is Ollama reachable; `whisper_model` = model the helper actually
  loaded after VRAM probing; `vram_gb` = probed VRAM).
- `GET /models` → `{"models":[<chat model names>]}` — Ollama models with any
  "embed" model filtered out. Populates the popup dropdown.
- `WS /transcribe`:
  - First client message:
    `{"videoId":str,"startTime":float,"language":null|str,"engine":"ollama"|"whisper","model":str,"preBuffer":bool,"quality":"auto"|"max"|"balanced"|"lite","cleanAudio":"off"|"light"|"music","diarize":bool,"enrolledOnly":bool,"hotwords":null|str,"glossary":[{"term":str,"preferred":str}]}`
    (`language: null` = auto-detect; `hotwords` derived from `glossary`;
    `enrolledOnly` keeps only the enrolled speaker and implies diarization).
  - Ongoing client messages while watching:
    `{"type":"position","currentTime":float}` — sent every few seconds and on
    seek. Read concurrently by the server; drives lead-following/pre-buffer.
  - Server streams:
    - `{"type":"status","message":"..."}` — e.g. `resolving stream`,
      `loading model`, `transcribing`, `cached`, or the Ollama-fallback notice.
    - `{"type":"segment","start":<float>,"end":<float>,"text":"<english>","speaker":"<Speaker N>"?}` —
      absolute video timestamps; the optional `speaker` field is present only
      when diarization is on.
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
`videoId + language + task + engine + model + variant`. The `variant` folds in
the resolved Whisper model, the clean-audio mode, diarized-vs-mono, and the
enrolled-only flag, so different feature combos cache separately. Coverage is tracked as `covered`
intervals. Partial transcripts are persisted as windows complete; re-opening a
video streams the covered range instantly and resumes transcribing only the
uncovered tail. Legacy `covered_until`-only files and old keys without a
`variant` auto-heal (treated as a miss). Delete files in `cache/` or use
`POST /reset` to force re-transcription.
