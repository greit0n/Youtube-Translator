# YouTube English Subtitle Helper

Local Python helper for the YouTube subtitle extension. It resolves a video's
audio stream with `yt-dlp`, fetches short audio windows with `ffmpeg`,
transcribes locally with WhisperX, optionally translates with Ollama, and
streams subtitle segments to the extension over a localhost WebSocket.

Subtitles are generated in rolling windows around the current playback position.
This avoids downloading full VODs up front and lets the extension show captions
quickly on long videos.

## Translation Engines

- `ollama` (default): WhisperX transcribes source speech first, then a local
  Ollama chat model translates each line to English. The current best-tested
  default is Czech source speech with `gemma2:9b`. This path preserves names,
  slang, and profanity better than direct Whisper translation.
- `whisper`: Whisper translates directly to English with `task="translate"`.
  This is faster and does not require Ollama, but can be less faithful.

If Ollama is unavailable or returns invalid output, the helper reports an error
for that window and avoids caching bad output.

## Runtime Model

The extension is per-tab opt-in. After a YouTube tab is enabled, the content
script connects to `ws://127.0.0.1:8765/transcribe` and sends playback position
updates. The helper keeps subtitle coverage ahead of the playhead:

- `LEAD = 90.0` seconds ahead of playback.
- `WINDOW = 45.0` seconds for normal fetches.
- `FIRST_WINDOW = 12.0` seconds for fast first captions.

Coverage is tracked as intervals, not a single cursor, so seeks and mid-video
starts can fill only the missing ranges.

## Hardware-Adaptive Quality

The `quality` setting chooses the Whisper model:

- `auto`: probes VRAM and picks `large-v3`, `medium`, `small`, or `base`.
- `max`: `large-v3`.
- `balanced`: `medium`.
- `lite`: `small`.

Changing quality can briefly reload the model. The active model appears as
`whisper_model` in `/health`.

`ctranslate2==4.8.0` is pinned intentionally. It is required for the tested RTX
5070 Ti / Blackwell GPU path. WhisperX can otherwise pull a different
CTranslate2 version, which may break CUDA on that hardware or force CPU
fallback. After dependency changes, verify `/health` reports `"cuda": true` and
confirm a real video transcribes on GPU.

## Optional Features

`denoise.py` exposes `clean(wav, mode)`:

- `off`: no-op.
- `light`: DeepFilterNet.
- `music`: Demucs vocal isolation.

These require `requirements-optional.txt`. Without the optional dependencies,
audio passes through unchanged.

`diarize.py` exposes `SpeakerTracker` for pyannote/WhisperX speaker labels. It
needs a HuggingFace token via `HF_TOKEN`, `HUGGINGFACE_TOKEN`, or
`helper/hf_token.txt`. Without a token, diarization is a graceful no-op.

The `enrolledOnly` setting keeps only an enrolled speaker matched against clips
in `helper/enroll/`. It implies diarization. If no enroll clip or token exists,
the helper warns and shows all speakers.

## System Dependencies

Install ffmpeg:

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
```

Install a CUDA-enabled PyTorch build when using an NVIDIA GPU:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install Ollama for the default translation path:

```powershell
ollama pull gemma2:9b
ollama pull aya-expanse:8b
```

Ollama should be reachable at `http://127.0.0.1:11434`.

## Install

```powershell
cd helper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional clean-audio extras:

```powershell
pip install -r requirements-optional.txt
```

## Run

```powershell
python server.py
```

The server listens on `http://127.0.0.1:8765`. The first run downloads the
selected Whisper model. Model loading runs in the background, so `/health` may
report `"model_loaded": false` until the model is ready.

Example `/health` response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "cuda": true,
  "device": "cuda",
  "ollama": true,
  "cookies": true,
  "whisper_model": "large-v3",
  "vram_gb": 16.0
}
```

## HTTP API

- `GET /health` returns helper status, model load state, CUDA/device info,
  Ollama reachability, cookie detection, active Whisper model, and probed VRAM.
- `GET /models` returns installed Ollama chat models. Embedding models are
  filtered out.
- `POST /reset` with `{"videoId":"..."}` clears cached subtitles for a video.

## WebSocket Protocol

Endpoint:

```text
ws://127.0.0.1:8765/transcribe
```

First client message:

```json
{
  "videoId": "abc123",
  "startTime": 0,
  "language": "cs",
  "engine": "ollama",
  "model": "gemma2:9b",
  "preBuffer": true,
  "quality": "auto",
  "cleanAudio": "off",
  "diarize": false,
  "enrolledOnly": false,
  "hotwords": null,
  "glossary": [{"term": "name", "preferred": "Name"}]
}
```

Ongoing client messages:

```json
{"type": "position", "currentTime": 123.45}
```

Server messages:

- `{"type":"status","message":"..."}`
- `{"type":"segment","start":1.0,"end":2.0,"text":"...","speaker":"Speaker 1"}`
- `{"type":"progress","start":0.0,"until":45.0}`
- `{"type":"done"}`
- `{"type":"error","message":"..."}`

Segment timestamps are absolute video timestamps. The optional `speaker` field
appears only when diarization is active.

## CLI Test

Transcribe or translate a local audio file without the browser:

```powershell
python transcribe.py path\to\audio.m4a
python transcribe.py path\to\audio.m4a cs
```

For the full helper path, run `server.py` and drive the WebSocket directly with
the `websockets` Python package. Send the start message, then send
`{"type":"position","currentTime":N}` frames and print incoming `status`,
`segment`, and `progress` messages.

## Cache

Cache files live under `helper/cache/` and are ignored by Git. Keys include:

- video ID
- source language
- task
- engine
- Ollama model
- pipeline variant

The variant includes the resolved Whisper model, clean-audio mode, diarization
mode, enrolled-only mode, prompt/glossary profile, and pipeline version.

Each cache stores `covered` intervals and `segments`. Reopening a video replays
cached coverage and resumes from uncovered ranges. Legacy `covered_until` cache
files and old keys without a variant are treated as misses.
