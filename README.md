# YouTube English Subtitles

A local Chrome extension and Python helper that overlays live English subtitles
on YouTube videos. Audio is resolved with `yt-dlp`, decoded in short windows
with `ffmpeg`, transcribed locally with WhisperX, and optionally translated by a
local Ollama chat model.

The project is built around a local-first privacy model: there is no hosted
backend, telemetry, paid API, or cloud transcription service. The helper runs on
`127.0.0.1:8765`, and local files such as `cookies.txt`, `hf_token.txt`, cache
JSON, logs, and enrolled voice clips must stay private on your machine.

## What It Does

- Chrome MV3 extension overlays generated subtitles on the YouTube player.
- Local FastAPI helper streams subtitle segments over a localhost WebSocket.
- Rolling window transcription starts near the playhead instead of downloading
  the whole video first, so captions can appear quickly even on long VODs.
- WhisperX provides batched transcription and optional pyannote diarization.
- Ollama translation mode transcribes source speech first, then translates with
  a local chat model. The current default and best-tested path is Czech to
  English with `gemma2:9b`.
- Fast Whisper mode can translate directly to English without Ollama.
- Hardware-adaptive quality can choose `large-v3`, `medium`, `small`, or `base`
  based on available VRAM.
- Optional clean-audio and speaker-label features degrade gracefully when their
  dependencies or tokens are missing.
- Per-video cache stores completed windows and resumes from uncovered ranges.

## Requirements

- Chrome or another Chromium browser that can load unpacked MV3 extensions.
- Python 3.10 or newer.
- `ffmpeg` on `PATH`.
- A CUDA-capable NVIDIA GPU is strongly recommended. CPU fallback exists, but it
  is much slower.
- Ollama is optional for direct Whisper mode, but recommended for the default
  Czech/Ollama workflow.

Windows is the primary tested platform today. Other platforms may work if
Python, ffmpeg, CUDA/PyTorch, and the browser extension setup are available.

## Quick Start

Install ffmpeg:

```powershell
winget install Gyan.FFmpeg
```

Install the helper:

```powershell
cd helper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the recommended Ollama translation path:

```powershell
ollama pull gemma2:9b
```

Run the helper:

```powershell
python server.py
```

Then load the extension:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select the `extension/` folder.
5. Open a YouTube video, click the extension popup, and enable subtitles on that
   tab.

The first helper run downloads the Whisper model selected by your quality
setting. `GET http://127.0.0.1:8765/health` reports whether the model is loaded,
CUDA is active, Ollama is reachable, and cookies are detected.

## Important Notes

`ctranslate2==4.8.0` is pinned intentionally. It is required for the tested RTX
5070 Ti / Blackwell GPU path. Do not upgrade or loosen that pin unless you have
verified GPU transcription on the target hardware.

Do not commit local runtime files:

- `helper/cookies.txt`
- `helper/hf_token.txt`
- `helper/cache/`
- `helper/*.log`
- `helper/enroll/*` except `helper/enroll/.gitkeep`
- `__pycache__/` and `*.pyc`

Normal public videos may work without cookies, but YouTube often serves
anonymous audio URLs that are poor at range seeking. A `cookies.txt` export can
make windowed seeking much more reliable, and age-restricted videos require
additional auth setup. See `how-to.html` for the full user guide.

## Documentation

- `how-to.html` - end-user setup and troubleshooting guide.
- `helper/README.md` - helper internals, API, WebSocket protocol, cache, and CLI
  testing.
- `ROADMAP.md` - release roadmap and future work.
- `PRIVACY.md` - local-first data handling and browser permission notes.
- `CHANGELOG.md` - notable project changes.
- `AGENTS.md` - maintainer/agent implementation notes.

## Development

There is no formal test suite yet. The closest build check is:

```powershell
cd helper
python -m py_compile audio.py server.py cache.py transcribe.py translate_llm.py denoise.py diarize.py
```

The canonical end-to-end check is to run the helper and drive the WebSocket
path:

```powershell
cd helper
python smoke_ws.py VIDEO_ID --start 0 --language cs --engine whisper
```

If you change extension behavior, bump `extension/manifest.json` so users can
reload the unpacked extension cleanly.

## License

MIT. See `LICENSE`.
