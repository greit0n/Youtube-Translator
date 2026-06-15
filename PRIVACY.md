# Privacy

YouTube English Subtitles is designed as a local-first tool. There is no hosted
project backend, telemetry service, analytics endpoint, or paid cloud
transcription API.

## What Stays Local

These stay on your machine:

- Audio windows fetched by the helper.
- WhisperX transcription work.
- Ollama translation requests and responses when using the default local Ollama
  engine.
- Subtitle cache files in `helper/cache/`.
- `helper/cookies.txt`, if you use YouTube cookies.
- `helper/hf_token.txt`, if you use speaker labels.
- Voice enrollment clips in `helper/enroll/`.

The Chrome extension talks to the helper on `127.0.0.1:8765`.

## External Services You May Contact

The project can contact these services as part of normal operation:

- YouTube, through your browser and through `yt-dlp`, to resolve and fetch video
  audio.
- Ollama on `127.0.0.1:11434`, if you use the local Ollama translation engine.
- HuggingFace, if you enable speaker diarization and need to download pyannote
  models with your token.
- Model/package hosts such as PyPI, PyTorch, HuggingFace, or Ollama during
  installation and first model downloads.

## Cookies

`cookies.txt` can contain YouTube login tokens. Treat it like a password. It is
ignored by Git and should never be uploaded to issues, pull requests, logs, or
screenshots.

## Voice Enrollment

Files in `helper/enroll/` are personal voice samples. They are ignored by Git
except for `.gitkeep`. Do not publish them.

## Browser Permissions

The extension uses storage for settings, accesses YouTube watch pages to render
the overlay, and connects to the local helper over HTTP/WebSocket on
`127.0.0.1:8765`.
