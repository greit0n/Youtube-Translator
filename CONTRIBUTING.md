# Contributing

Thanks for helping improve YouTube English Subtitles. This project is still a
local-first tool with a small codebase, so keep changes focused and easy to
verify.

## Development Setup

Install helper dependencies:

```powershell
cd helper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional clean-audio dependencies:

```powershell
pip install -r requirements-optional.txt
```

Run the helper:

```powershell
python server.py
```

Load the extension from `extension/` through `chrome://extensions` with
Developer mode enabled.

## Verification

There is no formal unit-test suite yet. Use the syntax check as the closest
build check:

```powershell
cd helper
python -m py_compile audio.py server.py cache.py transcribe.py translate_llm.py denoise.py diarize.py
```

The canonical end-to-end check is the WebSocket path:

1. Start `python server.py`.
2. Run `python smoke_ws.py VIDEO_ID --start 0 --language cs --engine whisper`.
3. Confirm incoming `status`, `progress`, and `segment` messages.

For custom cases, connect to `ws://127.0.0.1:8765/transcribe` with the
`websockets` Python package and send the same start payload the extension sends.

Browser verification is still useful for UI changes: reload the unpacked
extension, hard-reload the YouTube tab, enable the popup toggle, and watch for
subtitles.

## Extension Versioning

When changing extension behavior, bump `extension/manifest.json` `version`.
Users of the unpacked extension need that bump to update and verify what is
running.

## Documentation

Update docs when setup steps, settings, WebSocket fields, cache keys, or user
visible behavior changes:

- `README.md` for project-level setup and positioning.
- `how-to.html` for end-user setup/troubleshooting.
- `helper/README.md` for helper API and protocol details.
- `AGENTS.md` for implementation notes and gotchas.

## Secrets and Local Artifacts

Never commit:

- `helper/cookies.txt`
- `helper/hf_token.txt`
- `helper/cache/`
- `helper/*.log`
- `helper/enroll/*` except `.gitkeep`
- `__pycache__/` or `*.pyc`

These files can contain account tokens, private audio, local paths, or generated
transcripts.

## Dependency Care

Keep `ctranslate2==4.8.0` pinned unless you have verified CUDA transcription on
the tested Blackwell path and any other target hardware you claim to support.
