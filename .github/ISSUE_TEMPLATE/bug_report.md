---
name: Bug report
about: Report a reproducible problem
title: "[Bug] "
labels: bug
assignees: ""
---

## What happened?

Describe the problem and what you expected instead.

## Steps to reproduce

1.
2.
3.

## Environment

- OS:
- Browser:
- Extension version:
- Python version:
- GPU / CPU:
- Helper `/health` output with secrets removed:

## Helper output

Paste relevant helper logs only after removing local paths, cookies, tokens, and
private video/audio details.

## Verification tried

- [ ] Reloaded the unpacked extension
- [ ] Hard-reloaded the YouTube tab
- [ ] Restarted `python server.py`
- [ ] Ran `python -m py_compile audio.py server.py cache.py transcribe.py translate_llm.py denoise.py diarize.py`
