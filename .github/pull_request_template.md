## Summary

What changed?

## Verification

- [ ] Helper syntax check:
  `python -m py_compile audio.py server.py cache.py transcribe.py translate_llm.py denoise.py diarize.py`
- [ ] WebSocket smoke test, if helper behavior changed
- [ ] Browser extension check, if extension behavior changed

## Docs

- [ ] Setup docs updated, if requirements or user-facing behavior changed
- [ ] `extension/manifest.json` version bumped, if extension behavior changed

## Secret Hygiene

- [ ] No `cookies.txt`, `hf_token.txt`, cache files, logs, enrolled voice clips,
      or local private artifacts included
