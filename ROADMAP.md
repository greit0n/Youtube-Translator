# Roadmap - top 15 things to add next

_Last updated: 2026-06-14_

Goal: ship this as a free, local, broadly runnable YouTube to English subtitle
translator. The list is ranked by public-release impact across reach, quality,
and setup friction.

Each item: What, why, how, and effort. Effort rough scale: S = hours, M = one or
two days, L = multi-day.

---

## 1. Background noise and music suppression
- **What:** Strip music, SFX, and ambient noise before Whisper sees the audio.
- **Why:** Noise and music are common causes of missing or low-quality captions.
- **How:** Run a preprocessing step after `audio.fetch_window`: DeepFilterNet or
  RNNoise for light noise, Demucs vocals stem for music-heavy content. Keep it
  behind the existing "Clean audio: off / light / music" setting.
- **Effort:** M for light denoise, L for the Demucs path and tuning.

## 2. Speaker diarization and speaker-labeled captions
- **What:** Detect who is speaking and tag captions as Speaker 1, Speaker 2,
  etc., with stable labels across a video.
- **Why:** Interviews, podcasts, debates, and multiplayer videos are easier to
  follow with speaker labels.
- **How:** Continue building on WhisperX and pyannote. Gate it behind a setting,
  emit `speaker` on `segment` frames, and render per-speaker color in
  `content.js`.
- **Effort:** L.

## 3. Hardware-adaptive model selection
- **What:** Pick the Whisper model based on available hardware.
- **Why:** Public users will have a wide range of GPUs and CPUs.
- **How:** Probe VRAM and choose `large-v3`, `medium`, `small`, or `base`.
  Preserve manual Quality overrides in the popup.
- **Effort:** M.

## 4. One-click installer / bundled helper
- **What:** Ship a download that runs the helper without manual Python setup.
- **Why:** Python, CUDA, ffmpeg, Ollama, and cookies are a lot for normal users.
- **How:** Package the helper with PyInstaller, include or bootstrap ffmpeg, and
  download models on first run with visible progress.
- **Effort:** L.

## 5. First-run onboarding and tray helper
- **What:** Guided setup checks for ffmpeg, CUDA, model state, Ollama, cookies,
  and optional tokens.
- **Why:** Users should get actionable setup errors instead of silent failure.
- **How:** Add `/diagnostics`, expand popup onboarding, and wrap the helper in a
  system-tray process with optional autostart.
- **Effort:** M-L.

## 6. Translate to any target language
- **What:** Let users pick target languages beyond English.
- **Why:** The Ollama path can translate into many languages.
- **How:** Add `targetLang` to the WebSocket start message and popup, then prompt
  the LLM to translate source transcripts into that language. Keep Whisper direct
  mode English-only.
- **Effort:** M.

## 7. Whisper hallucination and repetition filtering
- **What:** Suppress phantom lines and repeated phrases over silence or music.
- **Why:** Hallucinated captions erode user trust quickly.
- **How:** Filter with `no_speech_prob`, `avg_logprob`, `compression_ratio`, and
  an n-gram repetition detector in `transcribe.py`.
- **Effort:** S-M.

## 8. Dual subtitles
- **What:** Optionally show source text above translated text.
- **Why:** Useful for language learning and debugging translation quality.
- **How:** Send both translated text and source text on `segment` frames, then
  render two stacked lines in the overlay.
- **Effort:** S-M.

## 9. Transcript panel and export
- **What:** Add a transcript panel and export SRT, VTT, or TXT.
- **Why:** Makes the tool useful for study, clipping, and accessibility.
- **How:** Serialize the content script's segment buffer, with an optional cache
  read endpoint for full-video export.
- **Effort:** M.

## 10. Caption styling, size, and position controls
- **What:** Let users control font, color, background opacity, and position.
- **Why:** Accessibility and non-overlap with YouTube UI matter for real use.
- **How:** Extend existing popup settings and persist style values in
  `chrome.storage.sync`.
- **Effort:** S-M.

## 11. Names and terms glossary UI
- **What:** Let users add names, jargon, and preferred translations.
- **Why:** Proper nouns and domain terms are common failure points.
- **How:** Continue sending glossary entries in the WebSocket start message and
  inject them into Ollama prompts; use hotwords carefully only where they improve
  ASR.
- **Effort:** S-M.

## 12. Word-level timestamps and karaoke highlighting
- **What:** Add word timing and current-word highlighting.
- **Why:** Makes subtitles feel more live and improves playback gating.
- **How:** Use WhisperX word-level data, send word arrays on segments, and
  animate current words in `content.js`.
- **Effort:** M.

## 13. Robust cookie health and guided refresh
- **What:** Detect expired or rotated cookies and guide users through refresh.
- **Why:** Cookie issues are a common YouTube failure mode.
- **How:** Add a cheap authenticated `/diagnostics` probe, show cookie state in
  the popup, and surface a clear refresh message when authenticated extraction
  fails.
- **Effort:** M.

## 14. Work beyond YouTube
- **What:** Support Twitch, Vimeo, and generic HTML5 video sites.
- **Why:** The overlay and helper architecture can apply beyond YouTube.
- **How:** Generalize player detection and audio-source resolution. Add per-site
  adapters where `yt-dlp` metadata differs.
- **Effort:** M-L per platform.

## 15. Live mic / system-audio mode
- **What:** Translate live microphone or system output.
- **Why:** This turns the project into a general live-translation utility.
- **How:** Add a WASAPI loopback capture path on Windows and feed chunks into
  the existing transcription loop without the YouTube content script.
- **Effort:** L.

---

## Parking Lot
- Confidence-based caption styling.
- Clean-language toggle.
- Reconnection hardening and clearer error surfacing.
- Extension UI localization.
- Optional shareable transcript links, with local-first privacy preserved.

## Guardrails
- Stay local-first and free: no paid APIs, central servers, or telemetry.
- Keep heavy features optional and graceful when dependencies are missing.
- Keep the WebSocket contract in lockstep between `server.py` and `content.js`.
  New start fields or segment fields must be documented in `AGENTS.md` and
  `helper/README.md`.
