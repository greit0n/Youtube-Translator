# Roadmap — top 15 things to add next

_Last updated: 2026-06-14_

Goal: ship this as a **free, local, anyone-can-run** YouTube → English (and
beyond) live translator. The list is ranked by impact for a public release,
blending **reach** (people who aren't on an RTX 5070 Ti), **quality** (better
voice detection / cleaner captions), and **friction** (getting non-technical
users running at all).

Each item: **What · Why · How (in this stack) · Effort** (S ≈ hours,
M ≈ a day or two, L ≈ multi-day).

---

## 1. Background noise & music suppression (vocal isolation)
- **What:** Strip music/SFX/ambience before Whisper sees the audio, so speech is
  cleaner and detection is far more reliable on noisy videos.
- **Why:** Music beds and crowd noise are the #1 cause of garbage/missing
  captions. You called this out — it's high value and visible.
- **How:** Add a preprocessing step in `audio.fetch_window` output → run a
  denoiser/vocal-isolator on the 16 kHz WAV before `transcribe`. Options by
  cost: **DeepFilterNet** or **RNNoise** (light, fast, good for ambient noise);
  **Demucs `htdemucs` vocals stem** (heavy, GPU, best for music-heavy content).
  Make it a settings toggle ("Clean audio: off / light / music") so low-end
  machines can skip it.
- **Effort:** M (light denoise) → L (Demucs path + perf tuning).

## 2. Speaker diarization ("voice fingerprint") + speaker-labeled captions
- **What:** Detect *who* is speaking and tag/colour captions per speaker
  (Speaker 1 / 2 …), optionally with persistent voice fingerprints across a
  video.
- **Why:** Interviews, podcasts, debates become readable. Your example; pairs
  naturally with #1.
- **How:** Strongly consider migrating the transcription core to **WhisperX**
  (wraps faster-whisper + word-level alignment + **pyannote** diarization) — it
  gives #2 *and* #5 in one move. Pyannote needs a (free) HF token; gate the
  feature behind a setting. Emit a `speaker` field on `segment` frames; render
  per-speaker colour in `content.js`.
- **Effort:** L (diarization is the heaviest single feature here).

## 3. Hardware-adaptive model selection (run on any GPU/CPU)
- **What:** Auto-pick the Whisper model to fit the user's hardware instead of
  hard-coding `large-v3`.
- **Why:** Right now this *requires* a strong GPU. "Free for everybody" means it
  must run on a 6 GB GPU, a laptop iGPU, or CPU — even if slower/smaller.
- **How:** In `transcribe._detect_device`, probe VRAM (via `torch`/`nvml`) and
  choose: `large-v3` (≥10 GB) → `distil-large-v3` / `medium` (≥6 GB) →
  `small`/`base` int8 on CPU. Surface an override in the popup ("Quality:
  Auto / Max / Balanced / Lite"). Keep `large-v3` as the default when it fits.
- **Effort:** M.

## 4. One-click installer / bundled helper (no Python setup)
- **What:** A single download that runs the helper without the user installing
  Python, pip deps, or ffmpeg.
- **Why:** The current setup (Python + CUDA + ffmpeg + Ollama + cookies) is a
  wall for normal users. This is the difference between "10 people use it" and
  "10,000 people use it."
- **How:** **PyInstaller** one-file build of `helper/`, bundle ffmpeg via
  `imageio-ffmpeg` (or ship the binary), auto-download the model on first run
  with a progress UI. Provide signed Windows build first (your platform), then
  macOS/Linux. Pair with #5 (tray app).
- **Effort:** L.

## 5. First-run onboarding wizard + system-tray helper
- **What:** A guided first-run that checks ffmpeg / CUDA / Ollama / cookies and
  tells the user exactly what's missing; the helper lives in the system tray and
  auto-starts.
- **Why:** Removes the "is the helper running? why no captions?" confusion. The
  popup already shows health — turn that into actionable setup.
- **How:** Expand `GET /health` into a `GET /diagnostics` (ffmpeg present, CUDA
  ok, model state, Ollama, cookie validity/expiry). New onboarding page in the
  extension reads it and links to fixes in `how-to.html`. Wrap the helper in
  **pystray** (or a tiny Tauri shell) for tray + autostart.
- **Effort:** M–L.

## 6. Translate to ANY language, not just English
- **What:** Let users pick their output language (Spanish, Hindi, Arabic, …),
  not only English.
- **Why:** "For everybody" is mostly non-English speakers. Whisper's built-in
  `translate` only targets English — the Ollama path is the unlock.
- **How:** Generalize the `ollama` engine: `transcribe` faithfully, then prompt
  the LLM to translate into the chosen **target language** (new `targetLang`
  field in the WS start message + popup selector). Keep Whisper-translate as the
  fast English-only path. Document the quality/latency trade-off.
- **Effort:** M.

## 7. Whisper hallucination & repetition filtering
- **What:** Suppress the classic phantom lines ("Thank you for watching", looped
  phrases) Whisper invents over silence/music.
- **Why:** These erode trust instantly and look broken in a public release.
- **How:** faster-whisper exposes `no_speech_prob`, `avg_logprob`,
  `compression_ratio` per segment — drop segments past tuned thresholds; add an
  n-gram repetition detector; tighten `vad_filter` params. Centralize in
  `transcribe.py` so both engines benefit. Pairs with #1.
- **Effort:** S–M.

## 8. Dual subtitles (original + translation)
- **What:** Optionally show the source-language line above the English/target
  line.
- **Why:** Huge for language learners — a major audience for a free tool. Low
  cost once the `ollama` engine carries source text.
- **How:** The `transcribe_source` path already keeps the source transcript;
  send both `text` and `srcText` on `segment` frames and render two stacked
  lines in `content.js`/`overlay.css`. Toggle in popup.
- **Effort:** S–M.

## 9. Transcript panel + export (SRT / VTT / TXT)
- **What:** A scrollable transcript with timestamps you can read, copy, and
  download as subtitles.
- **Why:** Turns the tool into something people keep around (study, clipping,
  accessibility) and is very shareable.
- **How:** `content.js` already buffers all segments; add a togglable side panel
  and an export that serializes the buffer to SRT/VTT. Optional `GET
  /transcript/{videoId}` from the existing cache for full-video export.
- **Effort:** M.

## 10. Caption styling, size & position controls
- **What:** User control over font, colour, background opacity, and a
  draggable/position setting; remember per-user.
- **Why:** Accessibility (low vision, colour needs) and not blocking YouTube's
  own UI. Cheap polish that makes it feel finished.
- **How:** Extend the existing font-size setting in `overlay.css` +
  popup; persist to `chrome.storage.sync`. Add drag-to-move with a saved offset.
- **Effort:** S–M.

## 11. Names/terms glossary UI (hotwords + consistent translations)
- **What:** Let users add names, jargon, and game terms so they're recognized
  and translated consistently.
- **Why:** Proper nouns are where both Whisper and the LLM stumble; a glossary is
  a big perceived-quality win.
- **How:** The WS contract already has a `hotwords` field (unused in the UI).
  Add a popup editor → pass through to faster-whisper `hotwords`/`initial_prompt`
  and inject a glossary block into the Ollama translation prompt.
- **Effort:** S–M.

## 12. Word-level timestamps + karaoke highlighting
- **What:** Word-accurate timing and current-word highlight for tight sync.
- **Why:** Makes captions feel "live" instead of arriving in blocks; also
  improves the auto-pause gate's precision.
- **How:** `word_timestamps=True` in faster-whisper (free if you adopt WhisperX
  in #2). Send word arrays on segments; animate in `content.js`.
- **Effort:** M (S if WhisperX already in).

## 13. Robust cookie health + guided refresh
- **What:** Detect expired/rotated cookies up front and walk the user through
  re-exporting them — instead of the silent mid-session freeze.
- **Why:** Cookies are *the* recurring failure mode (sequential-URL freeze).
  For a public tool this must be self-diagnosing, not tribal knowledge.
- **How:** Validate cookies on `/diagnostics` (a cheap authenticated probe);
  show expiry + a one-click "how to refresh" flow in the popup; detect the
  cookie→anon fallback at runtime and surface a clear "cookies expired" message
  instead of a stuck spinner.
- **Effort:** M.

## 14. Work beyond YouTube (Twitch, Vimeo, generic HTML5 video)
- **What:** Run the same overlay on other video sites.
- **Why:** Multiplies the audience and usefulness with mostly-existing code.
- **How:** Generalize player detection in `content.js` (it already finds a
  `<video>`); widen `manifest.json` `matches`; per-site adapters for video-ID /
  audio-source resolution (Twitch VODs, Vimeo). yt-dlp already supports many
  sites for `audio.extract_info`.
- **Effort:** M–L per platform.

## 15. Live mic / system-audio mode (translate anything)
- **What:** Translate live audio from the mic or system output — calls,
  meetings, any app — not just web video.
- **Why:** Turns a YouTube add-on into a general live-translation utility; strong
  word-of-mouth feature.
- **How:** New capture path in the helper using WASAPI loopback on Windows
  (`pyaudiowpatch`/`soundcard`); stream chunks straight into the existing
  windowed transcribe loop (skip yt-dlp/ffmpeg-URL). A small desktop/overlay UI
  instead of the YouTube content script.
- **Effort:** L.

---

## Also considered (parking lot)
- Confidence-based caption styling (dim low-confidence words).
- Clean-language toggle (the engines already preserve profanity by design).
- Auto-resume / reconnection hardening and clearer error surfacing.
- i18n of the extension UI itself.
- Optional shareable transcript links (note: no server-side storage — keep it
  local-first to stay free and private).

## Notes / guardrails
- **Stay local-first & free.** No paid APIs, no central servers, no telemetry —
  it's the whole pitch. Anything cloud-shaped should be opt-in and local by
  default.
- **Protect low-end users.** Every heavy feature (#1 Demucs, #2 diarization)
  must be a toggle that degrades gracefully, with `large-v3` quality preserved
  where the hardware allows.
- **Keep the WS contract in lockstep.** New `segment`/start fields (`speaker`,
  `srcText`, `targetLang`, words) must be added in both `server.py` and
  `content.js` together (see CLAUDE.md "WS protocol").
