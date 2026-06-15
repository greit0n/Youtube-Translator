# Subtitle-aware playback gate — design

**Date:** 2026-06-14
**Status:** approved (Approach B)

## Goal

Auto-pause the YouTube video whenever the current playback moment has **not been
transcribed yet**, and auto-resume once a translated subtitle covering "now" is
ready. Behaves like buffering, but for subtitles. This eliminates the "plays a
few seconds, then I'm watching with no/lagging captions" experience.

## Core problem

"No caption on screen" ≠ "not ready." Silent stretches (intros, music, pauses)
produce **no subtitle** but are still processed by the helper. The gate must
distinguish:

- **not-yet-transcribed** at `currentTime` → pause
- **transcribed but silent** at `currentTime` → play (blank caption, as today)

The reliable signal is the helper's actual **coverage intervals**, mirrored on
the client. (Approach A — a single `max(progress.until)` frontier — was rejected
because it mis-handles seeking backward into a skipped gap, and seeking is
central to this tool.)

## Behavior (agreed)

- **Trigger:** pause when `!isCovered(currentTime)`. Applies at session start /
  seek **and** mid-playback underrun (uniform rule).
- **Resume:** when coverage reaches `currentTime + RESUME_MARGIN` (~1.5s
  hysteresis, to avoid pause/play flicker) — only if we were the one who paused.
- **Stuck case:** if subtitles never arrive (throttle/cookie freeze), stay
  paused with an on-screen `⏳ waiting for subtitles…` message **indefinitely**.
  Manual play always overrides.
- **Never fight the user:** if the user manually pauses, we never auto-resume it.
  If the user manually presses play while we're holding, we back off (stop gating
  until coverage next catches up to the playhead, then re-arm).
- **Toggle:** new `autoPause` setting (default ON). When OFF, behaves exactly as
  today (no pause/resume).

## Protocol changes (server ↔ content, keep in lockstep)

1. **Enrich `progress`:** `{type:"progress", until}` → `{type:"progress", start,
   until}` so the client can rebuild covered intervals. (`server.py:487`)
2. **Send cached coverage on start:** on a cache hit, after the `cached` status,
   emit the cached `covered` intervals so the gate knows already-done (incl.
   silent) regions. Implementation: one `progress {start, until}` frame per
   cached interval (no new message type needed). (`server.py:~289`)

No other server logic changes; the windowed loop is untouched.

## Content script (`content.js`) — new "playback gate" unit

Self-contained; reads coverage + `video.currentTime`, calls `pause/play`, renders
a status. Does not touch the socket or segment-rendering paths.

State (on the session object `s`):
- `coveredIntervals: [[start, until], ...]` — merged from `progress` frames.
- `pausedByUs: boolean` — true when the gate issued the pause (vs the user).
- `userOverride: boolean` — set when the user plays through a hold; suppresses
  re-pausing until `isCovered(currentTime)` becomes true again, then cleared.

Functions:
- `mergeCovered(s, start, until)` — insert+merge into `coveredIntervals`.
- `isCovered(s, t)` — interval membership test.
- `updateGate(s)` — the decision, called from `timeupdate`/`seeking` and on each
  `progress`:
  - gate disabled (`!settings.autoPause`) → no-op.
  - `userOverride` and still uncovered → leave playing.
  - `!isCovered(now)` and playing → `video.pause()`, `pausedByUs=true`, show
    `⏳ waiting for subtitles…`.
  - covered to `now + RESUME_MARGIN` and `pausedByUs` → `video.play()`,
    clear `pausedByUs`.

User-intent detection (bind once per session):
- We wrap our own `pause()/play()` calls with an internal `selfAction` flag.
- `video` `pause` event without `selfAction` → user paused → clear `pausedByUs`
  (we won't auto-resume).
- `video` `play` event without `selfAction` while uncovered → `userOverride=true`.

Constants: `RESUME_MARGIN = 1.5` (s).

## Settings plumbing

- `popup.html` / `popup.js`: add an `autoPause` toggle (label, like the others).
- `background.js`: seed `autoPause: true` default.
- `content.js`: read `autoPause` in `loadSettingsThenInit` + react in
  `chrome.storage.onChanged`. **No reinit needed** — it's a client-only behavior
  toggle (unlike engine/model/language/preBuffer). Just enable/disable the gate;
  if turning off while holding, resume if `pausedByUs`.

## Files touched

- `helper/server.py` — enrich `progress`, send cached coverage on start.
- `extension/content.js` — playback gate unit + settings wiring.
- `extension/popup.html`, `extension/popup.js` — `autoPause` toggle.
- `extension/background.js` — default seed.
- `extension/manifest.json` — bump `version`.
- `AGENTS.md` (WS protocol section) + `helper/README.md` — document enriched
  `progress` frame.

## Edge cases

- **Cache hit, silent region at start:** covered intervals sent on start → gate
  sees it covered → plays. Without this it would wrongly hold.
- **Backward seek into covered region:** `isCovered` true → no pause.
- **Forward seek into uncovered region:** uncovered → pause until filled.
- **Gate toggled off mid-hold:** resume if `pausedByUs`.
- **Session teardown while holding:** nothing special — we never persist a pause;
  a fresh session re-evaluates.
- **Orphaned content script:** existing watchdog tears down the session; the gate
  dies with it (no leftover paused state since the page video keeps its own).
