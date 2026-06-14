// content.js — core of the YouTube → English subtitle overlay.
//
// Lifecycle overview:
//   - We detect the current video ID from the URL and (re)initialize on
//     YouTube SPA navigations.
//   - For each active video we open a WebSocket to the local helper, send a
//     start message with the current playback position + chosen language,
//     and buffer incoming "segment" messages.
//   - On each `timeupdate` we look up the segment covering the current time
//     and render it in an overlay appended inside the player container so it
//     tracks default / theater / fullscreen layouts.
//
// The whole thing is wrapped in an IIFE to avoid leaking globals onto the page.
(() => {
  "use strict";

  const HELPER_BASE = "http://127.0.0.1:8765";
  const HELPER_WS = "ws://127.0.0.1:8765/transcribe";

  // ---- Module state -------------------------------------------------------

  // Current settings, mirrored from chrome.storage.sync.
  let settings = {
    enabled: true,
    language: null, // null = auto-detect
    fontSize: "medium",
    engine: "whisper", // "whisper" (fast built-in) | "ollama" (faithful LLM)
    model: "qwen2.5:7b", // Ollama chat model (used when engine === "ollama")
    preBuffer: true, // ask the helper to look ahead / pre-buffer
    autoPause: true // pause playback until subtitles for "now" are ready
  };

  // How often we report playback position to the helper so it can keep its
  // lead-following window ~90s ahead and pre-buffer.
  const POSITION_INTERVAL_MS = 4000;

  // Playback gate: when auto-pause is on we hold the video whenever the current
  // moment hasn't been transcribed yet, and resume once coverage reaches a bit
  // past it (hysteresis avoids pause/play flicker on the boundary).
  const RESUME_MARGIN = 1.5; // seconds of covered cushion required to resume
  const COVER_EPS = 0.05; // float slop for interval membership
  // Seeks make YouTube fire its own play/pause events that aren't user intent;
  // ignore those for override detection for a short grace window after a seek.
  const SEEK_GRACE_MS = 1200;
  const WAIT_MSG = "⏳ Waiting for subtitles…";

  // The video session currently active. Encapsulates everything that must be
  // torn down when we navigate away or re-init: the socket, the buffered
  // segments, the bound listeners, and a generation token to guard against
  // stale async callbacks from a previous session.
  let session = null;

  // Monotonic token so that async work (socket events, reconnect timers)
  // belonging to an old session can detect that it has been superseded.
  let sessionGeneration = 0;

  // Overlay DOM. Created lazily and reused; re-parented if the player changes.
  let captionContainer = null;
  let captionEl = null;

  // Handle for the 1s URL-poll / orphan-watchdog interval (assigned at boot).
  let urlPollTimer = null;

  // ---- Utilities ----------------------------------------------------------

  // True while this script's extension context is still valid. After the user
  // reloads/updates the extension, Chrome severs the bridge to THIS already-
  // injected script (the page JS keeps running, but `chrome.runtime.id` throws
  // or goes undefined). We use this to detect that we've been orphaned and
  // clean up our own overlay instead of leaving stale captions frozen on screen.
  function isExtensionAlive() {
    try {
      return !!(chrome && chrome.runtime && chrome.runtime.id);
    } catch (_e) {
      return false;
    }
  }

  // One-shot self-destruct for an orphaned script: drop the overlay, stop all
  // timers/listeners, and never touch chrome.* again.
  let selfDestructed = false;
  function selfDestructIfOrphaned() {
    if (selfDestructed || isExtensionAlive()) return false;
    selfDestructed = true;
    try {
      teardownSession();
    } catch (_e) {
      /* chrome.* may already be dead; ignore */
    }
    removeOverlay();
    if (urlPollTimer) clearInterval(urlPollTimer);
    return true;
  }

  // Extract the ?v= video id from the current location. Returns null if none.
  function getVideoIdFromUrl() {
    try {
      const params = new URLSearchParams(window.location.search);
      return params.get("v");
    } catch (_e) {
      return null;
    }
  }

  // Find the player container and the <video> element. YouTube uses
  // .html5-video-player as the outer player; the <video> lives inside it.
  function findPlayerElements() {
    const player = document.querySelector(".html5-video-player");
    const video = player ? player.querySelector("video.html5-main-video, video") : null;
    return { player, video };
  }

  // Poll for the player/video to appear (it may not exist immediately after
  // navigation). Resolves with {player, video} or null on timeout.
  function waitForPlayer(timeoutMs = 15000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const tick = () => {
        const found = findPlayerElements();
        if (found.player && found.video) {
          resolve(found);
          return;
        }
        if (Date.now() - start > timeoutMs) {
          resolve(null);
          return;
        }
        setTimeout(tick, 300);
      };
      tick();
    });
  }

  // ---- Overlay rendering --------------------------------------------------

  // Ensure the overlay DOM exists and is parented inside the given player.
  function ensureOverlay(player) {
    if (!captionContainer) {
      captionContainer = document.createElement("div");
      captionContainer.className = "ytx-caption-container";
      captionEl = document.createElement("div");
      captionEl.className = "ytx-caption";
      captionContainer.appendChild(captionEl);
    }
    // Re-parent if needed (player element may be recreated by YouTube).
    if (player && captionContainer.parentElement !== player) {
      player.appendChild(captionContainer);
    }
    applyFontSizeClass();
  }

  // Apply the font-size class from settings to the container.
  function applyFontSizeClass() {
    if (!captionContainer) return;
    captionContainer.classList.remove(
      "ytx-size-small",
      "ytx-size-medium",
      "ytx-size-large"
    );
    const size = ["small", "medium", "large"].includes(settings.fontSize)
      ? settings.fontSize
      : "medium";
    captionContainer.classList.add("ytx-size-" + size);
  }

  // Render plain caption text (subtitle). Empty string clears the caption.
  function renderCaption(text) {
    if (!captionEl) return;
    captionEl.classList.remove("ytx-error", "ytx-status");
    captionEl.textContent = text || "";
  }

  // Render a status/error message that persists until replaced.
  function renderStatus(text, isError) {
    if (!captionEl) return;
    captionEl.classList.remove("ytx-error", "ytx-status");
    captionEl.classList.add(isError ? "ytx-error" : "ytx-status");
    captionEl.textContent = text || "";
  }

  // Remove the overlay entirely from the DOM. Removes the module-tracked
  // container AND defensively sweeps the whole document for any stray
  // `.ytx-caption-container` nodes — e.g. one left behind by a previous
  // (now-orphaned) injection of this script after an extension reload.
  function removeOverlay() {
    if (captionContainer && captionContainer.parentElement) {
      captionContainer.parentElement.removeChild(captionContainer);
    }
    try {
      document
        .querySelectorAll(".ytx-caption-container")
        .forEach((el) => el.remove());
    } catch (_e) {
      /* ignore */
    }
  }

  // Keep a fullscreen marker on <html> so CSS can adapt the overlay.
  function syncFullscreenClass() {
    const fs = !!document.fullscreenElement;
    document.documentElement.classList.toggle("ytx-fullscreen", fs);
  }

  // ---- Segment buffer + lookup --------------------------------------------

  // Insert a segment keeping the array sorted by `start`. Segments usually
  // arrive in order, so we optimize for the append case.
  function insertSegment(buffer, seg) {
    if (buffer.length === 0 || seg.start >= buffer[buffer.length - 1].start) {
      buffer.push(seg);
      return;
    }
    // Binary search for the insertion point.
    let lo = 0;
    let hi = buffer.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (buffer[mid].start < seg.start) lo = mid + 1;
      else hi = mid;
    }
    buffer.splice(lo, 0, seg);
  }

  // Find the segment whose [start, end) contains `t`. Binary search on start.
  function findSegmentAt(buffer, t) {
    let lo = 0;
    let hi = buffer.length - 1;
    let candidate = -1;
    // Find the rightmost segment with start <= t.
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (buffer[mid].start <= t) {
        candidate = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (candidate === -1) return null;
    const seg = buffer[candidate];
    return t < seg.end ? seg : null;
  }

  // ---- Session management -------------------------------------------------

  // Tear down the active session: close socket, clear timers/listeners.
  function teardownSession() {
    if (!session) return;
    const s = session;
    session = null;
    sessionGeneration++; // invalidate any pending async callbacks

    if (s.reconnectTimer) clearTimeout(s.reconnectTimer);
    // Stop the periodic position reporter for this session.
    if (s.positionTimer) {
      clearInterval(s.positionTimer);
      s.positionTimer = null;
    }
    if (s.video && s.onTimeUpdate) {
      s.video.removeEventListener("timeupdate", s.onTimeUpdate);
      s.video.removeEventListener("seeking", s.onTimeUpdate);
    }
    // Remove the gate's play/pause/seek listeners and never leave a video stuck
    // on a pause WE issued — resume it so the page isn't frozen after teardown.
    if (s.video && s.onPause) s.video.removeEventListener("pause", s.onPause);
    if (s.video && s.onPlay) s.video.removeEventListener("play", s.onPlay);
    if (s.video && s.onSeekGate) {
      s.video.removeEventListener("seeking", s.onSeekGate);
      s.video.removeEventListener("seeked", s.onSeekGate);
    }
    if (s.video && s.pausedByUs) {
      try {
        const p = s.video.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } catch (_e) {
        /* ignore */
      }
    }
    // Remove the position-reporting seek listeners.
    if (s.video && s.onSeekPosition) {
      s.video.removeEventListener("seeking", s.onSeekPosition);
      s.video.removeEventListener("seeked", s.onSeekPosition);
    }
    if (s.socket) {
      try {
        // Detach handlers so a close event doesn't trigger a reconnect.
        s.socket.onopen = null;
        s.socket.onmessage = null;
        s.socket.onerror = null;
        s.socket.onclose = null;
        if (
          s.socket.readyState === WebSocket.OPEN ||
          s.socket.readyState === WebSocket.CONNECTING
        ) {
          s.socket.close();
        }
      } catch (_e) {
        /* ignore */
      }
    }
    renderCaption("");
  }

  // ---- Position reporting (lead-following) --------------------------------

  // Send a single {type:"position"} frame if the socket is OPEN. Guarded by
  // the generation token so stale sessions never write to a live socket.
  function sendPosition(s) {
    if (s.generation !== sessionGeneration) return;
    const sock = s.socket;
    if (!sock || sock.readyState !== WebSocket.OPEN) return;
    if (!s.video || !Number.isFinite(s.video.currentTime)) return;
    try {
      sock.send(
        JSON.stringify({ type: "position", currentTime: s.video.currentTime })
      );
    } catch (_e) {
      /* will surface via onerror/onclose */
    }
  }

  // Start (or restart) the periodic position reporter for this session, plus
  // immediate updates on seek. Called from onopen so it follows the socket
  // lifecycle; it keeps running for the life of the session (we do NOT stop
  // once segments arrive — this drives the helper's look-ahead).
  function startPositionReporting(s) {
    // Clear any prior timer (e.g. from a previous reconnect) before re-arming.
    if (s.positionTimer) {
      clearInterval(s.positionTimer);
      s.positionTimer = null;
    }
    // Send one immediately so the helper learns our position without waiting.
    sendPosition(s);
    s.positionTimer = setInterval(() => {
      if (s.generation !== sessionGeneration) return;
      sendPosition(s);
    }, POSITION_INTERVAL_MS);

    // Bind seek listeners once per session for immediate position updates.
    if (!s.onSeekPosition) {
      s.onSeekPosition = () => {
        if (s.generation !== sessionGeneration) return;
        sendPosition(s);
      };
      s.video.addEventListener("seeking", s.onSeekPosition);
      s.video.addEventListener("seeked", s.onSeekPosition);
    }
  }

  // Open (or reopen) the WebSocket for the given session.
  function connectSocket(s) {
    const myGen = s.generation;
    let socket;
    try {
      socket = new WebSocket(HELPER_WS);
    } catch (_e) {
      scheduleReconnect(s);
      renderStatus("⚠ Translator helper not running on :8765", true);
      return;
    }
    s.socket = socket;

    socket.onopen = () => {
      if (myGen !== sessionGeneration) return; // superseded
      s.reconnectDelay = 1000; // reset backoff on a good connection
      // Send the start message with the current playback position so the
      // helper can process from where we are (and stay ahead).
      const startTime =
        s.video && Number.isFinite(s.video.currentTime) ? s.video.currentTime : 0;
      const startMsg = {
        videoId: s.videoId,
        startTime: startTime,
        language: settings.language, // null => auto-detect
        engine: settings.engine, // "ollama" | "whisper"
        model: settings.model, // Ollama chat model name
        preBuffer: settings.preBuffer, // bool: helper look-ahead/pre-buffer
        hotwords: null // reserved; not used by the client yet
      };
      try {
        socket.send(JSON.stringify(startMsg));
      } catch (_e) {
        /* will surface via onerror/onclose */
      }
      s.statusText = WAIT_MSG;
      renderStatus(s.statusText, false);
      // Begin streaming playback position so the helper can stay ahead.
      startPositionReporting(s);
    };

    socket.onmessage = (event) => {
      if (myGen !== sessionGeneration) return;
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (_e) {
        return; // ignore malformed frames
      }
      handleHelperMessage(s, msg);
    };

    socket.onerror = () => {
      if (myGen !== sessionGeneration) return;
      // onerror is typically followed by onclose; show a hint now.
      renderStatus("⚠ Translator helper not running on :8765", true);
    };

    socket.onclose = () => {
      if (myGen !== sessionGeneration) return;
      s.socket = null;
      // Reconnect only if the helper hasn't finished AND didn't send a hard
      // error (e.g. age-restricted / can't resolve). Reconnecting on those just
      // hammers YouTube every few seconds and risks IP throttling — the user
      // re-triggers via reload or the ♻ Re-translate button after fixing it.
      if (!s.done && !s.gotError) {
        scheduleReconnect(s);
      }
    };
  }

  // Map the helper's raw status phases to ONE friendly, emoji'd overlay line so
  // the user sees a single consistent "waiting" message instead of a parade of
  // internal phase names (resolving/transcribing/…). Informative states keep
  // their own clear message.
  function friendlyStatus(raw) {
    const m = String(raw || "").toLowerCase();
    if (m.includes("throttl")) return "🐢 YouTube is throttling — buffering…";
    if (m.includes("ollama")) return "🔁 Switching to fast translation…";
    if (m.includes("model")) return "⏳ Warming up the model…";
    // resolving stream / transcribing / cached / anything else generic:
    return WAIT_MSG;
  }

  // Handle a parsed message from the helper.
  function handleHelperMessage(s, msg) {
    if (!msg || typeof msg !== "object") return;
    switch (msg.type) {
      case "status":
        // Remember the latest status (mapped to one friendly message); it stays
        // on screen whenever there's no caption at the current moment.
        s.statusText = friendlyStatus(msg.message);
        updateCaptionForCurrentTime(s);
        break;
      case "segment":
        if (
          typeof msg.start === "number" &&
          typeof msg.end === "number" &&
          typeof msg.text === "string"
        ) {
          insertSegment(s.segments, {
            start: msg.start,
            end: msg.end,
            text: msg.text
          });
          // Show the new segment if it covers now (else status stays).
          updateCaptionForCurrentTime(s);
        }
        break;
      case "progress":
        // Coverage frontier advanced (or cached coverage replayed at start).
        // Track it so the playback gate knows what's transcribed, and re-evaluate
        // (this is what resumes playback once "now" becomes covered).
        if (typeof msg.until === "number") {
          const start = typeof msg.start === "number" ? msg.start : msg.until;
          mergeCovered(s, start, msg.until);
          updateGate(s);
        }
        break;
      case "done":
        s.done = true;
        break;
      case "error":
        // Hard error from the helper (e.g. age-restricted, can't resolve).
        // Stop the auto-reconnect loop so we don't hammer YouTube.
        s.gotError = true;
        renderStatus(
          "⚠ " + String(msg.message || "Translator error"),
          true
        );
        break;
      default:
        break; // unknown type — ignore
    }
  }

  // Schedule a reconnect with capped exponential backoff. Avoids spamming.
  function scheduleReconnect(s) {
    if (s.generation !== sessionGeneration) return;
    if (s.reconnectTimer) return; // already scheduled
    const delay = Math.min(s.reconnectDelay || 1000, 15000);
    s.reconnectTimer = setTimeout(() => {
      s.reconnectTimer = null;
      if (s.generation !== sessionGeneration) return;
      s.reconnectDelay = Math.min((s.reconnectDelay || 1000) * 2, 15000);
      connectSocket(s);
    }, delay);
  }

  // ---- Playback gate (subtitle-aware pause/resume) ------------------------

  // Merge [start, until] into the session's covered intervals, kept sorted and
  // non-overlapping. Coverage is fed by the helper's `progress` frames (incl.
  // the cached intervals it replays on session start).
  function mergeCovered(s, start, until) {
    if (!(until > start)) return;
    const ivs = s.coveredIntervals;
    ivs.push([start, until]);
    ivs.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const iv of ivs) {
      const last = merged[merged.length - 1];
      if (last && iv[0] <= last[1] + COVER_EPS) {
        if (iv[1] > last[1]) last[1] = iv[1];
      } else {
        merged.push([iv[0], iv[1]]);
      }
    }
    s.coveredIntervals = merged;
  }

  // End of the covered interval containing `t`, or null if `t` isn't covered.
  function coveredEndAt(s, t) {
    for (const [a, b] of s.coveredIntervals) {
      if (t >= a - COVER_EPS && t < b + COVER_EPS) return b;
      if (a > t) break; // sorted; no later interval can contain t
    }
    return null;
  }

  // Our own pause/play, flagged so the play/pause event handlers can tell our
  // actions apart from the user's. We only act when it actually changes state
  // (calling pause() on a paused video fires no event, which would desync the
  // flag), so each self-action maps to exactly one consumed event.
  function gatePause(s) {
    if (!s.video || s.video.paused) return;
    s.selfAction = true;
    s.video.pause();
  }
  function gatePlay(s) {
    if (!s.video || !s.video.paused) return;
    s.selfAction = true;
    const p = s.video.play();
    // On success the 'play' event consumes selfAction; if play() is rejected
    // (autoplay policy) no event fires, so clear the flag here to avoid desync.
    if (p && typeof p.catch === "function") {
      p.catch(() => {
        s.selfAction = false;
      });
    }
  }

  // The decision, written as a RECONCILER between desired and actual state so a
  // stray YouTube play/pause around a seek can't leave us wedged: we re-derive
  // what we want every call rather than trusting accumulated flags.
  // Called on timeupdate, seek, and whenever coverage grows.
  function updateGate(s) {
    if (!settings.autoPause) return;
    if (!s || s.generation !== sessionGeneration || !s.video) return;
    const t = s.video.currentTime;
    if (!Number.isFinite(t)) return;

    const covEnd = coveredEndAt(s, t);
    const coveredNow = covEnd !== null;

    if (coveredNow) {
      // "Now" is transcribed → never hold here. Re-arm any override, and if we
      // were the one holding, resume once there's enough cushion (or near EOF).
      s.userOverride = false;
      if (s.pausedByUs) {
        const dur = Number.isFinite(s.video.duration) ? s.video.duration : Infinity;
        const enough = covEnd >= t + RESUME_MARGIN || covEnd >= dur - 0.1;
        if (enough) {
          s.pausedByUs = false;
          gatePlay(s);
          updateCaptionForCurrentTime(s);
        }
      }
      return;
    }

    // Uncovered. If the user deliberately played through the hold, leave it be.
    if (s.userOverride) return;

    // We want it held. Re-assert the pause even if pausedByUs is already set
    // (a stray play may have un-paused the video underneath us).
    if (!s.video.paused) {
      s.pausedByUs = true;
      gatePause(s);
      updateCaptionForCurrentTime(s); // shows WAIT_MSG while held
    } else if (s.pausedByUs) {
      updateCaptionForCurrentTime(s);
    }
  }

  // Look up and render the caption for the video's current time.
  function updateCaptionForCurrentTime(s) {
    if (!s || !s.video || !captionEl) return;
    // Don't overwrite a sticky error message.
    if (captionEl.classList.contains("ytx-error")) return;
    // While the gate is holding, the overlay always shows the waiting message.
    if (s.pausedByUs) {
      renderStatus(WAIT_MSG, false);
      return;
    }

    const t = s.video.currentTime;
    const seg = findSegmentAt(s.segments, t);
    if (seg) {
      if (captionEl.textContent !== seg.text || captionEl.classList.contains("ytx-status")) {
        renderCaption(seg.text);
      }
      return;
    }

    // No caption covers 'now'. If we're still WAITING for content here — nothing
    // transcribed yet, or the playhead is past everything we have (e.g. buffering
    // / throttled) — keep the latest status on screen so it's clear work is going
    // on. Only blank out during a normal short gap between already-buffered lines.
    const lastEnd = s.segments.length ? s.segments[s.segments.length - 1].end : -1;
    const waiting = lastEnd < t;
    if (waiting && s.statusText) {
      if (
        captionEl.textContent !== s.statusText ||
        !captionEl.classList.contains("ytx-status")
      ) {
        renderStatus(s.statusText, false);
      }
    } else if (captionEl.textContent !== "") {
      renderCaption("");
    }
  }

  // Start a fresh session for the current video.
  async function startSession() {
    if (!settings.enabled) return;
    const videoId = getVideoIdFromUrl();
    if (!videoId) return;

    const { player, video } = (await waitForPlayer()) || {};
    if (!player || !video) return;

    // If settings got disabled while we were waiting, bail.
    if (!settings.enabled) return;

    sessionGeneration++;
    const gen = sessionGeneration;

    const s = {
      generation: gen,
      videoId,
      player,
      video,
      segments: [],
      socket: null,
      reconnectTimer: null,
      reconnectDelay: 1000,
      done: false,
      onTimeUpdate: null,
      positionTimer: null, // setInterval handle for position reporting
      onSeekPosition: null, // seek listener that sends immediate positions
      coveredIntervals: [], // [[start, until], ...] transcribed regions (gate)
      pausedByUs: false, // true while the gate is holding playback
      userOverride: false, // user played through a hold; suppress re-pausing
      selfAction: false, // our own pause()/play() in flight (vs the user's)
      lastSeekAt: 0, // ts of last seek; play/pause within grace = not user intent
      onPlay: null, // user-intent detection listeners
      onPause: null,
      onSeekGate: null // seeking/seeked handler for the gate
    };
    session = s;

    ensureOverlay(player);
    syncFullscreenClass();

    // Bind timeupdate for caption lookup + playback gating during playback.
    s.onTimeUpdate = () => {
      if (s.generation !== sessionGeneration) return;
      updateGate(s);
      updateCaptionForCurrentTime(s);
    };
    video.addEventListener("timeupdate", s.onTimeUpdate);

    // Seeks get their own handler: stamp the seek time (so play/pause events it
    // triggers aren't mistaken for user intent), drop any stale override, and
    // re-evaluate the gate at the new position.
    s.onSeekGate = () => {
      if (s.generation !== sessionGeneration) return;
      s.lastSeekAt = Date.now();
      s.userOverride = false;
      updateGate(s);
      updateCaptionForCurrentTime(s);
    };
    video.addEventListener("seeking", s.onSeekGate);
    video.addEventListener("seeked", s.onSeekGate);

    // User-intent detection for the gate: distinguish our pause/play AND
    // seek-induced play/pause from a genuine manual action.
    s.onPause = () => {
      if (s.generation !== sessionGeneration) return;
      if (s.selfAction) { s.selfAction = false; return; } // our pause
      if (Date.now() - s.lastSeekAt < SEEK_GRACE_MS) return; // seek artifact
      s.pausedByUs = false; // genuine user pause -> we won't auto-resume it
    };
    s.onPlay = () => {
      if (s.generation !== sessionGeneration) return;
      if (s.selfAction) { s.selfAction = false; return; } // our play
      const seekInduced = Date.now() - s.lastSeekAt < SEEK_GRACE_MS;
      if (s.pausedByUs && !seekInduced) {
        // Genuine user play through an active hold: respect it and back off
        // until coverage catches up to the playhead again.
        s.pausedByUs = false;
        if (coveredEndAt(s, s.video.currentTime) === null) s.userOverride = true;
      }
      // Reconcile: a seek-induced play (or a normal start) is re-held here if
      // the current spot still isn't covered.
      updateGate(s);
    };
    video.addEventListener("pause", s.onPause);
    video.addEventListener("play", s.onPlay);

    connectSocket(s);
  }

  // Full re-initialization: tear down then start (used on nav/setting change).
  let reinitTimer = null;
  function scheduleReinit() {
    if (reinitTimer) clearTimeout(reinitTimer);
    // Small debounce so rapid events (navigate + url change) collapse into one.
    reinitTimer = setTimeout(() => {
      reinitTimer = null;
      teardownSession();
      if (settings.enabled && getVideoIdFromUrl()) {
        startSession();
      }
    }, 250);
  }

  // ---- Navigation + URL change detection ----------------------------------

  let lastVideoId = getVideoIdFromUrl();

  // YouTube fires this custom event when SPA navigation finishes.
  document.addEventListener("yt-navigate-finish", () => {
    const id = getVideoIdFromUrl();
    if (id !== lastVideoId) {
      lastVideoId = id;
      scheduleReinit();
    }
  });

  // Fallback: poll the URL in case the custom event is missed or changes
  // (e.g. clicking a related video that swaps ?v= without a full event). This
  // same tick is our orphan watchdog: if the extension was reloaded, we tear
  // our own (now-stale) overlay down so the user isn't stuck with frozen
  // captions that no longer respond to the popup toggle.
  urlPollTimer = setInterval(() => {
    if (selfDestructIfOrphaned()) return;
    const id = getVideoIdFromUrl();
    if (id !== lastVideoId) {
      lastVideoId = id;
      scheduleReinit();
    }
  }, 1000);

  // ---- Fullscreen / resize handling ---------------------------------------

  document.addEventListener("fullscreenchange", () => {
    syncFullscreenClass();
    // Re-parent the overlay in case YouTube swapped the player element.
    if (session && session.player) {
      const { player } = findPlayerElements();
      if (player) {
        session.player = player;
        ensureOverlay(player);
      }
    }
  });

  window.addEventListener("resize", () => {
    // Positioning is percentage-based so no manual recompute is needed, but
    // re-ensure the overlay is still attached to the live player element.
    if (session) {
      const { player } = findPlayerElements();
      if (player) {
        session.player = player;
        ensureOverlay(player);
      }
    }
  });

  // ---- Settings: load + react to changes ----------------------------------

  function loadSettingsThenInit() {
    chrome.storage.sync.get(
      ["enabled", "language", "fontSize", "engine", "model", "preBuffer", "autoPause"],
      (stored) => {
      settings.enabled = stored.enabled !== false; // default true
      settings.language =
        stored.language === undefined ? null : stored.language;
      settings.fontSize = stored.fontSize || "medium";
      settings.engine = stored.engine === "ollama" ? "ollama" : "whisper";
      settings.model = stored.model || "qwen2.5:7b";
      settings.preBuffer = stored.preBuffer !== false; // default true
      settings.autoPause = stored.autoPause !== false; // default true
      if (settings.enabled) {
        startSession();
      }
    });
  }

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    let needsReinit = false;

    if ("enabled" in changes) {
      settings.enabled = changes.enabled.newValue !== false;
      needsReinit = true;
    }
    if ("language" in changes) {
      const v = changes.language.newValue;
      settings.language = v === undefined ? null : v;
      needsReinit = true; // language affects the helper start message
    }
    if ("fontSize" in changes) {
      settings.fontSize = changes.fontSize.newValue || "medium";
      applyFontSizeClass(); // cheap, no reinit needed
    }
    if ("engine" in changes) {
      settings.engine =
        changes.engine.newValue === "whisper" ? "whisper" : "ollama";
      needsReinit = true; // engine is part of the start message → reconnect
    }
    if ("model" in changes) {
      settings.model = changes.model.newValue || "qwen2.5:7b";
      needsReinit = true; // model is part of the start message → reconnect
    }
    if ("preBuffer" in changes) {
      settings.preBuffer = changes.preBuffer.newValue !== false;
      needsReinit = true; // preBuffer is part of the start message → reconnect
    }
    if ("autoPause" in changes) {
      settings.autoPause = changes.autoPause.newValue !== false;
      // Client-only behavior toggle — no reconnect. If turning it OFF while
      // we're holding the video, release it immediately.
      if (!settings.autoPause && session && session.pausedByUs) {
        session.pausedByUs = false;
        session.userOverride = false;
        gatePlay(session);
        updateCaptionForCurrentTime(session);
      }
    }

    if (!settings.enabled) {
      // Disabled: tear down and remove overlay entirely.
      teardownSession();
      removeOverlay();
      return;
    }

    if (needsReinit) {
      scheduleReinit();
    }
  });

  // ---- Reset / re-translate (triggered by the popup button) ---------------
  // Clears the helper's cached transcript for the current video and restarts
  // the session so it re-transcribes from scratch. Useful when a cache is
  // wrong/partial (e.g. poisoned by a session that began mid-video).
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || msg.type !== "ytx-reset") return;
    const videoId = getVideoIdFromUrl();
    const restart = () => {
      teardownSession();
      if (settings.enabled && getVideoIdFromUrl()) startSession();
    };
    if (!videoId) {
      restart();
      if (sendResponse) sendResponse({ ok: false, reason: "no video" });
      return; // synchronous
    }
    renderStatus("♻ Re-translating…", false);
    fetch(HELPER_BASE + "/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoId })
    })
      .then((r) => r.json().catch(() => ({})))
      .then((data) => {
        restart();
        if (sendResponse) sendResponse({ ok: true, cleared: (data && data.cleared) || 0 });
      })
      .catch(() => {
        restart();
        if (sendResponse) sendResponse({ ok: false, reason: "helper" });
      });
    return true; // keep the channel open for the async sendResponse
  });

  // ---- Boot ---------------------------------------------------------------

  loadSettingsThenInit();
})();
