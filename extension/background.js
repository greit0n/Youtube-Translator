// background.js — minimal MV3 service worker.
// Responsibilities:
//   1. Seed default settings into chrome.storage.sync on install.
//   2. Provide a tiny messaging glue endpoint (currently used so other
//      contexts can ask for defaults without duplicating them).

// Single source of truth for default settings. Kept in sync with popup.js.
const DEFAULT_SETTINGS = {
  enabled: true, // master on/off switch
  language: null, // null = auto-detect; otherwise an ISO code like "cs"
  fontSize: "medium", // "small" | "medium" | "large"
  // Translation engine: "whisper" (fast built-in) or "ollama" (faithful LLM).
  engine: "whisper",
  // Ollama chat model used when engine === "ollama".
  model: "qwen2.5:7b",
  // Whether the helper should pre-buffer / look ahead of playback.
  preBuffer: true,
  // Pause playback until subtitles for the current moment are ready (client-side).
  autoPause: true,
  // Transcription quality tier (auto|max|balanced|lite). Drives model/beam selection on the helper.
  quality: "auto",
  // Audio pre-processing mode (off|light|music). Passed to the helper for denoise/separation.
  cleanAudio: "off",
  // Whether to run speaker diarization on the helper side.
  diarize: false,
  // Multiline glossary: one entry per line, "term" or "term = preferred". Sent to the helper.
  glossary: ""
};

// On install / update, fill in any missing settings without clobbering
// values the user may have already set.
chrome.runtime.onInstalled.addListener(async () => {
  try {
    const current = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
    const merged = {};
    for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
      merged[key] = key in current ? current[key] : value;
    }
    await chrome.storage.sync.set(merged);
  } catch (err) {
    // Storage may be unavailable in rare cases; nothing we can do but log.
    console.warn("[YT-Translator] Failed to initialize defaults:", err);
  }
});

// Simple message handler. Returns defaults on request. Extend as needed.
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message && message.type === "getDefaults") {
    sendResponse({ defaults: DEFAULT_SETTINGS });
    return true; // keep the channel open for the (synchronous) response
  }
  return false;
});
