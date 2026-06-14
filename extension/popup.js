// popup.js — settings UI logic + live helper status indicator.
//
// Reads/writes settings to chrome.storage.sync (shared with content.js) and
// pings the helper's GET /health endpoint to display connection status.

"use strict";

const HELPER_HEALTH = "http://127.0.0.1:8765/health";
const HELPER_MODELS = "http://127.0.0.1:8765/models";

// Matches background.js DEFAULT_SETTINGS.
const DEFAULTS = {
  enabled: true,
  language: null, // null = auto-detect
  fontSize: "medium",
  engine: "ollama", // "whisper" | "ollama"
  model: "gemma2:9b", // Ollama chat model
  preBuffer: true,
  autoPause: true, // pause until subtitles for "now" are ready
  quality: "auto", // "auto" | "max" | "balanced" | "lite"
  cleanAudio: "off", // "off" | "light" | "music"
  diarize: false, // speaker diarization
  enrolledOnly: false, // show only the enrolled "your voice" speaker
  glossary: "", // multiline: "term" or "term = preferred" per line
  highlightName: "", // display name for the enrolled "your voice" speaker
  highlightColor: "#ff3b30" // colour for the enrolled speaker (default red)
};

// DOM references.
const enabledEl = document.getElementById("enabled");
const languageEl = document.getElementById("language");
const fontSizeEl = document.getElementById("fontSize");
const engineEl = document.getElementById("engine");
const modelEl = document.getElementById("model");
const modelHintEl = document.getElementById("modelHint");
const preBufferEl = document.getElementById("preBuffer");
const autoPauseEl = document.getElementById("autoPause");
const qualityEl = document.getElementById("quality");
const cleanAudioEl = document.getElementById("cleanAudio");
const diarizeEl = document.getElementById("diarize");
const enrolledOnlyEl = document.getElementById("enrolledOnly");
const glossaryEl = document.getElementById("glossary");
const highlightNameEl = document.getElementById("highlightName");
const highlightColorEl = document.getElementById("highlightColor");
const statusDot = document.getElementById("statusDot");
const statusTitle = document.getElementById("statusTitle");
const statusDetail = document.getElementById("statusDetail");
const livePillText = document.getElementById("livePillText");

// The model we want selected (stored value or default), applied once the
// dropdown is populated from GET /models.
let desiredModel = DEFAULTS.model;

// ---- Settings load / save ------------------------------------------------

function loadSettings() {
  chrome.storage.sync.get(Object.keys(DEFAULTS), (stored) => {
    const enabled = stored.enabled !== false; // default true
    const language =
      stored.language === undefined ? DEFAULTS.language : stored.language;
    const fontSize = stored.fontSize || DEFAULTS.fontSize;
    const engine = stored.engine === "whisper" ? "whisper" : "ollama";
    const model = stored.model || DEFAULTS.model;
    const preBuffer = stored.preBuffer !== false; // default true
    const autoPause = stored.autoPause !== false; // default true
    const quality = stored.quality || "auto";
    const cleanAudio = stored.cleanAudio || "off";
    const diarize = stored.diarize === true;
    const enrolledOnly = stored.enrolledOnly === true;
    const glossary = stored.glossary || "";
    const highlightName = stored.highlightName || "";
    const highlightColor = stored.highlightColor || "#ff3b30";

    enabledEl.checked = enabled;
    // language null -> "auto" option value
    languageEl.value = language === null ? "auto" : language;
    fontSizeEl.value = fontSize;
    engineEl.value = engine;
    preBufferEl.checked = preBuffer;
    autoPauseEl.checked = autoPause;
    qualityEl.value = quality;
    cleanAudioEl.value = cleanAudio;
    diarizeEl.checked = diarize;
    enrolledOnlyEl.checked = enrolledOnly;
    glossaryEl.value = glossary;
    highlightNameEl.value = highlightName;
    highlightColorEl.value = highlightColor;

    // Remember the desired model so it can be selected after /models loads.
    desiredModel = model;
    syncModelEnabled(engine);
  });
}

// Enable/disable the model dropdown based on the chosen engine. The model
// only matters when the Ollama engine is active.
function syncModelEnabled(engine) {
  modelEl.disabled = engine !== "ollama";
}

// Persist a single setting.
function save(key, value) {
  chrome.storage.sync.set({ [key]: value });
}

enabledEl.addEventListener("change", () => {
  save("enabled", enabledEl.checked);
});

languageEl.addEventListener("change", () => {
  // Map the "auto" UI value back to null for the contract.
  const value = languageEl.value === "auto" ? null : languageEl.value;
  save("language", value);
});

fontSizeEl.addEventListener("change", () => {
  save("fontSize", fontSizeEl.value);
});

engineEl.addEventListener("change", () => {
  const engine = engineEl.value === "whisper" ? "whisper" : "ollama";
  save("engine", engine);
  syncModelEnabled(engine);
});

modelEl.addEventListener("change", () => {
  if (modelEl.value) {
    desiredModel = modelEl.value;
    save("model", modelEl.value);
  }
});

preBufferEl.addEventListener("change", () => {
  save("preBuffer", preBufferEl.checked);
});

autoPauseEl.addEventListener("change", () => {
  save("autoPause", autoPauseEl.checked);
});

qualityEl.addEventListener("change", () => {
  save("quality", qualityEl.value);
});

cleanAudioEl.addEventListener("change", () => {
  save("cleanAudio", cleanAudioEl.value);
});

diarizeEl.addEventListener("change", () => {
  save("diarize", diarizeEl.checked);
});

enrolledOnlyEl.addEventListener("change", () => {
  save("enrolledOnly", enrolledOnlyEl.checked);
});

// Glossary textarea fires many input events while typing — debounce the save
// so we don't write to chrome.storage.sync on every keystroke.
let glossarySaveTimer = null;
glossaryEl.addEventListener("input", () => {
  if (glossarySaveTimer) clearTimeout(glossarySaveTimer);
  glossarySaveTimer = setTimeout(() => {
    save("glossary", glossaryEl.value);
  }, 400);
});

// "Your voice" name — debounce like the glossary (fires per keystroke).
let highlightNameSaveTimer = null;
highlightNameEl.addEventListener("input", () => {
  if (highlightNameSaveTimer) clearTimeout(highlightNameSaveTimer);
  highlightNameSaveTimer = setTimeout(() => {
    save("highlightName", highlightNameEl.value.trim());
  }, 400);
});

highlightColorEl.addEventListener("input", () => {
  save("highlightColor", highlightColorEl.value);
});

// ---- Ollama model list ---------------------------------------------------

// Fetch GET /models and populate the dropdown. Selects the stored/default
// model if present; otherwise selects the first model. Shows a hint when no
// usable chat model is available.
async function loadModels() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    const resp = await fetch(HELPER_MODELS, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store"
    });
    clearTimeout(timer);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    const models = Array.isArray(data && data.models) ? data.models : [];
    populateModels(models);
  } catch (_err) {
    // Helper down or no /models — leave the dropdown empty and hint the user.
    populateModels([]);
  }
}

function populateModels(models) {
  // Reset the dropdown (clear existing options safely, no innerHTML).
  while (modelEl.firstChild) {
    modelEl.removeChild(modelEl.firstChild);
  }

  if (!models || models.length === 0) {
    // No usable chat model — show the hint and disable selection.
    modelHintEl.hidden = false;
    return;
  }
  modelHintEl.hidden = true;

  for (const name of models) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    modelEl.appendChild(opt);
  }

  // Prefer the stored/default model if the helper offers it; else fall back
  // to the first model so the dropdown always reflects a real selection.
  if (models.includes(desiredModel)) {
    modelEl.value = desiredModel;
  } else {
    modelEl.value = models[0];
    desiredModel = models[0];
    // Persist the fallback so content.js sends a model the helper actually has.
    save("model", models[0]);
  }
}

// ---- Helper health check -------------------------------------------------

function setStatus(state, title, detail) {
  statusDot.classList.remove("connected", "disconnected");
  if (state) statusDot.classList.add(state);
  statusTitle.textContent = title;
  statusDetail.textContent = detail;
  // Mirror the connection state into the compact header pill.
  if (livePillText) {
    livePillText.textContent =
      state === "connected" ? "live" : state === "disconnected" ? "off" : "…";
  }
}

async function checkHealth() {
  setStatus(null, "Checking helper…", "127.0.0.1:8765");
  try {
    // Short timeout so a dead port doesn't hang the indicator.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    const resp = await fetch(HELPER_HEALTH, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store"
    });
    clearTimeout(timer);

    if (!resp.ok) {
      throw new Error("HTTP " + resp.status);
    }
    const data = await resp.json();
    if (data && data.status === "ok") {
      // Helper build id — lets the user confirm the RUNNING helper picked up the
      // latest code (a stale, un-restarted process is the #1 "nothing changed" cause).
      const ver = data.version ? `v${data.version} · ` : "";
      const device = (data.device || (data.cuda ? "cuda" : "cpu")).toUpperCase();
      const model = data.model_loaded ? "model loaded" : "model loading…";
      // Surface the active Whisper model tier (hardware-adaptive selection).
      const whisper = data.whisper_model ? `${data.whisper_model} · ` : "";
      // Show Ollama availability alongside the existing device/model info.
      const ollama = data.ollama ? "Ollama: ready" : "Ollama: ▢";
      const cookies = data.cookies ? "Cookies: ✓" : "Cookies: ▢";
      // Confirm any enrolled "your voice" clips the helper found.
      const enrolled = Array.isArray(data.enrolled) && data.enrolled.length
        ? ` · Voice: ${data.enrolled.join(", ")}`
        : "";
      setStatus(
        "connected",
        "Connected",
        `${ver}Device: ${device} · ${whisper}${model} · ${ollama} · ${cookies}${enrolled}`
      );
    } else {
      setStatus("disconnected", "Helper responded oddly", "Unexpected /health payload");
    }
  } catch (_err) {
    setStatus(
      "disconnected",
      "Disconnected",
      "Helper not running on :8765"
    );
  }
}

// ---- Re-translate button -------------------------------------------------
// Asks the content script in the active tab to clear the helper cache for the
// current video and restart transcription from scratch.
const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    const original = resetBtn.textContent;
    resetBtn.disabled = true;
    resetBtn.textContent = "♻ Re-translating…";
    const restore = (label) => {
      resetBtn.textContent = label || original;
      setTimeout(() => {
        resetBtn.textContent = original;
        resetBtn.disabled = false;
      }, 1200);
    };
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs && tabs[0];
      if (!tab || !tab.id) {
        restore("Open a YouTube video first");
        return;
      }
      chrome.tabs.sendMessage(tab.id, { type: "ytx-reset" }, (resp) => {
        if (chrome.runtime.lastError) {
          restore("Open a YouTube video first");
          return;
        }
        if (resp && resp.ok) {
          restore("✓ Re-translating…");
        } else {
          restore("Open a YouTube video first");
        }
      });
    });
  });
}

// ---- Boot ----------------------------------------------------------------

loadSettings();
loadModels();
checkHealth();
// Refresh status periodically while the popup is open.
const healthTimer = setInterval(checkHealth, 4000);
window.addEventListener("unload", () => clearInterval(healthTimer));
