// background.js — minimal MV3 service worker.
// Responsibilities:
//   1. Seed default settings into chrome.storage.sync on install.
//   2. Provide a tiny messaging glue endpoint (currently used so other
//      contexts can ask for defaults without duplicating them).

// Starter glossary (Czech → English). One entry per line: "term" or
// "term = preferred". Comma-separated options on the right are treated by the
// LLM as alternatives — it picks the one that best fits the sentence. Seeded
// into storage on install AND whenever the user's glossary is empty (see below),
// so it reaches existing installs on the next extension reload. Edit/clear it
// freely in the popup; a non-empty value is never overwritten.
const DEFAULT_GLOSSARY = `blbec = idiot, moron
idiot = idiot
debil = moron, dumbass
magor = psycho, nutcase
kretén = cretin, idiot
vůl = idiot, dumb ox
pitomec = fool, idiot
trouba = dummy, fool
dement = moron, idiot
retard = retard

kokot = asshole, dickhead
píča = cunt, dumb bitch
čurák = dickhead, prick
zmrd = scumbag, motherfucker
mrdka = piece of shit
sráč = asshole, coward
hajzl = asshole, bastard
kunda = cunt
buzna = faggot
šulina = dickhead, little dick

do prdele = for fuck's sake, damn it
kurva = fuck
kurva fix = fucking hell
ty vole = dude, bro, holy shit
doprdele práce = for fuck's sake
ježiši kriste = Jesus Christ
sakra = damn
do hajzlu = go to hell, fuck this
no ty píčo = holy fuck
kurva drát = fucking hell

co to je za kokotinu = what is this bullshit
to si děláš prdel = are you kidding me
běž do prdele = fuck off
ses posral ne = are you out of your mind
to je úplně v píči = this is completely fucked
tak tohle je mrdka = this is garbage
co je to za bullshit = what is this bullshit
ty vole neee = dude noooo
kurvaaa = fuuuck

cringe = cringe
npc = npc
brainrot = brainrot
autista = autist (insult)
lobotom = lobotomite, brain-dead person
schizo = schizo
opice = monkey, ape
klaun = clown
klauníček = little clown
copium = copium
cope = cope

kkt = asshole, dickhead
p*ča = cunt
pica = cunt
pyčo = fuck, holy shit
pyčo vole = holy fuck dude
čůrák = dickhead
curak = dickhead
kokutek = little dickhead
kokůtek = little dickhead
krva = fuck
kruci = darn, dang
doprkna = damn it

kámo = bro, mate
brácho = brother, bro
no tak = come on
počkej = wait
ježišmarja = oh my god
no do píči = holy fuck
cože = what
jak jako = what do you mean
to nemyslíš vážně = you can't be serious`;

// Single source of truth for default settings. Kept in sync with popup.js.
const DEFAULT_SETTINGS = {
  enabled: true, // master on/off switch
  language: null, // null = auto-detect; otherwise an ISO code like "cs"
  fontSize: "medium", // "small" | "medium" | "large"
  // Translation engine: "whisper" (fast built-in) or "ollama" (faithful LLM).
  engine: "ollama",
  // Ollama chat model used when engine === "ollama".
  model: "gemma2:9b",
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
  enrolledOnly: false, // when on, show ONLY the enrolled "your voice" speaker (drops game audio + other speakers)
  // Multiline glossary: one entry per line, "term" or "term = preferred". Sent to the helper.
  glossary: DEFAULT_GLOSSARY,
  // Display name for the enrolled "your voice" speaker (overrides the enroll file name).
  highlightName: "",
  // Colour painted on the enrolled speaker's captions (default red).
  highlightColor: "#ff3b30"
};

// On install / update, fill in any missing settings without clobbering
// values the user may have already set.
chrome.runtime.onInstalled.addListener(async () => {
  try {
    const current = await chrome.storage.sync.get([
      ...Object.keys(DEFAULT_SETTINGS),
      "glossarySeeded"
    ]);
    const merged = {};
    for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
      merged[key] = key in current ? current[key] : value;
    }
    // Seed the starter glossary EXACTLY ONCE (first run of a build that knows
    // the sentinel). This delivers the shipped default to EXISTING installs whose
    // stored glossary is still empty, while RESPECTING a later deliberate clear:
    // onInstalled also fires on every reload/update, so a plain empty-check would
    // keep re-adding the glossary the user just cleared.
    if (!current.glossarySeeded) {
      if (typeof merged.glossary !== "string" || merged.glossary.trim() === "") {
        merged.glossary = DEFAULT_SETTINGS.glossary;
      }
      merged.glossarySeeded = true;
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
