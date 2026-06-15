// background.js — minimal MV3 service worker.
// Responsibilities:
//   1. Seed default settings into chrome.storage.sync on install.
//   2. Provide a tiny messaging glue endpoint (currently used so other
//      contexts can ask for defaults without duplicating them).

// Starter glossary (Czech → English). One entry per line: "term" or
// "term = preferred". Comma-separated options on the right are treated by the
// LLM as alternatives — it picks the one that best fits the sentence. Seeded
// into storage on install. Versioned migrations may replace it when the shipped
// default glossary is intentionally updated.
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
to nemyslíš vážně = you can't be serious
do píči = holy fuck, fuck
do pici = holy fuck, fuck
chápu = I understand, I get it
ch?pu = I understand, I get it
vyhul = suck it
vykuř mi ho = suck my dick
vykur mi ho = suck my dick
fakt mi ho vykuř = seriously suck my dick
fakt mi ho vykur = seriously suck my dick
fakt mi ho vykuš = seriously suck my dick
fakt mi ho vykus = seriously suck my dick
fuck mi ho = suck my dick
vykuš mi ho = suck my dick
vykus mi ho = suck my dick
vykuš ty piču = suck it, you bitch
vykuš ty píču = suck it, you bitch
vykus ty picu = suck it, you bitch
dej mi ho = give it to me
dej mi ho dej mi ho = give it to me, give it to me
pochcal = pissed myself
nepochcal = didn't piss myself
pochcat = piss myself
oklepávám = shake my dick off after peeing
Jsou prostě Bíčí varlata = They're just bull's balls
Bíčí varlata = bull's balls
býčí varlata = bull's balls
varlata = balls, testicles
koule = balls
Můžu ty dveře zavřít hubu = Can those doors shut up
zavřít hubu = shut up
zavři hubu = shut up
drž hubu = shut up
sas = suspicious
čum = look
sus = suspicious
pískání = whistling
kecáš = you are joking
časák = magazine`;

// Single source of truth for default settings. Kept in sync with popup.js.
const DEFAULT_SETTINGS = {
  enabled: false, // legacy only; live activation is per-tab at runtime
  language: "cs", // null = auto-detect; otherwise an ISO code like "cs"
  fontSize: "medium", // "small" | "medium" | "large"
  // Translation engine: "ollama" (accurate Czech source-first) or "whisper" (fast direct).
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

const ACCURATE_CZECH_DEFAULTS_VERSION = 6;

// On install / update, fill in any missing settings without clobbering
// values the user may have already set.
chrome.runtime.onInstalled.addListener(async () => {
  try {
    const current = await chrome.storage.sync.get([
      ...Object.keys(DEFAULT_SETTINGS),
      "glossarySeeded",
      "stabilizedDefaultsVersion"
    ]);
    const merged = {};
    for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
      merged[key] = key in current ? current[key] : value;
    }
    // Seed the starter glossary once for fresh installs.
    if (!current.glossarySeeded) {
      if (typeof merged.glossary !== "string" || merged.glossary.trim() === "") {
        merged.glossary = DEFAULT_SETTINGS.glossary;
      }
      merged.glossarySeeded = true;
    }
    // v6 makes the default quality-first for Agraelus-style Czech streams:
    // forced Czech source transcription, Gemma translation, and no audio/speaker
    // filters that could drop valid speech. It also intentionally replaces the
    // stored starter glossary with the current maintained default.
    if ((current.stabilizedDefaultsVersion || 0) < ACCURATE_CZECH_DEFAULTS_VERSION) {
      merged.engine = "ollama";
      merged.model = "gemma2:9b";
      merged.language = "cs";
      merged.cleanAudio = "off";
      merged.diarize = false;
      merged.enrolledOnly = false;
      merged.enabled = false;
      merged.glossary = DEFAULT_GLOSSARY;
      merged.stabilizedDefaultsVersion = ACCURATE_CZECH_DEFAULTS_VERSION;
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
