"""Two-stage translation via a local Ollama chat model.

Whisper transcribes the SOURCE language faithfully; this module asks a local
Ollama model to translate each line to natural spoken English while preserving
profanity, slang, gaming terms, and proper nouns (Whisper's built-in translate
tends to censor/soften and drop names).

Talks to Ollama's HTTP API at http://127.0.0.1:11434 with `requests`. All calls
are blocking and meant to be run from a thread by the async server.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

import requests

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma2:9b"

# Short timeout for liveness/listing; generation gets a longer one.
_PROBE_TIMEOUT = 1.5
_GEN_TIMEOUT = 120

_BAD_OUTPUT_PATTERNS = [
    r"(?i)\btranscript\s*:",
    r"(?i)\btranslation\s*:",
    r"(?i)\bimportant note\b",
    r"(?i)\bnote\s*:",
    r"(?i)\bas requested\b",
    r"(?i)\bprovided terms\b",
    r"(?i)\bthe rest of\b",
    r"(?i)\bcontinues as requested\b",
    r"(?i)\blet'?s (?:get this )?translat",
    r"(?i)\bplease provide\b",
    r"(?i)\bprovide (?:the|a) .*subtitle line\b",
    r"(?i)\bi need (?:the|a) .*text\b",
    r"(?i)\bi'?m ready to translate\b",
    r"(?i)^here we go\s*:",
    r"(?i)^alright,\s*let",
]

_NON_LATIN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    r"\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]"
)


def is_up() -> bool:
    """Quick check that Ollama is reachable (short timeout, never raises)."""
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=_PROBE_TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_chat_models() -> List[str]:
    """Return Ollama model names, excluding embedding models.

    Any model whose name contains "embed" is filtered out (embedding models
    can't do chat/generate). Returns [] if Ollama is unreachable.
    """
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=_PROBE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return []

    names = []
    for m in data.get("models") or []:
        name = m.get("name")
        if name and "embed" not in name.lower():
            names.append(name)
    return names


def _build_prompt(
    text: str,
    src_lang: Optional[str],
    prev_context: Optional[List[Tuple[str, str]]],
    glossary: Optional[List[dict]] = None,
) -> str:
    """Build the translation prompt, optionally with a couple lines of context.

    `prev_context` is a list of (source_text, english_text) pairs from earlier in
    the stream, used to keep names/terms/tone consistent across windows.

    `glossary` is a list of {term, preferred} dicts. Terms with a preferred
    translation are pinned to that exact English; bare terms (no preferred) are
    treated as proper nouns to keep as-is. This keeps names/jargon consistent.
    """
    lang = src_lang or "foreign"
    lines = [
        f"Translate this single {lang} subtitle line into ONE short, natural line "
        "of spoken English. Translate EVERY word — never leave a source-language "
        "word in the output. Prefer literal meaning over smoothing. Render "
        "swearing and sexual insults as the natural, equivalent English swearing "
        "— never censor, euphemise, bracket, or omit it. Do not turn sexual Czech "
        "insults into generic fighting threats. Preserve slang and gaming terms, "
        "and keep names/proper nouns. If the source contains a YouTube censor "
        "marker like [ __ ], translate it as likely profanity (usually fuck, "
        "fucking, shit, or damn) from context; do not output the brackets. The "
        "output MUST be English only: NEVER include any other language or "
        "non-Latin script. Output ONLY the "
        "translated line — no notes, no extra sentences, no continuation.",
    ]

    if glossary:
        # Inject ONLY the glossary entries whose term actually appears in THIS
        # line. Dumping the whole glossary (can be dozens of entries) bloats the
        # prompt and makes the model ramble and leak untranslated source terms.
        low = text.lower()
        pinned = []
        keep = []
        for entry in glossary:
            term = (entry.get("term") or "").strip()
            if not term:
                continue
            if term.lower() not in low:
                continue
            preferred = (entry.get("preferred") or "").strip()
            if preferred:
                pinned.append(f"- {term} => {preferred}")
            else:
                keep.append(term)
        if pinned:
            lines.append("")
            lines.append(
                "Glossary — translate each source term using the English given. "
                "When several comma-separated options are listed, choose the ONE "
                "that best fits the sentence (do not output the whole list):"
            )
            lines.extend(pinned)
        if keep:
            lines.append("")
            lines.append(
                "Keep these terms as-is (proper nouns, do not translate): "
                + ", ".join(keep)
            )

    if prev_context:
        lines.append("")
        lines.append("Recent context (for continuity, do not translate again):")
        for src, eng in prev_context[-2:]:
            lines.append(f"- {src} => {eng}")

    lines.append("")
    lines.append(f"Transcript: {text}")
    lines.append("English:")
    return "\n".join(lines)


def _relevant_glossary_block(text: str, glossary: Optional[List[dict]]) -> list[str]:
    if not glossary:
        return []
    low = text.lower()
    pinned = []
    keep = []
    for entry in glossary:
        term = (entry.get("term") or "").strip()
        if not term or term.lower() not in low:
            continue
        preferred = (entry.get("preferred") or "").strip()
        if preferred:
            pinned.append(f"- {term} => {preferred}")
        else:
            keep.append(term)

    lines: list[str] = []
    if pinned:
        lines.append("Glossary mappings to honor:")
        lines.extend(pinned)
    if keep:
        lines.append("Keep these proper nouns as-is: " + ", ".join(keep))
    return lines


def _build_batch_prompt(
    items: List[Tuple[str, Optional[str]]],
    prev_context: Optional[List[Tuple[str, str]]],
    glossary: Optional[List[dict]] = None,
) -> str:
    lines = [
        "Translate each subtitle item into short, natural spoken English.",
        "Return ONLY valid JSON with this exact shape: "
        '{"translations":["line 1","line 2"]}.',
        "The array length MUST match the number of input items.",
        "Do not include notes, explanations, labels, markdown, or prompt text.",
        "Translate every source-language word; preserve profanity, slang, gaming "
        "terms, and proper nouns. Output English only.",
    ]

    if prev_context:
        lines.append("")
        lines.append("Recent context for continuity only:")
        for src, eng in prev_context[-2:]:
            lines.append(f"- {src} => {eng}")

    glossary_lines = []
    for text, _lang in items:
        glossary_lines.extend(_relevant_glossary_block(text, glossary))
    if glossary_lines:
        lines.append("")
        lines.extend(list(dict.fromkeys(glossary_lines)))

    lines.append("")
    lines.append("Input items:")
    for idx, (text, lang) in enumerate(items, start=1):
        lines.append(f"{idx}. ({lang or 'foreign'}) {text}")
    return "\n".join(lines)


def _invalid_output(out: str) -> bool:
    if not out:
        return True
    if _NON_LATIN_RE.search(out):
        return True
    return any(re.search(pattern, out) for pattern in _BAD_OUTPUT_PATTERNS)


def _clean_response(response: str, source: str) -> str:
    """Normalise a raw Ollama reply into ONE short subtitle line.

    A subtitle line translates to roughly one line. The classic failure mode —
    a confused/weak model fed too much glossary — is to ramble into a whole
    paragraph (or even invent dialogue from the glossary words). num_predict +
    per-line glossary filtering make that rare, but this is the last-line backstop
    so a few seconds of speech can NEVER fill the screen with a paragraph:

      1. Drop an echoed "English:" prefix.
      2. Keep only the first non-empty line (paragraph runaway emits blank-line
         separated blocks).
      3. If still wildly longer than the source, cut to the first sentence.
    """
    out = (response or "").strip()
    if not out:
        return ""

    # 1. The model occasionally echoes the prompt's "English:" label.
    if out[:8].lower() == "english:":
        out = out[8:].strip()

    # 2. One subtitle line. Multiple lines => runaway; take the first real one.
    if "\n" in out:
        for ln in out.splitlines():
            ln = ln.strip()
            if ln:
                out = ln
                break

    # 3. A faithful translation of a short line stays short. If it's still
    #    dramatically longer than the source, the model is rambling — keep the
    #    first sentence (or a hard char cap if there's no sentence break).
    limit = max(140, len(source) * 4)
    if len(out) > limit:
        m = re.search(r"[.!?](?:\s|$)", out)
        if m and m.end() >= 8:
            out = out[: m.end()].strip()
        else:
            out = out[:limit].rstrip() + "…"

    if _invalid_output(out):
        return ""

    return out


def _extract_batch_response(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty Ollama response")
    try:
        data = json.loads(raw)
    except ValueError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Ollama did not return JSON")
        data = json.loads(raw[start : end + 1])
    translations = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(translations, list):
        raise ValueError("Ollama JSON missing translations array")
    return [str(x).strip() for x in translations]


def translate(
    text: str,
    src_lang: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prev_context: Optional[List[Tuple[str, str]]] = None,
    glossary: Optional[List[dict]] = None,
) -> str:
    """Translate a single transcript line to English via Ollama.

    Includes up to ~2 previous source/English lines as context for continuity.
    `glossary` pins names/jargon to consistent translations (see _build_prompt).
    Raises on connection error so the caller can fall back to Whisper translate.
    """
    text = (text or "").strip()
    if not text:
        return ""

    prompt = _build_prompt(text, src_lang, prev_context, glossary)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Low temperature: we want a faithful translation, not creativity.
            "temperature": 0.2,
            # Hard cap on output length: one subtitle line is short, so this stops
            # the model from running away into a whole paragraph for a few seconds
            # of speech (a failure mode seen with a large glossary in-prompt).
            "num_predict": 160,
        },
    }

    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=_GEN_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = _clean_response(data.get("response") or "", text)
    if not out:
        raise ValueError("invalid Ollama translation")
    return out


def translate_many(
    items: List[Tuple[str, Optional[str]]],
    model: str = DEFAULT_MODEL,
    prev_context: Optional[List[Tuple[str, str]]] = None,
    glossary: Optional[List[dict]] = None,
) -> List[str]:
    """Translate a window of subtitle lines in one Ollama call.

    Raises if the model returns prompt/meta text, wrong-length output, non-Latin
    script leakage, or anything other than clean subtitle lines.
    """
    clean_items = [(text.strip(), lang) for text, lang in items if text and text.strip()]
    if not clean_items:
        return []
    if len(clean_items) == 1:
        text, lang = clean_items[0]
        return [translate(text, lang, model, prev_context, glossary)]

    prompt = _build_batch_prompt(clean_items, prev_context, glossary)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": min(1024, 120 + 120 * len(clean_items)),
        },
    }

    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=_GEN_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    raw_items = _extract_batch_response(data.get("response") or "")
    if len(raw_items) != len(clean_items):
        raise ValueError(
            f"Ollama returned {len(raw_items)} translations for {len(clean_items)} lines"
        )

    out = []
    for raw, (source, _lang) in zip(raw_items, clean_items):
        cleaned = _clean_response(raw, source)
        if not cleaned:
            raise ValueError("invalid Ollama translation")
        out.append(cleaned)
    return out
