"""Two-stage translation via a local Ollama chat model.

Whisper transcribes the SOURCE language faithfully; this module asks a local
Ollama model to translate each line to natural spoken English while preserving
profanity, slang, gaming terms, and proper nouns (Whisper's built-in translate
tends to censor/soften and drop names).

Talks to Ollama's HTTP API at http://127.0.0.1:11434 with `requests`. All calls
are blocking and meant to be run from a thread by the async server.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import requests

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma2:9b"

# Short timeout for liveness/listing; generation gets a longer one.
_PROBE_TIMEOUT = 1.5
_GEN_TIMEOUT = 120


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
        "word in the output. Render swearing as the natural, equivalent English "
        "swearing — never censor, euphemise, bracket, or omit it. Preserve slang "
        "and gaming terms, and keep names/proper nouns. The output MUST be English "
        "only: NEVER include any other language or non-Latin script. Output ONLY "
        "the translated line — no notes, no extra sentences, no continuation.",
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

    return out


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
    return _clean_response(data.get("response") or "", text)
