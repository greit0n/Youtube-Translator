"""Two-stage translation via a local Ollama chat model.

Whisper transcribes the SOURCE language faithfully; this module asks a local
Ollama model to translate each line to natural spoken English while preserving
profanity, slang, gaming terms, and proper nouns (Whisper's built-in translate
tends to censor/soften and drop names).

Talks to Ollama's HTTP API at http://127.0.0.1:11434 with `requests`. All calls
are blocking and meant to be run from a thread by the async server.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import requests

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b"

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
) -> str:
    """Build the translation prompt, optionally with a couple lines of context.

    `prev_context` is a list of (source_text, english_text) pairs from earlier in
    the stream, used to keep names/terms/tone consistent across windows.
    """
    lang = src_lang or "foreign"
    lines = [
        f"Translate the following {lang} speech transcript to natural, spoken "
        "English. Preserve profanity, slang, and gaming terms faithfully — "
        "DO NOT censor or soften. Keep names and proper nouns. Output ONLY the "
        "English translation, no notes.",
    ]

    if prev_context:
        lines.append("")
        lines.append("Recent context (for continuity, do not translate again):")
        for src, eng in prev_context[-2:]:
            lines.append(f"- {src} => {eng}")

    lines.append("")
    lines.append(f"Transcript: {text}")
    lines.append("English:")
    return "\n".join(lines)


def translate(
    text: str,
    src_lang: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prev_context: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Translate a single transcript line to English via Ollama.

    Includes up to ~2 previous source/English lines as context for continuity.
    Raises on connection error so the caller can fall back to Whisper translate.
    """
    text = (text or "").strip()
    if not text:
        return ""

    prompt = _build_prompt(text, src_lang, prev_context)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Low temperature: we want a faithful translation, not creativity.
            "temperature": 0.2,
        },
    }

    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=_GEN_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()
