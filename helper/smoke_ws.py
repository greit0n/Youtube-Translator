"""Small WebSocket smoke test for the local subtitle helper.

Run the helper first:

    python server.py

Then run, for example:

    python smoke_ws.py VIDEO_ID --start 0 --language cs --engine whisper

This intentionally avoids browser automation. It exercises the same WebSocket
path the extension uses and prints status/progress/segment messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import websockets


DEFAULT_URL = "ws://127.0.0.1:8765/transcribe"


def _build_start(args: argparse.Namespace) -> dict[str, Any]:
    language = None if args.language.lower() == "auto" else args.language
    return {
        "videoId": args.video_id,
        "startTime": args.start,
        "language": language,
        "engine": args.engine,
        "model": args.model,
        "preBuffer": True,
        "quality": args.quality,
        "cleanAudio": "off",
        "diarize": False,
        "enrolledOnly": False,
        "hotwords": None,
        "glossary": [],
    }


async def _run(args: argparse.Namespace) -> None:
    async with websockets.connect(args.url, ping_interval=None) as ws:
        await ws.send(json.dumps(_build_start(args)))
        await ws.send(json.dumps({"type": "position", "currentTime": args.start}))

        segments = 0
        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")
            print(json.dumps(msg, ensure_ascii=False))

            if msg_type == "segment":
                segments += 1
                if segments >= args.segments:
                    return
            if msg_type in {"done", "error"}:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="YouTube video ID, not the full URL")
    parser.add_argument("--url", default=DEFAULT_URL, help="helper WebSocket URL")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--language", default="auto", help="source language code or auto")
    parser.add_argument("--engine", choices=("ollama", "whisper"), default="whisper")
    parser.add_argument("--model", default="gemma2:9b", help="Ollama model name")
    parser.add_argument(
        "--quality",
        choices=("auto", "max", "balanced", "lite"),
        default="auto",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=3,
        help="stop after this many segment messages",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
