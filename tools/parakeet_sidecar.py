#!/usr/bin/env python3
"""
Reference sidecar for the Go transcription pipeline.

Reads one JSON request on stdin and writes one JSON response on stdout. The Go
side invokes this script via `--parakeet-cmd "python3 tools/parakeet_sidecar.py"`
when the user wants Parakeet (NeMo) candidates or post-judge alignment.

Request shape (transcribe):
    {
        "command": "transcribe",
        "audio_path": "/abs/path/to.wav",
        "speakers": ["Alice"],
        "model": "nvidia/parakeet-ctc-0.6b"
    }

Request shape (align):
    {
        "command": "align",
        "audio_path": "/abs/path/to.wav",
        "segments": [{"timestamp": "[00:00:00]", "speaker": "Alice", "text": "..."}],
        "model": "nvidia/parakeet-ctc-0.6b"
    }

Response shape (any command):
    {
        "segments": [TranscriptSegment, ...],   # optional
        "words":    [{"word": "...", "start": 0.0, "end": 0.0}, ...],  # optional
        "error":    "..."                          # populated on failure
    }

This file deliberately stays a thin reference wrapper so it can be replaced
without touching the Go code. It expects a Parakeet/NeMo implementation exposing
`agents.timestamp_tool` (plus `dependencies` and `models`) to be importable on
PYTHONPATH; supply your own, or point it at the original Python project. If the
import fails, the sidecar reports an error and the Go side annotates the
Parakeet candidate as unavailable.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


def _emit(payload: Dict[str, Any]) -> None:
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        _emit({"error": f"failed to parse request: {exc}"})
        return 1

    command = request.get("command")
    if command not in {"transcribe", "align"}:
        _emit({"error": f"unsupported command: {command!r}"})
        return 1

    try:
        # Lazy import so the sidecar can be tested without NeMo installed.
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
        from agents import timestamp_tool  # type: ignore
        from dependencies import TranscriptionDeps  # type: ignore
        from models import TranscriptSegment  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _emit({"error": f"sidecar deps unavailable: {exc}"})
        return 1

    deps = TranscriptionDeps(api_key="dummy", parakeet_model=request.get("model") or "nvidia/parakeet-ctc-0.6b")

    import asyncio  # local import keeps top-of-file fast

    async def run() -> Dict[str, Any]:
        if command == "transcribe":
            segs = await timestamp_tool.transcribe_with_parakeet(
                deps,
                request["audio_path"],
                request.get("speakers") or None,
            )
            return {"segments": [s.model_dump() for s in segs]}
        # command == "align"
        segs_in = [TranscriptSegment(**seg) for seg in request.get("segments") or []]
        segs_out = await timestamp_tool.fix_timestamps_with_parakeet(
            deps,
            request["audio_path"],
            segs_in,
        )
        return {"segments": [s.model_dump() for s in segs_out]}

    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        _emit({"error": str(exc)})
        return 1
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
