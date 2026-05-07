"""Launch the live WikiRisk ingestion stack locally.

Starts the SSE collector and Spark streaming processor as sibling
processes so the prediction store keeps receiving fresh records.

Usage:
    python scripts/run_live_stack.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _start_process(module: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def main() -> int:
    collector = _start_process("src.streaming.collector")
    processor = _start_process("src.streaming.processor")
    children = [collector, processor]

    def shutdown(signum, frame):  # noqa: ARG001
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    child.kill()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        exit_codes = [child.wait() for child in children]
        return max(exit_codes)
    finally:
        shutdown(signal.SIGTERM, None)


if __name__ == "__main__":
    raise SystemExit(main())