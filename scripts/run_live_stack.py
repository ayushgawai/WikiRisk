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
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _start_process(module: str, args: list[str] | None = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Spark on macOS can fall back to loopback/heartbeat issues unless the
        # local bind address is pinned explicitly.
        "SPARK_LOCAL_IP": os.environ.get("SPARK_LOCAL_IP", "127.0.0.1"),
        "SPARK_LOCAL_HOSTNAME": os.environ.get("SPARK_LOCAL_HOSTNAME", "localhost"),
    }
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
    )


def main() -> int:
    # Keep the demo feed visibly fresh: flush collector batches and process
    # micro-batches on a 5-second cadence.
    collector = _start_process(
        "src.streaming.collector",
        ["--flush-interval", "5"],
    )
    processor = _start_process(
        "scripts.run_demo_scoring",
    )
    children = [collector, processor]

    def _restart(child: subprocess.Popen, module: str, args: list[str] | None = None) -> subprocess.Popen:
        log_path = "/tmp/wikirisk_live_stack.log"
        try:
            code = child.poll()
        except Exception:
            code = None
        if code is not None:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"Restarting {module} after exit code {code}\n")
            return _start_process(module, args)
        return child

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
        while True:
            time.sleep(2)
            collector = _restart(collector, "src.streaming.collector", ["--flush-interval", "5"])
            processor = _restart(processor, "scripts.run_demo_scoring")
            children[:] = [collector, processor]
    finally:
        shutdown(signal.SIGTERM, None)


if __name__ == "__main__":
    raise SystemExit(main())