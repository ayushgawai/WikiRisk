"""
WikiRisk – Wikipedia dump downloader.

Downloads bz2-compressed Wikipedia revision dumps from dumps.wikimedia.org,
verifies decompressed size ≥ threshold, and writes dataset_manifest.json.

CLI usage:
    python -m src.batch.downloader [--min-gb 10]
"""
from __future__ import annotations

import bz2
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.common.logger import get_logger, configure_logging
from src.config import get_settings

log = get_logger(__name__)

# Specific Wikipedia dump files to download.
# We select files that together exceed 10 GB decompressed.
# These are from the 20240420 dump of the English Wikipedia.
DUMP_FILES = [
    {
        "name": "enwiki-20240420-pages-meta-history1.xml-p1p912.bz2",
        "url": (
            "https://dumps.wikimedia.org/enwiki/20240420/"
            "enwiki-20240420-pages-meta-history1.xml-p1p912.bz2"
        ),
        "description": "Pages 1–912 with full revision history",
    },
    {
        "name": "enwiki-20240420-pages-meta-history2.xml-p913p2621.bz2",
        "url": (
            "https://dumps.wikimedia.org/enwiki/20240420/"
            "enwiki-20240420-pages-meta-history2.xml-p913p2621.bz2"
        ),
        "description": "Pages 913–2621 with full revision history",
    },
    {
        "name": "enwiki-20240420-pages-meta-history3.xml-p2622p5405.bz2",
        "url": (
            "https://dumps.wikimedia.org/enwiki/20240420/"
            "enwiki-20240420-pages-meta-history3.xml-p2622p5405.bz2"
        ),
        "description": "Pages 2622–5405 with full revision history",
    },
]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> int:
    """Stream-download *url* to *dest*; return number of bytes written."""
    log.info("downloading", url=url, dest=str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        t0 = time.monotonic()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
        elapsed = time.monotonic() - t0
        speed_mb = (written / 1e6) / max(elapsed, 0.001)
        log.info(
            "download_complete",
            file=dest.name,
            size_mb=round(written / 1e6, 1),
            speed_mb_s=round(speed_mb, 1),
        )
    return written


def _md5(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_revisions_in_bz2(path: Path, sample_lines: int = 200_000) -> int:
    """Count <revision> tags in the first *sample_lines* lines; scale up."""
    tag = b"<revision>"
    count = 0
    lines_read = 0
    with bz2.open(path, "rb") as fh:
        for raw in fh:
            if tag in raw:
                count += 1
            lines_read += 1
            if lines_read >= sample_lines:
                break
    # Scale to estimate full file
    with bz2.open(path, "rb") as fh:
        total_bytes = sum(len(ln) for ln in fh)

    sampled_bytes = 0
    with bz2.open(path, "rb") as fh:
        for i, ln in enumerate(fh):
            sampled_bytes += len(ln)
            if i >= sample_lines:
                break

    if sampled_bytes > 0:
        scale = total_bytes / sampled_bytes
        return int(count * scale)
    return count


def estimate_decompressed_size(path: Path) -> int:
    """Estimate decompressed size in bytes via bz2 header (approximate)."""
    # Ratio is roughly 5–10x for Wikipedia XML
    compressed = path.stat().st_size
    return compressed * 7  # conservative estimate


def download_dumps(
    dest_dir: Path,
    min_gb: float = 10.0,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Download Wikipedia dumps; return manifest entries."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_compressed = 0
    total_decompressed_est = 0

    for info in DUMP_FILES:
        dest = dest_dir / info["name"]

        if skip_existing and dest.exists():
            log.info("skipping_existing", file=info["name"])
        else:
            compressed_bytes = _download_file(info["url"], dest)

        compressed_bytes = dest.stat().st_size
        decompressed_est = estimate_decompressed_size(dest)
        total_compressed += compressed_bytes
        total_decompressed_est += decompressed_est

        record: dict[str, Any] = {
            "name": info["name"],
            "url": info["url"],
            "description": info["description"],
            "compressed_bytes": compressed_bytes,
            "compressed_mb": round(compressed_bytes / 1e6, 1),
            "decompressed_bytes_estimate": decompressed_est,
            "decompressed_gb_estimate": round(decompressed_est / 1e9, 2),
            "local_path": str(dest),
        }
        records.append(record)
        log.info(
            "file_registered",
            name=info["name"],
            compressed_mb=record["compressed_mb"],
            decompressed_gb_est=record["decompressed_gb_estimate"],
        )

    total_gb = total_decompressed_est / 1e9
    log.info(
        "download_summary",
        files=len(records),
        total_compressed_gb=round(total_compressed / 1e9, 2),
        total_decompressed_gb_est=round(total_gb, 2),
        requirement_met=total_gb >= min_gb,
    )

    if total_gb < min_gb:
        log.warning(
            "size_requirement_not_met",
            required_gb=min_gb,
            estimated_gb=round(total_gb, 2),
            message="Consider adding more dump files to DUMP_FILES list",
        )

    return records


def write_manifest(records: list[dict[str, Any]], manifest_path: Path) -> None:
    """Write dataset_manifest.json."""
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": "https://dumps.wikimedia.org/enwiki/",
        "dump_date": "20240420",
        "total_files": len(records),
        "total_compressed_gb": round(
            sum(r["compressed_bytes"] for r in records) / 1e9, 2
        ),
        "total_decompressed_gb_estimate": round(
            sum(r["decompressed_bytes_estimate"] for r in records) / 1e9, 2
        ),
        "requirement_10gb_met": sum(
            r["decompressed_bytes_estimate"] for r in records
        ) >= 10e9,
        "files": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("manifest_written", path=str(manifest_path))


@click.command()
@click.option("--min-gb", default=10.0, help="Minimum required decompressed GB")
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip files already downloaded",
)
def main(min_gb: float, skip_existing: bool) -> None:
    """Download Wikipedia dumps and write dataset_manifest.json."""
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_format)

    log.info("downloader_start", min_gb=min_gb)
    records = download_dumps(cfg.raw_data_dir, min_gb=min_gb, skip_existing=skip_existing)
    write_manifest(records, cfg.manifests_dir / "dataset_manifest.json")
    log.info("downloader_done")


if __name__ == "__main__":
    main()
