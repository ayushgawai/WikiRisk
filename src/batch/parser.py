"""
WikiRisk – Wikipedia XML dump parser.

Streams bz2-compressed Wikipedia XML dumps using iterparse (SAX-style),
extracts revision records, and writes them as Parquet files via Spark.

This avoids loading full XML into memory, enabling processing of
multi-gigabyte dump files on a single machine.
"""
from __future__ import annotations

import bz2
import re
from pathlib import Path
from typing import Iterator

from src.common.logger import get_logger

log = get_logger(__name__)

# Wikipedia XML namespace
NS = "http://www.mediawiki.org/xml/export-0.10/"
_NS = f"{{{NS}}}"

IP_PATTERN = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"       # IPv4
    r"|^[0-9a-fA-F:]+:[0-9a-fA-F:]+$"  # IPv6 (simple)
)


def _is_anon(username: str | None) -> bool:
    """Return True if the contributor appears to be an anonymous (IP) user."""
    if not username:
        return True
    return bool(IP_PATTERN.match(username.strip()))


def iter_revisions(dump_path: Path) -> Iterator[dict]:
    """
    Stream revisions from a bz2-compressed Wikipedia XML dump.

    Yields one dict per revision with fields:
        page_id, page_title, namespace,
        rev_id, parent_rev_id, timestamp,
        contributor_username, is_anon,
        comment, text_bytes
    """
    import xml.etree.ElementTree as ET

    opener = bz2.open if dump_path.suffix == ".bz2" else open
    page_id: str | None = None
    page_title: str | None = None
    namespace: int = 0
    current_rev: dict = {}
    in_revision = False

    with opener(dump_path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            tag = elem.tag.replace(_NS, "")

            if event == "start":
                if tag == "page":
                    page_id = None
                    page_title = None
                    namespace = 0
                elif tag == "revision":
                    in_revision = True
                    current_rev = {}
                elif tag == "contributor" and in_revision:
                    current_rev["_in_contributor"] = True

            elif event == "end":
                if tag == "title" and not in_revision:
                    page_title = (elem.text or "").strip()

                elif tag == "ns" and not in_revision:
                    try:
                        namespace = int(elem.text or 0)
                    except ValueError:
                        namespace = 0

                elif tag == "id" and not in_revision:
                    if page_id is None:
                        page_id = elem.text

                elif tag == "id" and in_revision:
                    if "rev_id" not in current_rev:
                        current_rev["rev_id"] = elem.text

                elif tag == "parentid" and in_revision:
                    current_rev["parent_rev_id"] = elem.text

                elif tag == "timestamp" and in_revision:
                    current_rev["timestamp"] = elem.text

                elif tag == "username" and in_revision:
                    current_rev["contributor_username"] = elem.text

                elif tag == "ip" and in_revision:
                    current_rev["contributor_username"] = elem.text
                    current_rev["is_anon_from_ip_tag"] = True

                elif tag == "contributor" and in_revision:
                    current_rev.pop("_in_contributor", None)

                elif tag == "comment" and in_revision:
                    current_rev["comment"] = (elem.text or "").strip()

                elif tag == "text" and in_revision:
                    text_bytes = elem.get("bytes")
                    try:
                        current_rev["text_bytes"] = int(text_bytes) if text_bytes else 0
                    except (ValueError, TypeError):
                        current_rev["text_bytes"] = len((elem.text or "").encode())

                elif tag == "revision":
                    in_revision = False
                    username = current_rev.get("contributor_username", "")
                    is_anon = current_rev.get("is_anon_from_ip_tag", False) or _is_anon(username)

                    record = {
                        "page_id": page_id or "",
                        "page_title": page_title or "",
                        "namespace": namespace,
                        "rev_id": current_rev.get("rev_id", ""),
                        "parent_rev_id": current_rev.get("parent_rev_id", ""),
                        "timestamp": current_rev.get("timestamp", ""),
                        "contributor_username": username or "",
                        "is_anon": is_anon,
                        "comment": current_rev.get("comment", ""),
                        "text_bytes": current_rev.get("text_bytes", 0),
                    }
                    yield record
                    current_rev = {}

                elif tag == "page":
                    # Free memory for processed page element
                    elem.clear()


def parse_dump_to_parquet(
    dump_path: Path,
    output_dir: Path,
    spark_session=None,
    batch_size: int = 50_000,
) -> int:
    """
    Parse a Wikipedia dump and write revisions to Parquet.

    Returns total number of revisions written.
    """
    from pyspark.sql import Row
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, IntegerType, BooleanType, LongType,
    )

    if spark_session is None:
        from src.common.spark_session import get_spark
        spark = get_spark()
    else:
        spark = spark_session

    schema = StructType([
        StructField("page_id", StringType(), True),
        StructField("page_title", StringType(), True),
        StructField("namespace", IntegerType(), True),
        StructField("rev_id", StringType(), True),
        StructField("parent_rev_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("contributor_username", StringType(), True),
        StructField("is_anon", BooleanType(), True),
        StructField("comment", StringType(), True),
        StructField("text_bytes", LongType(), True),
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / dump_path.stem.replace(".xml", "")

    log.info("parse_start", dump=dump_path.name, output=str(output_file))

    batch: list[dict] = []
    total = 0
    part = 0

    def _flush(records: list[dict], part_idx: int) -> None:
        rows = [Row(**r) for r in records]
        df = spark.createDataFrame(rows, schema=schema)
        part_path = str(output_file) + f"_part{part_idx:04d}"
        df.write.mode("overwrite").parquet(part_path)
        log.info("parquet_part_written", part=part_idx, rows=len(records))

    for rev in iter_revisions(dump_path):
        batch.append(rev)
        total += 1

        if len(batch) >= batch_size:
            _flush(batch, part)
            batch = []
            part += 1

            if total % 500_000 == 0:
                log.info("parse_progress", revisions=total)

    if batch:
        _flush(batch, part)

    log.info("parse_complete", dump=dump_path.name, total_revisions=total)
    return total
