#!/usr/bin/env python3
"""Download one Amazon Reviews 2023 category to MiniOneRec-friendly filenames."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


HF_DATASET_ROOT = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
CONTENT_RANGE_RE = re.compile(r"bytes \d+-\d+/(\d+|\*)")


def build_hf_raw_url(category: str, kind: str) -> str:
    if kind == "review":
        return f"{HF_DATASET_ROOT}/raw/review_categories/{category}.jsonl"
    if kind == "meta":
        return f"{HF_DATASET_ROOT}/raw/meta_categories/meta_{category}.jsonl"
    raise ValueError(f"Unsupported raw Amazon file kind: {kind}")


def parse_total_size(content_range: str) -> Optional[int]:
    match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
    if not match or match.group(1) == "*":
        return None
    return int(match.group(1))


def iter_url_lines(url: str, chunk_bytes: int = 1024 * 1024, retries: int = 3) -> Iterable[bytes]:
    try:
        import requests
    except ModuleNotFoundError:
        with urllib.request.urlopen(url, timeout=60) as response:
            yield from response
        return

    start = 0
    pending = b""
    total_size: Optional[int] = None

    while True:
        end = start + chunk_bytes - 1
        headers = {"Range": f"bytes={start}-{end}"}
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, headers=headers, timeout=60) as response:
                    response.raise_for_status()
                    if response.status_code == 200 and start > 0:
                        raise RuntimeError("Server did not honor Range requests")
                    if response.status_code == 200:
                        total_size = int(response.headers.get("Content-Length", "0")) or None
                    else:
                        total_size = parse_total_size(response.headers.get("Content-Range", ""))
                    data = response.content
                break
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                time.sleep(attempt)
        else:
            raise RuntimeError("unreachable range download state") from last_error

        if not data:
            break
        combined = pending + data
        lines = combined.split(b"\n")
        pending = lines.pop() if not combined.endswith(b"\n") else b""
        for line in lines:
            if line:
                yield line + b"\n"

        start += len(data)
        if total_size is not None and start >= total_size:
            break

    if pending.strip():
        yield pending + b"\n"


def stream_jsonl_sample(url: str, output_path: Path, max_rows: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_path.open("wb") as out:
        for line in iter_url_lines(url):
            if not line.strip():
                continue
            out.write(line)
            rows += 1
            if rows >= max_rows:
                break
    return rows


def load_review_asins(path: Path) -> set[str]:
    asins: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            asin = str(row.get("parent_asin") or row.get("asin") or "")
            if asin:
                asins.add(asin)
    return asins


def stream_matched_meta_sample(
    category: str,
    review_path: Path,
    output_path: Path,
    max_scan_rows: int,
) -> tuple[int, int, int]:
    wanted_asins = load_review_asins(review_path)
    matched_asins: set[str] = set()
    scanned_rows = 0
    written_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out:
        for line in iter_url_lines(build_hf_raw_url(category, "meta")):
            if max_scan_rows > 0 and scanned_rows >= max_scan_rows:
                break
            scanned_rows += 1
            row = json.loads(line)
            asin = str(row.get("parent_asin") or row.get("asin") or "")
            if asin in wanted_asins:
                out.write(line)
                matched_asins.add(asin)
                written_rows += 1
                if len(matched_asins) == len(wanted_asins):
                    break
    return scanned_rows, written_rows, len(wanted_asins)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument(
        "--stream_sample_rows",
        type=int,
        default=-1,
        help="Stream the first N review rows and N metadata rows from HF raw files without downloading full category files.",
    )
    parser.add_argument(
        "--stream_meta_sample_rows",
        type=int,
        default=-1,
        help="Override metadata rows for stream sample mode. Defaults to --stream_sample_rows.",
    )
    parser.add_argument(
        "--match_meta_from_reviews",
        default="",
        help="Stream metadata and keep only rows whose parent_asin appears in this local review JSONL.",
    )
    parser.add_argument(
        "--match_meta_max_scan_rows",
        type=int,
        default=250000,
        help="Maximum metadata rows to scan when --match_meta_from_reviews is used. Use -1 for no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    review_path = out / f"{args.category}.reviews.jsonl"
    meta_path = out / f"meta_{args.category}.jsonl"

    if args.match_meta_from_reviews:
        scanned_rows, written_rows, wanted_rows = stream_matched_meta_sample(
            args.category,
            Path(args.match_meta_from_reviews),
            meta_path,
            args.match_meta_max_scan_rows,
        )
        print("metadata rows scanned:", scanned_rows)
        print("matched meta rows written:", written_rows)
        print("target review unique items:", wanted_rows)
        print("meta file:", meta_path)
        print("matched metadata mode: True")
        return

    if args.stream_sample_rows > 0:
        meta_rows = args.stream_meta_sample_rows
        if meta_rows <= 0:
            meta_rows = args.stream_sample_rows
        review_rows = stream_jsonl_sample(
            build_hf_raw_url(args.category, "review"),
            review_path,
            args.stream_sample_rows,
        )
        meta_rows = stream_jsonl_sample(
            build_hf_raw_url(args.category, "meta"),
            meta_path,
            meta_rows,
        )
        print("review rows:", review_rows)
        print("meta rows:", meta_rows)
        print("review file:", review_path)
        print("meta file:", meta_path)
        print("stream sample mode: True")
        return

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install HuggingFace datasets before "
            "running the downloader."
        ) from exc

    review_name = f"raw_review_{args.category}"
    meta_name = f"raw_meta_{args.category}"

    review = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        review_name,
        split="full",
        trust_remote_code=True,
    )
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        meta_name,
        split="full",
        trust_remote_code=True,
    )

    if args.max_rows > 0:
        review = review.select(range(min(args.max_rows, len(review))))

    review.to_json(str(review_path), force_ascii=False)
    meta.to_json(str(meta_path), force_ascii=False)

    print("review rows:", len(review))
    print("meta rows:", len(meta))
    print("review file:", review_path)
    print("meta file:", meta_path)


if __name__ == "__main__":
    main()
