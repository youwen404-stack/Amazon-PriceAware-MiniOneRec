#!/usr/bin/env python3
"""Merge Amazon SFT evaluation shards and recompute full metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_amazon_sft import compute_metrics, load_sid_value_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge evaluation result JSON shards.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="shard_*.json")
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--item-meta-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(
        path
        for path in input_dir.glob(args.pattern)
        if not path.name.endswith(".metrics.json")
    )
    if not paths:
        raise FileNotFoundError(f"No result shards matched {input_dir / args.pattern}")

    records = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            shard = json.load(f)
        if not isinstance(shard, list):
            raise ValueError(f"Expected prediction list in {path}, got {type(shard).__name__}")
        if shard and not isinstance(shard[0], dict):
            raise ValueError(f"Expected prediction records in {path}, got {type(shard[0]).__name__}")
        print(f"loaded {len(shard)} rows from {path}")
        records.extend(shard)

    sid_to_price, sid_to_bucket = load_sid_value_maps(args.info_file, args.item_meta_path)
    metrics = compute_metrics(records, sid_to_price, sid_to_bucket)

    output_json = Path(args.output_json)
    metrics_json = Path(args.metrics_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"merged rows: {len(records)}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
