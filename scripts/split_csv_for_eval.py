#!/usr/bin/env python3
"""Split a CSV file into shard CSV files for parallel evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a CSV into N balanced shards.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--prefix", default="shard")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"No CSV header found in {args.input_path}")

    writers = []
    files = []
    counts = [0 for _ in range(args.num_shards)]
    try:
        for shard_id in range(args.num_shards):
            path = output_dir / f"{args.prefix}_{shard_id}.csv"
            handle = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            files.append(handle)
            writers.append(writer)

        for row_id, row in enumerate(rows):
            shard_id = row_id % args.num_shards
            writers[shard_id].writerow(row)
            counts[shard_id] += 1
    finally:
        for handle in files:
            handle.close()

    print(f"input rows: {len(rows)}")
    for shard_id, count in enumerate(counts):
        print(f"{args.prefix}_{shard_id}.csv rows: {count}")


if __name__ == "__main__":
    main()
