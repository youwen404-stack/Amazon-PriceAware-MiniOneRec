#!/usr/bin/env python3
"""Add MiniOneRec SID columns to Amazon value split CSV files."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-like value, got: {value!r}")
    return [str(item) for item in parsed]


def sid_to_text(tokens: list[str]) -> str:
    return "".join(tokens)


def write_info_file(path: Path, item_meta: dict[str, dict[str, Any]], sid_index: dict[str, list[str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for item_id in sorted(item_meta, key=lambda x: int(x)):
            if item_id not in sid_index:
                continue
            title = str(item_meta[item_id].get("title", "")).replace("\t", " ").strip()
            f.write(f"{sid_to_text(sid_index[item_id])}\t{title}\t{item_id}\n")
            count += 1
    return count


def export_split(
    input_csv: Path,
    output_csv: Path,
    sid_index: dict[str, list[str]],
) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with input_csv.open(encoding="utf-8", newline="") as f_in, output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {input_csv}")
        fieldnames = list(reader.fieldnames)
        for extra in ["history_item_sid", "item_sid"]:
            if extra not in fieldnames:
                fieldnames.append(extra)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            history_item_ids = parse_list(row["history_item_id"])
            target_item_id = str(row["item_id"])
            if target_item_id not in sid_index or any(item_id not in sid_index for item_id in history_item_ids):
                skipped += 1
                continue
            row["history_item_sid"] = [sid_to_text(sid_index[item_id]) for item_id in history_item_ids]
            row["item_sid"] = sid_to_text(sid_index[target_item_id])
            writer.writerow(row)
            written += 1

    return {"written": written, "skipped": skipped}


def export_with_sid(processed_dir: Path | str, dataset_name: str, sid_index_path: Path | str) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    sid_index_path = Path(sid_index_path)
    sid_index = {str(item_id): list(tokens) for item_id, tokens in load_json(sid_index_path).items()}
    item_meta_path = processed_dir / "index" / f"{dataset_name}.item.json"
    item_meta = {str(item_id): dict(value) for item_id, value in load_json(item_meta_path).items()}
    expected_ids = {str(i) for i in range(len(item_meta))}
    if set(item_meta) != expected_ids:
        raise ValueError("item metadata keys must be contiguous internal item IDs")
    if set(sid_index) != expected_ids:
        raise ValueError("SID index keys do not match item metadata keys")

    summary = {
        "dataset_name": dataset_name,
        "processed_dir": str(processed_dir),
        "sid_index_path": str(sid_index_path),
        "splits": {},
    }
    for split in ["train", "valid", "test"]:
        input_csv = processed_dir / "value_splits" / f"{dataset_name}.{split}.csv"
        output_csv = processed_dir / "minionerec" / split / f"{dataset_name}.csv"
        summary["splits"][split] = export_split(input_csv, output_csv, sid_index)

    info_count = write_info_file(processed_dir / "info" / f"{dataset_name}.txt", item_meta, sid_index)
    summary["info_items"] = info_count
    write_json(processed_dir / f"{dataset_name}.sid_export_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--sid-index-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_with_sid(
        processed_dir=args.processed_dir,
        dataset_name=args.dataset_name,
        sid_index_path=args.sid_index_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
