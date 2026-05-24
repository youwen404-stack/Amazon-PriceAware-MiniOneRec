#!/usr/bin/env python3
"""Convert priced Amazon Reviews 2023 data into MiniOneRec-style files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from audit_amazon_raw import iter_jsonl, parse_price


DESCRIPTION_JOIN_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ReviewEdge:
    user_raw_id: str
    item_raw_id: str
    timestamp: int
    rating: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_review_timestamp(row: dict[str, Any]) -> int:
    value = row.get("timestamp")
    if value not in (None, ""):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0
    value = row.get("unixReviewTime")
    if value not in (None, ""):
        try:
            return int(float(value) * 1000)
        except (TypeError, ValueError):
            return 0
    return 0


def parse_rating(row: dict[str, Any]) -> float:
    try:
        return float(row.get("rating", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def clean_text(value: Any, fallback: str = "") -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if str(item).strip())
    text = DESCRIPTION_JOIN_RE.sub(" ", str(value or "").strip())
    return text if text else fallback


def price_bucket(price: float, cutoffs: list[float]) -> int:
    bucket = 0
    for cutoff in cutoffs:
        if price > cutoff:
            bucket += 1
    return bucket


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def load_priced_metadata(meta_path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(meta_path):
        asin = str(row.get("parent_asin") or row.get("asin") or "")
        if not asin:
            continue
        price = parse_price(row.get("price"))
        if price is None:
            continue
        items[asin] = {
            "raw_parent_asin": asin,
            "title": clean_text(row.get("title"), fallback="Amazon product"),
            "description": clean_text(row.get("description")),
            "price": price,
            "main_category": clean_text(row.get("main_category")),
            "categories": row.get("categories", []),
        }
    return items


def load_priced_review_edges(
    reviews_path: Path,
    priced_items: dict[str, dict[str, Any]],
) -> tuple[list[ReviewEdge], dict[str, int], dict[str, int], dict[str, int]]:
    edges: list[ReviewEdge] = []
    user_degree: Counter[str] = Counter()
    item_degree: Counter[str] = Counter()
    stats = {
        "raw_rows": 0,
        "missing_user_or_item": 0,
        "unpriced_item": 0,
    }

    for row in iter_jsonl(reviews_path):
        stats["raw_rows"] += 1
        user_id = str(row.get("user_id") or "")
        asin = str(row.get("parent_asin") or row.get("asin") or "")
        if not user_id or not asin:
            stats["missing_user_or_item"] += 1
            continue
        if asin not in priced_items:
            stats["unpriced_item"] += 1
            continue
        edge = ReviewEdge(
            user_raw_id=user_id,
            item_raw_id=asin,
            timestamp=parse_review_timestamp(row),
            rating=parse_rating(row),
        )
        edges.append(edge)
        user_degree[user_id] += 1
        item_degree[asin] += 1

    return edges, dict(user_degree), dict(item_degree), stats


def apply_k_core(edges: list[ReviewEdge], k: int) -> list[ReviewEdge]:
    kept = edges
    while True:
        user_degree = Counter(edge.user_raw_id for edge in kept)
        item_degree = Counter(edge.item_raw_id for edge in kept)
        next_kept = [
            edge
            for edge in kept
            if user_degree[edge.user_raw_id] >= k and item_degree[edge.item_raw_id] >= k
        ]
        if len(next_kept) == len(kept):
            return next_kept
        kept = next_kept


def stable_item_maps(item_ids: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(set(item_ids))
    item_to_raw = {str(idx): raw_id for idx, raw_id in enumerate(ordered)}
    raw_to_item = {raw_id: item_id for item_id, raw_id in item_to_raw.items()}
    return item_to_raw, raw_to_item


def stable_user_maps(user_ids: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(set(user_ids))
    user_to_raw = {str(idx): raw_id for idx, raw_id in enumerate(ordered)}
    raw_to_user = {raw_id: user_id for user_id, raw_id in user_to_raw.items()}
    return user_to_raw, raw_to_user


def build_item_features(
    priced_items: dict[str, dict[str, Any]],
    item_to_raw: dict[str, str],
    price_cutoffs: list[float],
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for item_id in sorted(item_to_raw, key=lambda x: int(x)):
        raw_id = item_to_raw[item_id]
        item = priced_items[raw_id]
        bucket = price_bucket(float(item["price"]), price_cutoffs)
        copied = dict(item)
        copied.update(
            {
                "raw_parent_asin": raw_id,
                "item_type": "I",
                "price_value": float(item["price"]),
                "item_value_score": float(item["price"]),
                "final_value": float(item["price"]),
                "value_bucket": bucket,
                "item_value_bucket": bucket,
                "value_token": f"[VAL_{bucket}]",
            }
        )
        features[item_id] = copied
    return features


def split_bounds(length: int, train_ratio: float, valid_ratio: float) -> tuple[int, int]:
    train_end = int(length * train_ratio)
    valid_end = int(length * (train_ratio + valid_ratio))
    train_end = min(max(train_end, 1), length)
    valid_end = min(max(valid_end, train_end + 1), length)
    return train_end, valid_end


def select_train_indices(candidates: list[int], max_targets: int) -> list[int]:
    if max_targets <= 0 or len(candidates) <= max_targets:
        return candidates
    if max_targets == 1:
        return [candidates[-1]]
    step = (len(candidates) - 1) / (max_targets - 1)
    return [candidates[round(i * step)] for i in range(max_targets)]


def build_examples(
    edges: list[ReviewEdge],
    raw_to_item: dict[str, str],
    raw_to_user: dict[str, str],
    item_features: dict[str, dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    min_history_len: int,
    max_history_len: int,
    max_train_targets_per_user: int,
) -> dict[str, list[dict[str, Any]]]:
    by_user: dict[str, list[ReviewEdge]] = defaultdict(list)
    for edge in edges:
        by_user[edge.user_raw_id].append(edge)
    for user_edges in by_user.values():
        user_edges.sort(key=lambda x: (x.timestamp, x.item_raw_id))

    examples: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for raw_user_id in sorted(by_user):
        user_edges = by_user[raw_user_id]
        if len(user_edges) <= min_history_len:
            continue
        train_end, valid_end = split_bounds(len(user_edges), train_ratio, valid_ratio)
        split_targets = {
            "train": select_train_indices(
                list(range(min_history_len, train_end)),
                max_train_targets_per_user,
            ),
            "valid": [idx for idx in [train_end] if idx < valid_end and idx >= min_history_len],
            "test": [idx for idx in [valid_end] if idx < len(user_edges) and idx >= min_history_len],
        }

        for split, target_indices in split_targets.items():
            for target_idx in target_indices:
                history = user_edges[max(0, target_idx - max_history_len) : target_idx]
                target = user_edges[target_idx]
                history_item_ids = [raw_to_item[edge.item_raw_id] for edge in history]
                target_item_id = raw_to_item[target.item_raw_id]
                target_feature = item_features[target_item_id]
                history_titles = [item_features[item_id]["title"] for item_id in history_item_ids]
                examples[split].append(
                    {
                        "user_id": f"U{raw_to_user[raw_user_id]}",
                        "raw_user_id": raw_user_id,
                        "history_item_title": history_titles,
                        "item_title": target_feature["title"],
                        "history_item_id": history_item_ids,
                        "item_id": target_item_id,
                        "history_raw_parent_asin": [edge.item_raw_id for edge in history],
                        "raw_parent_asin": target.item_raw_id,
                        "rating": target.rating,
                        "price_value": target_feature["price_value"],
                        "item_value_score": target_feature["item_value_score"],
                        "final_value": target_feature["final_value"],
                        "value_bucket": target_feature["value_bucket"],
                        "value_token": target_feature["value_token"],
                        "target_timestamp": target.timestamp,
                    }
                )
    return examples


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_inter_file(path: Path, examples: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["user_id", "item_sequence", "target_item"])
        for row in examples:
            writer.writerow([row["user_id"], " ".join(row["history_item_id"]), row["item_id"]])


def write_value_csv(path: Path, examples: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "user_id",
        "raw_user_id",
        "history_item_title",
        "item_title",
        "history_item_id",
        "item_id",
        "history_raw_parent_asin",
        "raw_parent_asin",
        "rating",
        "price_value",
        "item_value_score",
        "final_value",
        "value_bucket",
        "value_token",
        "target_timestamp",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in examples:
            writer.writerow({key: row[key] for key in fieldnames})


def convert_category(
    category: str,
    data_root: Path,
    output_dir: Path,
    dataset_name: str,
    k_core: int,
    min_history_len: int,
    max_history_len: int,
    max_train_targets_per_user: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, Any]:
    category_dir = data_root / category
    reviews_path = category_dir / "reviews.jsonl"
    meta_path = category_dir / "meta.jsonl"
    if not reviews_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Expected reviews/meta under {category_dir}")

    print(f"Loading priced metadata: {meta_path}", flush=True)
    priced_items = load_priced_metadata(meta_path)
    print(f"Priced metadata items: {len(priced_items)}", flush=True)

    print(f"Loading priced reviews: {reviews_path}", flush=True)
    priced_edges, priced_user_degree, priced_item_degree, review_stats = load_priced_review_edges(
        reviews_path,
        priced_items,
    )
    print(
        f"Priced interactions: {len(priced_edges)} users={len(priced_user_degree)} items={len(priced_item_degree)}",
        flush=True,
    )

    print(f"Applying {k_core}-core...", flush=True)
    core_edges = apply_k_core(priced_edges, k_core)
    core_user_ids = {edge.user_raw_id for edge in core_edges}
    core_item_ids = {edge.item_raw_id for edge in core_edges}
    print(
        f"K-core kept interactions={len(core_edges)} users={len(core_user_ids)} items={len(core_item_ids)}",
        flush=True,
    )

    item_to_raw, raw_to_item = stable_item_maps(core_item_ids)
    user_to_raw, raw_to_user = stable_user_maps(core_user_ids)
    price_values = sorted(float(priced_items[raw_id]["price"]) for raw_id in core_item_ids)
    price_cutoffs = [quantile(price_values, q) for q in (0.25, 0.5, 0.75)]
    item_features = build_item_features(priced_items, item_to_raw, price_cutoffs)

    print("Building chronological examples...", flush=True)
    examples = build_examples(
        core_edges,
        raw_to_item=raw_to_item,
        raw_to_user=raw_to_user,
        item_features=item_features,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        max_train_targets_per_user=max_train_targets_per_user,
    )

    index_dir = output_dir / "index"
    inter_dir = output_dir / "inter"
    value_dir = output_dir / "value_splits"
    write_json(index_dir / f"{dataset_name}.item.json", item_features)
    write_json(index_dir / f"{dataset_name}.item_id_map.json", item_to_raw)
    write_json(index_dir / f"{dataset_name}.raw_item_id_map.json", raw_to_item)
    write_json(index_dir / f"{dataset_name}.user_id_map.json", user_to_raw)
    write_json(index_dir / f"{dataset_name}.raw_user_id_map.json", raw_to_user)

    for split, rows in examples.items():
        write_inter_file(inter_dir / f"{dataset_name}.{split}.inter", rows)
        write_value_csv(value_dir / f"{dataset_name}.{split}.csv", rows)
        print(f"Wrote {split}: {len(rows)} examples", flush=True)

    manifest = {
        "dataset_name": dataset_name,
        "category": category,
        "reviews_path": str(reviews_path),
        "meta_path": str(meta_path),
        "k_core": {
            "k": k_core,
            "interactions": len(core_edges),
            "users": len(core_user_ids),
            "items": len(core_item_ids),
        },
        "priced": {
            "meta_items": len(priced_items),
            "interactions": len(priced_edges),
            "users": len(priced_user_degree),
            "items": len(priced_item_degree),
        },
        "raw_review_stats": review_stats,
        "price_cutoffs": price_cutoffs,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "min_history_len": min_history_len,
        "max_history_len": max_history_len,
        "max_train_targets_per_user": max_train_targets_per_user,
        "split_counts": {split: len(rows) for split, rows in examples.items()},
        "output_layout": {
            "item_json": str(index_dir / f"{dataset_name}.item.json"),
            "inter_dir": str(inter_dir),
            "value_splits_dir": str(value_dir),
        },
    }
    write_json(output_dir / f"{dataset_name}.manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Amazon Reviews 2023 to MiniOneRec-style data.")
    parser.add_argument("--category", default="Automotive")
    parser.add_argument(
        "--data-root",
        default="data/raw_amazon_2023",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_price_aware",
    )
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--min-history-len", type=int, default=3)
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--max-train-targets-per-user", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or f"Amazon_{args.category}_priced_{args.k_core}core"
    manifest = convert_category(
        category=args.category,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir) / dataset_name,
        dataset_name=dataset_name,
        k_core=args.k_core,
        min_history_len=args.min_history_len,
        max_history_len=args.max_history_len,
        max_train_targets_per_user=args.max_train_targets_per_user,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    print("=== Manifest ===")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
