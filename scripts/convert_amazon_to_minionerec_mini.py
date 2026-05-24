#!/usr/bin/env python3
"""Build a MiniOneRec-scale priced Amazon subset with consistent k-core splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from convert_amazon_to_minionerec import (
    apply_k_core,
    build_examples,
    build_item_features,
    load_priced_metadata,
    load_priced_review_edges,
    quantile,
    stable_item_maps,
    stable_user_maps,
    write_inter_file,
    write_json,
    write_value_csv,
)


def select_top_items(edges, candidate_items: int) -> set[str]:
    item_degree = Counter(edge.item_raw_id for edge in edges)
    ranked = sorted(item_degree.items(), key=lambda pair: (-pair[1], pair[0]))
    return {item_id for item_id, _ in ranked[:candidate_items]}


def filter_edges_by_items(edges, allowed_items: set[str]):
    return [edge for edge in edges if edge.item_raw_id in allowed_items]


def convert_mini_category(
    category: str,
    data_root: Path,
    output_dir: Path,
    dataset_name: str,
    k_core: int,
    candidate_items: int,
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
    if candidate_items <= 0:
        raise ValueError("candidate_items must be positive")

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

    selected_items = select_top_items(priced_edges, candidate_items)
    subset_edges = filter_edges_by_items(priced_edges, selected_items)
    subset_user_ids = {edge.user_raw_id for edge in subset_edges}
    print(
        f"Popular subset before k-core: interactions={len(subset_edges)} "
        f"users={len(subset_user_ids)} items={len(selected_items)}",
        flush=True,
    )

    print(f"Applying {k_core}-core on popular subset...", flush=True)
    core_edges = apply_k_core(subset_edges, k_core)
    core_user_ids = {edge.user_raw_id for edge in core_edges}
    core_item_ids = {edge.item_raw_id for edge in core_edges}
    print(
        f"Mini k-core kept interactions={len(core_edges)} users={len(core_user_ids)} items={len(core_item_ids)}",
        flush=True,
    )
    if not core_edges:
        raise ValueError("Mini subset became empty after k-core; increase candidate_items or lower k_core")

    item_to_raw, raw_to_item = stable_item_maps(core_item_ids)
    user_to_raw, raw_to_user = stable_user_maps(core_user_ids)
    price_values = sorted(float(priced_items[raw_id]["price"]) for raw_id in core_item_ids)
    price_cutoffs = [quantile(price_values, q) for q in (0.25, 0.5, 0.75)]
    item_features = build_item_features(priced_items, item_to_raw, price_cutoffs)

    print("Building chronological mini examples...", flush=True)
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
        "mini_selection": {
            "method": "top_priced_items_then_recore",
            "candidate_items": candidate_items,
            "pre_kcore_interactions": len(subset_edges),
            "pre_kcore_users": len(subset_user_ids),
            "pre_kcore_items": len(selected_items),
        },
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
    parser = argparse.ArgumentParser(description="Convert a popular Amazon subset to MiniOneRec-style data.")
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--data-root", default="/home/youwen/data/rec/amazon_price_aware/raw_amazon_2023")
    parser.add_argument("--output-dir", default="/home/youwen/data/rec/amazon_price_aware/processed_price_aware")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--candidate-items", type=int, default=5000)
    parser.add_argument("--min-history-len", type=int, default=3)
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--max-train-targets-per-user", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or (
        f"Amazon_{args.category}_priced_{args.k_core}core_mini{args.candidate_items}"
    )
    manifest = convert_mini_category(
        category=args.category,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir) / dataset_name,
        dataset_name=dataset_name,
        k_core=args.k_core,
        candidate_items=args.candidate_items,
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
