#!/usr/bin/env python3
"""Audit priced-only Amazon categories after user/item k-core filtering."""

from __future__ import annotations

import argparse
import json
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from audit_amazon_raw import iter_jsonl, parse_price


DEFAULT_CATEGORIES = [
    "Automotive",
    "Industrial_and_Scientific",
    "Tools_and_Home_Improvement",
    "Health_and_Household",
    "Toys_and_Games",
]


def load_priced_asins(meta_path: Path) -> set[str]:
    priced_asins: set[str] = set()
    for row in iter_jsonl(meta_path):
        asin = str(row.get("parent_asin") or row.get("asin") or "")
        if asin and parse_price(row.get("price")) is not None:
            priced_asins.add(asin)
    return priced_asins


def load_priced_edges(
    reviews_path: Path,
    priced_asins: set[str],
) -> tuple[array, array, int, int, int, int]:
    user_ids: dict[str, int] = {}
    item_ids: dict[str, int] = {}
    edge_users = array("I")
    edge_items = array("I")
    raw_rows = 0
    valid_rows = 0

    for row in iter_jsonl(reviews_path):
        raw_rows += 1
        user = str(row.get("user_id") or "")
        asin = str(row.get("parent_asin") or row.get("asin") or "")
        if not user or not asin:
            continue
        valid_rows += 1
        if asin not in priced_asins:
            continue

        user_id = user_ids.setdefault(user, len(user_ids))
        item_id = item_ids.setdefault(asin, len(item_ids))
        edge_users.append(user_id)
        edge_items.append(item_id)

    return edge_users, edge_items, len(user_ids), len(item_ids), raw_rows, valid_rows


def degree_quantiles(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}

    def q(pos: float) -> int:
        idx = int((len(ordered) - 1) * pos)
        return ordered[idx]

    return {
        "min": ordered[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def kcore_stats(
    edge_users: array,
    edge_items: array,
    num_users: int,
    num_items: int,
    k: int,
) -> dict[str, Any]:
    active_users = bytearray(b"\x01") * num_users
    active_items = bytearray(b"\x01") * num_items
    iteration_count = 0

    while True:
        iteration_count += 1
        user_degree = [0] * num_users
        item_degree = [0] * num_items
        kept_edges = 0

        for user_id, item_id in zip(edge_users, edge_items):
            if active_users[user_id] and active_items[item_id]:
                user_degree[user_id] += 1
                item_degree[item_id] += 1
                kept_edges += 1

        changed = False
        for user_id, degree in enumerate(user_degree):
            if active_users[user_id] and degree < k:
                active_users[user_id] = 0
                changed = True
        for item_id, degree in enumerate(item_degree):
            if active_items[item_id] and degree < k:
                active_items[item_id] = 0
                changed = True

        if not changed:
            break

    kept_user_degrees = [degree for user_id, degree in enumerate(user_degree) if active_users[user_id]]
    kept_item_degrees = [degree for item_id, degree in enumerate(item_degree) if active_items[item_id]]

    return {
        "k": k,
        "kcore_users": len(kept_user_degrees),
        "kcore_items": len(kept_item_degrees),
        "kcore_interactions": kept_edges,
        "kcore_interaction_ratio": kept_edges / max(len(edge_users), 1),
        "iterations": iteration_count,
        "user_degree": degree_quantiles(kept_user_degrees),
        "item_degree": degree_quantiles(kept_item_degrees),
    }


def audit_category(category: str, data_root: Path, k_values: list[int]) -> dict[str, Any]:
    category_dir = data_root / category
    reviews_path = category_dir / "reviews.jsonl"
    meta_path = category_dir / "meta.jsonl"

    if not reviews_path.exists() or not meta_path.exists():
        missing = [str(path) for path in [reviews_path, meta_path] if not path.exists()]
        raise FileNotFoundError(f"{category} missing required files: {missing}")

    priced_asins = load_priced_asins(meta_path)
    edge_users, edge_items, num_users, num_items, raw_rows, valid_rows = load_priced_edges(
        reviews_path,
        priced_asins,
    )

    priced_user_counts = Counter(edge_users)
    priced_item_counts = Counter(edge_items)
    return {
        "category": category,
        "raw_review_rows": raw_rows,
        "valid_review_rows": valid_rows,
        "priced_meta_items": len(priced_asins),
        "priced_interactions": len(edge_users),
        "priced_interaction_ratio": len(edge_users) / max(valid_rows, 1),
        "priced_users": num_users,
        "priced_items": num_items,
        "priced_user_degree": degree_quantiles(priced_user_counts.values()),
        "priced_item_degree": degree_quantiles(priced_item_counts.values()),
        "kcore": [
            kcore_stats(edge_users, edge_items, num_users, num_items, k)
            for k in k_values
        ],
    }


def parse_k_values(value: str) -> list[int]:
    k_values = sorted({int(part) for part in value.split(",") if part.strip()})
    if not k_values or any(k <= 0 for k in k_values):
        raise argparse.ArgumentTypeError("--k-values must contain positive integers")
    return k_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/raw_amazon_2023",
        help="Directory containing <category>/reviews.jsonl and <category>/meta.jsonl.",
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("5,10"))
    parser.add_argument("--jsonl-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    reports = [audit_category(category, data_root, args.k_values) for category in args.categories]

    if args.jsonl_output:
        output_path = Path(args.jsonl_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for report in reports:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")

    print(
        "\t".join(
            [
                "category",
                "priced_interactions",
                "priced_users",
                "priced_items",
                "priced_ratio",
                "k",
                "kcore_interactions",
                "kcore_users",
                "kcore_items",
                "kcore_ratio",
            ]
        )
    )
    for report in reports:
        for core in report["kcore"]:
            print(
                "\t".join(
                    [
                        str(report["category"]),
                        str(report["priced_interactions"]),
                        str(report["priced_users"]),
                        str(report["priced_items"]),
                        f"{report['priced_interaction_ratio']:.6f}",
                        str(core["k"]),
                        str(core["kcore_interactions"]),
                        str(core["kcore_users"]),
                        str(core["kcore_items"]),
                        f"{core['kcore_interaction_ratio']:.6f}",
                    ]
                )
            )


if __name__ == "__main__":
    main()
