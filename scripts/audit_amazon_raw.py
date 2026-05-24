#!/usr/bin/env python3
"""Audit raw Amazon Reviews 2023 review and metadata JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


PRICE_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def iter_jsonl(path: Path | str) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        price = float(value)
        return price if price > 0 else None

    text = str(value).replace("$", "").strip()
    if not text:
        return None
    match = PRICE_NUMBER_RE.search(text)
    if not match:
        return None
    try:
        price = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return price if price > 0 else None


def text_non_empty(value: Any) -> bool:
    return bool(str(value or "").strip())


def description_non_empty(value: Any) -> bool:
    if isinstance(value, list):
        return any(text_non_empty(item) for item in value)
    return text_non_empty(value)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def price_quantiles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [quantile(ordered, q) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99]]


def audit_raw_files(
    reviews_path: Path | str,
    meta_path: Path | str,
    max_scan_reviews: int = -1,
) -> dict[str, Any]:
    meta_items: dict[str, dict[str, Any]] = {}
    priced_meta_asins: set[str] = set()
    price_values: list[float] = []
    title_nonempty = 0
    desc_nonempty = 0

    for item in iter_jsonl(meta_path):
        asin = str(item.get("parent_asin") or item.get("asin") or "")
        if not asin:
            continue
        meta_items[asin] = item
        price = parse_price(item.get("price"))
        if price is not None:
            price_values.append(price)
            priced_meta_asins.add(asin)
        if text_non_empty(item.get("title")):
            title_nonempty += 1
        if description_non_empty(item.get("description")):
            desc_nonempty += 1

    users: Counter[str] = Counter()
    items: Counter[str] = Counter()
    ratings: Counter[str] = Counter()
    matched = 0
    priced_review_rows = 0
    matched_priced_review_rows = 0
    review_rows = 0

    for row in iter_jsonl(reviews_path):
        if max_scan_reviews > 0 and review_rows >= max_scan_reviews:
            break
        user = str(row.get("user_id") or "")
        asin = str(row.get("parent_asin") or row.get("asin") or "")
        if not user or not asin:
            continue
        review_rows += 1
        users[user] += 1
        items[asin] += 1
        is_matched = asin in meta_items
        is_priced = asin in priced_meta_asins
        matched += int(is_matched)
        priced_review_rows += int(is_priced)
        matched_priced_review_rows += int(is_matched and is_priced)
        ratings[str(row.get("rating"))] += 1

    metadata_count = len(meta_items)
    user_lengths = list(users.values())
    return {
        "review_rows_scanned": review_rows,
        "unique_users": len(users),
        "unique_review_items": len(items),
        "metadata_items": metadata_count,
        "review_item_matched_by_meta_ratio": matched / max(review_rows, 1),
        "priced_review_rows": priced_review_rows,
        "priced_review_row_ratio": priced_review_rows / max(review_rows, 1),
        "matched_priced_review_rows": matched_priced_review_rows,
        "matched_priced_review_row_ratio": matched_priced_review_rows / max(review_rows, 1),
        "unique_priced_review_items": sum(1 for asin in items if asin in priced_meta_asins),
        "unique_priced_review_item_ratio": (
            sum(1 for asin in items if asin in priced_meta_asins) / max(len(items), 1)
        ),
        "price_non_null_ratio": len(price_values) / max(metadata_count, 1),
        "valid_positive_price_ratio": len(price_values) / max(metadata_count, 1),
        "title_non_empty_ratio": title_nonempty / max(metadata_count, 1),
        "description_non_empty_ratio": desc_nonempty / max(metadata_count, 1),
        "price_quantiles": price_quantiles(price_values),
        "rating_distribution": dict(ratings.most_common()),
        "median_user_interactions_scanned": median(user_lengths) if user_lengths else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--max_scan_reviews", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_raw_files(args.reviews, args.meta, args.max_scan_reviews)

    print("=== Raw Amazon Audit ===")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
