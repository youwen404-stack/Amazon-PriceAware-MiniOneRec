import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_amazon_raw import audit_raw_files, description_non_empty, parse_price
from download_amazon_2023 import build_hf_raw_url, load_review_asins


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class AmazonRawAuditTest(unittest.TestCase):
    def test_build_hf_raw_url_uses_amazon_2023_raw_paths(self) -> None:
        self.assertEqual(
            build_hf_raw_url("Office_Products", "review"),
            "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Office_Products.jsonl",
        )
        self.assertEqual(
            build_hf_raw_url("Office_Products", "meta"),
            "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Office_Products.jsonl",
        )

    def test_load_review_asins_reads_parent_asin_and_asin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reviews.jsonl"
            write_jsonl(
                path,
                [
                    {"user_id": "u1", "parent_asin": "A1"},
                    {"user_id": "u2", "asin": "A2"},
                    {"user_id": "u3", "parent_asin": ""},
                ],
            )

            self.assertEqual(load_review_asins(path), {"A1", "A2"})

    def test_parse_price_accepts_common_amazon_formats(self) -> None:
        self.assertEqual(parse_price("$1,299.50"), 1299.50)
        self.assertEqual(parse_price(" 19.99 "), 19.99)
        self.assertEqual(parse_price(8), 8.0)

    def test_parse_price_rejects_missing_or_non_positive_values(self) -> None:
        self.assertIsNone(parse_price(None))
        self.assertIsNone(parse_price(""))
        self.assertIsNone(parse_price("Currently unavailable"))
        self.assertIsNone(parse_price("$0.00"))
        self.assertIsNone(parse_price("-3"))

    def test_description_non_empty_handles_list_and_string_shapes(self) -> None:
        self.assertTrue(description_non_empty(["", "Ergonomic desk organizer"]))
        self.assertTrue(description_non_empty("Compact stapler for office use"))
        self.assertFalse(description_non_empty([]))
        self.assertFalse(description_non_empty(["", "   "]))
        self.assertFalse(description_non_empty(None))

    def test_audit_raw_files_reports_matching_price_and_text_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reviews_path = tmp_path / "Office_Products.reviews.jsonl"
            meta_path = tmp_path / "meta_Office_Products.jsonl"
            write_jsonl(
                reviews_path,
                [
                    {"user_id": "u1", "parent_asin": "A1", "rating": 5, "timestamp": 3},
                    {"user_id": "u1", "parent_asin": "A2", "rating": 4, "timestamp": 4},
                    {"user_id": "u2", "parent_asin": "A3", "rating": 5, "timestamp": 5},
                ],
            )
            write_jsonl(
                meta_path,
                [
                    {
                        "parent_asin": "A1",
                        "title": "Desk tray",
                        "description": ["Metal mesh tray"],
                        "price": "$12.50",
                    },
                    {
                        "parent_asin": "A2",
                        "title": "Stapler",
                        "description": "",
                        "price": "Currently unavailable",
                    },
                ],
            )

            report = audit_raw_files(reviews_path, meta_path)

        self.assertEqual(report["review_rows_scanned"], 3)
        self.assertEqual(report["unique_users"], 2)
        self.assertEqual(report["unique_review_items"], 3)
        self.assertEqual(report["metadata_items"], 2)
        self.assertEqual(report["review_item_matched_by_meta_ratio"], 2 / 3)
        self.assertEqual(report["price_non_null_ratio"], 0.5)
        self.assertEqual(report["valid_positive_price_ratio"], 0.5)
        self.assertEqual(report["title_non_empty_ratio"], 1.0)
        self.assertEqual(report["description_non_empty_ratio"], 0.5)
        self.assertEqual(report["median_user_interactions_scanned"], 1.5)


if __name__ == "__main__":
    unittest.main()
