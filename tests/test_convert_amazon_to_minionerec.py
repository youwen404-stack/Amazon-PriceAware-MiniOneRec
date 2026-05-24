import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from convert_amazon_to_minionerec import convert_category, parse_review_timestamp


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class AmazonMiniOneRecConversionTest(unittest.TestCase):
    def test_parse_review_timestamp_prefers_milliseconds(self) -> None:
        self.assertEqual(parse_review_timestamp({"timestamp": 1700000000123}), 1700000000123)
        self.assertEqual(parse_review_timestamp({"unixReviewTime": 1700000000}), 1700000000000)
        self.assertEqual(parse_review_timestamp({}), 0)

    def test_convert_category_filters_priced_kcore_and_writes_minionerec_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw"
            category_dir = raw / "Demo"
            category_dir.mkdir(parents=True)
            write_jsonl(
                category_dir / "meta.jsonl",
                [
                    {"parent_asin": "A", "title": "Alpha", "description": ["First"], "price": "$10"},
                    {"parent_asin": "B", "title": "Beta", "description": ["Second"], "price": "$20"},
                    {"parent_asin": "C", "title": "Gamma", "description": ["Third"], "price": "$30"},
                    {"parent_asin": "D", "title": "Delta", "description": ["No price"], "price": None},
                ],
            )
            write_jsonl(
                category_dir / "reviews.jsonl",
                [
                    {"user_id": "u1", "parent_asin": "A", "rating": 5, "timestamp": 1},
                    {"user_id": "u1", "parent_asin": "B", "rating": 4, "timestamp": 2},
                    {"user_id": "u1", "parent_asin": "C", "rating": 5, "timestamp": 3},
                    {"user_id": "u2", "parent_asin": "A", "rating": 5, "timestamp": 1},
                    {"user_id": "u2", "parent_asin": "B", "rating": 4, "timestamp": 2},
                    {"user_id": "u2", "parent_asin": "C", "rating": 3, "timestamp": 3},
                    {"user_id": "u3", "parent_asin": "D", "rating": 5, "timestamp": 4},
                ],
            )

            manifest = convert_category(
                category="Demo",
                data_root=raw,
                output_dir=root / "processed",
                dataset_name="DemoPriceAware",
                k_core=2,
                min_history_len=1,
                max_history_len=5,
                max_train_targets_per_user=10,
                train_ratio=0.5,
                valid_ratio=0.25,
            )

            train_csv = root / "processed" / "value_splits" / "DemoPriceAware.train.csv"
            item_json = root / "processed" / "index" / "DemoPriceAware.item.json"

            self.assertEqual(manifest["k_core"]["interactions"], 6)
            self.assertEqual(manifest["k_core"]["users"], 2)
            self.assertEqual(manifest["k_core"]["items"], 3)
            self.assertTrue(train_csv.exists())
            self.assertTrue(item_json.exists())
            self.assertIn("history_item_title", train_csv.read_text(encoding="utf-8").splitlines()[0])
            item_features = json.loads(item_json.read_text(encoding="utf-8"))
            self.assertEqual(len(item_features), 3)
            self.assertEqual(item_features["0"]["raw_parent_asin"], "A")
            self.assertEqual(item_features["0"]["value_token"], "[VAL_0]")


if __name__ == "__main__":
    unittest.main()
