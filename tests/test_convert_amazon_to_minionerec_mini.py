import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from convert_amazon_to_minionerec_mini import convert_mini_category


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class AmazonMiniSubsetConversionTest(unittest.TestCase):
    def test_popular_subset_recores_and_reindexes_items(self) -> None:
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
                    {"parent_asin": "D", "title": "Delta", "description": ["Fourth"], "price": "$40"},
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
                    {"user_id": "u3", "parent_asin": "A", "rating": 5, "timestamp": 1},
                    {"user_id": "u3", "parent_asin": "B", "rating": 4, "timestamp": 2},
                    {"user_id": "u3", "parent_asin": "D", "rating": 3, "timestamp": 3},
                    {"user_id": "u4", "parent_asin": "D", "rating": 5, "timestamp": 1},
                ],
            )

            manifest = convert_mini_category(
                category="Demo",
                data_root=raw,
                output_dir=root / "processed",
                dataset_name="DemoMini",
                k_core=2,
                candidate_items=3,
                min_history_len=1,
                max_history_len=5,
                max_train_targets_per_user=10,
                train_ratio=0.5,
                valid_ratio=0.25,
            )

            item_json = root / "processed" / "index" / "DemoMini.item.json"
            raw_item_map = root / "processed" / "index" / "DemoMini.raw_item_id_map.json"
            train_csv = root / "processed" / "value_splits" / "DemoMini.train.csv"

            self.assertEqual(manifest["mini_selection"]["candidate_items"], 3)
            self.assertEqual(manifest["mini_selection"]["pre_kcore_items"], 3)
            self.assertEqual(manifest["k_core"]["items"], 3)
            self.assertTrue(train_csv.exists())
            item_features = json.loads(item_json.read_text(encoding="utf-8"))
            raw_to_item = json.loads(raw_item_map.read_text(encoding="utf-8"))
            self.assertEqual(set(item_features), {"0", "1", "2"})
            self.assertEqual(set(raw_to_item), {"A", "B", "C"})
            self.assertNotIn("D", raw_to_item)


if __name__ == "__main__":
    unittest.main()
