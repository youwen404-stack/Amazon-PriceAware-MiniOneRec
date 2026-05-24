import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_amazon_minionerec_with_sid import export_with_sid


class AmazonSIDExportTest(unittest.TestCase):
    def test_export_with_sid_adds_history_and_target_sid_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            value_dir = root / "value_splits"
            index_dir = root / "index"
            value_dir.mkdir()
            index_dir.mkdir()
            dataset = "Demo"

            item_meta = {
                "0": {"title": "Alpha"},
                "1": {"title": "Beta"},
                "2": {"title": "Gamma"},
            }
            sid_index = {
                "0": ["<a_0>", "<b_0>", "<c_0>"],
                "1": ["<a_1>", "<b_1>", "<c_1>"],
                "2": ["<a_2>", "<b_2>", "<c_2>"],
            }
            (index_dir / f"{dataset}.item.json").write_text(json.dumps(item_meta), encoding="utf-8")
            sid_path = index_dir / f"{dataset}.index.json"
            sid_path.write_text(json.dumps(sid_index), encoding="utf-8")

            for split in ["train", "valid", "test"]:
                with (value_dir / f"{dataset}.{split}.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "user_id",
                            "history_item_title",
                            "item_title",
                            "history_item_id",
                            "item_id",
                            "price_value",
                            "value_bucket",
                            "value_token",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "user_id": "U0",
                            "history_item_title": "['Alpha', 'Beta']",
                            "item_title": "Gamma",
                            "history_item_id": "['0', '1']",
                            "item_id": "2",
                            "price_value": "39.99",
                            "value_bucket": "2",
                            "value_token": "[VAL_2]",
                        }
                    )

            summary = export_with_sid(root, dataset, sid_path)

            self.assertEqual(summary["splits"]["train"]["written"], 1)
            out_csv = root / "minionerec" / "train" / f"{dataset}.csv"
            with out_csv.open(encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["history_item_sid"], "['<a_0><b_0><c_0>', '<a_1><b_1><c_1>']")
            self.assertEqual(row["item_sid"], "<a_2><b_2><c_2>")
            self.assertTrue((root / "info" / f"{dataset}.txt").exists())


if __name__ == "__main__":
    unittest.main()
