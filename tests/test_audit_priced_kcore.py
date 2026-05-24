import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_priced_kcore import audit_category


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class PricedKCoreAuditTest(unittest.TestCase):
    def test_audit_category_filters_unpriced_items_and_reports_kcore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cat_dir = root / "Demo"
            cat_dir.mkdir()
            write_jsonl(
                cat_dir / "meta.jsonl",
                [
                    {"parent_asin": "A", "price": "$10"},
                    {"parent_asin": "B", "price": "$20"},
                    {"parent_asin": "C", "price": "$30"},
                    {"parent_asin": "D", "price": None},
                ],
            )
            write_jsonl(
                cat_dir / "reviews.jsonl",
                [
                    {"user_id": "u1", "parent_asin": "A"},
                    {"user_id": "u1", "parent_asin": "B"},
                    {"user_id": "u2", "parent_asin": "A"},
                    {"user_id": "u2", "parent_asin": "B"},
                    {"user_id": "u3", "parent_asin": "C"},
                    {"user_id": "u4", "parent_asin": "D"},
                ],
            )

            report = audit_category("Demo", root, [2])

        self.assertEqual(report["raw_review_rows"], 6)
        self.assertEqual(report["valid_review_rows"], 6)
        self.assertEqual(report["priced_meta_items"], 3)
        self.assertEqual(report["priced_interactions"], 5)
        self.assertEqual(report["priced_users"], 3)
        self.assertEqual(report["priced_items"], 3)
        self.assertEqual(report["kcore"][0]["kcore_interactions"], 4)
        self.assertEqual(report["kcore"][0]["kcore_users"], 2)
        self.assertEqual(report["kcore"][0]["kcore_items"], 2)


if __name__ == "__main__":
    unittest.main()
