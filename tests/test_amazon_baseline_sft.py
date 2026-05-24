import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from amazon_sft_data import (
    build_fusion_prompt,
    build_history_text,
    build_item_metadata_prompt,
    build_sequence_prompt,
    parse_list,
)


class AmazonBaselineSFTPromptTest(unittest.TestCase):
    def test_parse_list_reads_csv_list_strings(self) -> None:
        self.assertEqual(parse_list("['0', '1']"), ["0", "1"])
        self.assertEqual(parse_list(["2", 3]), ["2", "3"])

    def test_baseline_history_text_uses_sid_only_without_category_or_value(self) -> None:
        item_meta = {
            "0": {"main_category": "Automotive", "value_token": "[VAL_0]"},
            "1": {"categories": ["Automotive", "Replacement Parts"], "value_token": "[VAL_3]"},
        }

        history = build_history_text(
            history_item_ids=["0", "1"],
            history_sids=["<a_0><b_0><c_0>", "<a_1><b_1><c_1>"],
        )

        self.assertIn("<a_0><b_0><c_0>", history)
        self.assertIn("<a_1><b_1><c_1>", history)
        self.assertNotIn("Automotive", history)
        self.assertNotIn("[VAL_", history)

    def test_sequence_prompt_targets_sid_only(self) -> None:
        prompt, target = build_sequence_prompt("sid history", "<a_2><b_2><c_2>")

        self.assertIn("Predict the next product semantic ID", prompt)
        self.assertEqual(target, "<a_2><b_2><c_2>\n")

    def test_item_metadata_task_maps_title_and_sid_without_value(self) -> None:
        item = {"title": "Brake Rotor", "description": "Front disc brake rotor.", "value_token": "[VAL_3]"}

        title_prompt, title_target = build_item_metadata_prompt("title2sid", "<a_1><b_1><c_1>", item)
        sid_prompt, sid_target = build_item_metadata_prompt("sid2title", "<a_1><b_1><c_1>", item)

        self.assertIn("Brake Rotor", title_prompt)
        self.assertEqual(title_target, "<a_1><b_1><c_1>\n")
        self.assertIn("<a_1><b_1><c_1>", sid_prompt)
        self.assertEqual(sid_target, "Brake Rotor\n")
        self.assertNotIn("[VAL_", title_prompt + sid_prompt)

    def test_fusion_task_predicts_target_title_from_sid_history_without_value(self) -> None:
        prompt, target = build_fusion_prompt(
            history_text="<a_0><b_0><c_0>, <a_1><b_1><c_1>",
            target_item={"title": "Cabin Air Filter", "value_token": "[VAL_2]"},
        )

        self.assertIn("<a_0><b_0><c_0>", prompt)
        self.assertEqual(target, "Cabin Air Filter\n")
        self.assertNotIn("[VAL_", prompt)


if __name__ == "__main__":
    unittest.main()
