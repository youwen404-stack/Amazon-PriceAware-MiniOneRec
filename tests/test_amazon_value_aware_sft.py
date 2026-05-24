import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from amazon_value_aware_sft_data import (
    VALUE_TOKENS,
    build_value_item_metadata_prompt,
    build_value_sequence_prompt,
    encode_prompt_target_with_loss_weights,
    value_sample_weight,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.special = {
            "<a_1>": 101,
            "<b_2>": 102,
            "<c_3>": 103,
            "[VAL_0]": 201,
            "[VAL_1]": 202,
            "[VAL_2]": 203,
            "[VAL_3]": 204,
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        i = 0
        tokens = sorted(self.special, key=len, reverse=True)
        while i < len(text):
            matched = False
            for token in tokens:
                if text.startswith(token, i):
                    ids.append(self.special[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(1000 + ord(text[i]))
                i += 1
        return ids


class AmazonValueAwareSFTTest(unittest.TestCase):
    def test_value_sample_weight_bucket_linear(self) -> None:
        self.assertEqual(
            value_sample_weight("[VAL_0]", scheme="bucket_linear", min_weight=1.0, max_weight=2.0),
            1.0,
        )
        self.assertAlmostEqual(
            value_sample_weight("[VAL_3]", scheme="bucket_linear", min_weight=1.0, max_weight=2.0),
            2.0,
        )

    def test_sequence_target_appends_value_token_after_sid(self) -> None:
        prompt, target_parts = build_value_sequence_prompt(
            "history sid",
            "<a_1><b_2><c_3>",
            "[VAL_2]",
        )

        self.assertIn("commercial value bucket", prompt)
        self.assertEqual(target_parts, [("<a_1><b_2><c_3>", 1.0), ("[VAL_2]", 1.0), ("\n", 1.0)])

    def test_encode_weights_sid_and_value_tokens_like_vsl(self) -> None:
        tokenizer = FakeTokenizer()
        encoded = encode_prompt_target_with_loss_weights(
            tokenizer,
            "Predict SID and value.",
            "history",
            [("<a_1><b_2><c_3>", 1.0), ("[VAL_3]", 1.5), ("\n", 1.0)],
            cutoff_len=512,
            sample_weight=2.0,
            value_lambda=1.5,
        )

        labels = encoded["labels"]
        weights = encoded["loss_weights"]
        sid_positions = [labels.index(101), labels.index(102), labels.index(103)]
        value_position = labels.index(204)

        for pos in range(sid_positions[0]):
            self.assertEqual(labels[pos], -100)
            self.assertEqual(weights[pos], 0.0)
        for pos in sid_positions:
            self.assertEqual(weights[pos], 2.0)
        self.assertEqual(weights[value_position], 3.0)
        self.assertEqual(labels[-1], tokenizer.eos_token_id)
        self.assertEqual(weights[-1], 2.0)

    def test_title2sid_metadata_predicts_sid_and_value_but_sid2title_stays_baseline(self) -> None:
        item = {"title": "Brake Rotor", "description": "Front rotor.", "value_token": "[VAL_3]"}

        title_prompt, title_target_parts = build_value_item_metadata_prompt(
            "title2sid",
            "<a_1><b_2><c_3>",
            item,
        )
        sid_prompt, sid_target_parts = build_value_item_metadata_prompt(
            "sid2title",
            "<a_1><b_2><c_3>",
            item,
        )

        self.assertIn("commercial value bucket", title_prompt)
        self.assertEqual(title_target_parts, [("<a_1><b_2><c_3>", 1.0), ("[VAL_3]", 1.0), ("\n", 1.0)])
        self.assertIn("<a_1><b_2><c_3>", sid_prompt)
        self.assertEqual(sid_target_parts, [("Brake Rotor\n", 1.0)])

    def test_baseline_data_file_remains_value_free(self) -> None:
        source = (ROOT / "scripts" / "amazon_sft_data.py").read_text(encoding="utf-8")

        for token in VALUE_TOKENS:
            self.assertNotIn(token, source)
        self.assertNotIn("loss_weights", source)


if __name__ == "__main__":
    unittest.main()
