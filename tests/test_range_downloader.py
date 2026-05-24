import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from range_download_amazon_2023 import (
    build_file_specs,
    parse_content_range_total,
    resume_start,
)


class RangeDownloaderTest(unittest.TestCase):
    def test_parse_content_range_total(self) -> None:
        self.assertEqual(parse_content_range_total("bytes 0-9/100"), 100)
        self.assertIsNone(parse_content_range_total("bytes 0-9/*"))
        self.assertIsNone(parse_content_range_total(""))

    def test_resume_start_uses_existing_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.bin"
            path.write_bytes(b"abc")
            self.assertEqual(resume_start(path), 3)
            self.assertEqual(resume_start(Path(tmpdir) / "missing.bin"), 0)

    def test_build_file_specs_uses_mirror_paths_and_project_names(self) -> None:
        specs = build_file_specs(
            category="Office_Products",
            endpoint="https://hf-mirror.com",
            output_dir=Path("data/raw"),
        )
        self.assertEqual(specs[0].output_path, Path("data/raw/Office_Products.reviews.jsonl"))
        self.assertEqual(specs[1].output_path, Path("data/raw/meta_Office_Products.jsonl"))
        self.assertIn("raw/review_categories/Office_Products.jsonl", specs[0].url)
        self.assertIn("raw/meta_categories/meta_Office_Products.jsonl", specs[1].url)


if __name__ == "__main__":
    unittest.main()
