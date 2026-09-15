# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the bundled release's dependency boundary."""
import ast
from pathlib import Path
import tomllib
import typing
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "qwen/vendor/Qwen3-ASR"


class QwenReleaseTests(unittest.TestCase):
    def test_dependency_removed(self):
        project = tomllib.loads((VENDOR / "pyproject.toml").read_text())
        dependencies = project["project"]["dependencies"]
        self.assertFalse(any("soynlp" in d.lower() for d in dependencies))
        self.assertNotIn("soynlp", (ROOT / "qwen/requirements-observed.txt").read_text())
        for path in VENDOR.rglob("*.py"):
            self.assertNotIn("soynlp", path.read_text(), str(path))

    def test_alignment_processor_without_korean_package(self):
        # Exercise the actual processor class without importing model/GPU libraries.
        path = VENDOR / "qwen_asr/inference/qwen3_forced_aligner.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "Qwen3ForceAlignProcessor")
        namespace = {"List": typing.List, "unicodedata": unicodedata}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        processor = namespace[cls.name]()
        self.assertEqual(processor.encode_timestamp("你好世界", "Chinese")[0], list("你好世界"))
        self.assertEqual(processor.encode_timestamp("Hello world", "English")[0], ["Hello", "world"])
        with self.assertRaisesRegex(ValueError, "Korean forced alignment"):
            processor.encode_timestamp("안녕하세요", "Korean")


if __name__ == "__main__":
    unittest.main()
