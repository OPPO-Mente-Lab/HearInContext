# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic paired-bootstrap contracts; no benchmark scores or data required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
try:
    import numpy
except ImportError:
    numpy = None
if numpy is not None:
    from bootstrap import paired_bootstrap


def fixture():
    manifest, base, tuned = {}, {}, {}
    # First parent has two branches/voices and short references; second has one
    # longer reference. Pooled deltas differ from unweighted group averages.
    for key, parent, reference in (("a", "p1", "pear"), ("b", "p1", "pear"),
                                   ("c", "p2", "we need a pear today")):
        manifest[key] = dict(example_id=key, parent_id=parent, case_id=key,
                             language="en", condition="C1", reference_text=reference,
                             branch_word="pear")
        base[key] = dict(example_id=key, hypothesis="" if parent == "p1" else reference)
        tuned[key] = dict(example_id=key, hypothesis=reference if parent == "p1" else "")
    return manifest, base, tuned


@unittest.skipIf(numpy is None, "Install requirements.txt for bootstrap tests")
class BootstrapTests(unittest.TestCase):
    def test_pooled_estimates_and_intact_clusters(self):
        result = paired_bootstrap(*fixture(), "en", "C1", replicates=4096)
        self.assertEqual(result["selected_rows"], 3)
        self.assertEqual(result["parent_groups"], 2)
        self.assertAlmostEqual(result["metrics"]["WER"]["delta_pp"], 300 / 7)
        self.assertAlmostEqual(result["metrics"]["target_recall"]["delta_pp"], 100 / 3)
        # Repeated sampling of either intact parent reaches both +/-100 ends.
        self.assertEqual(result["metrics"]["WER"]["ci95_pp"], [-100.0, 100.0])
        self.assertEqual(result["metrics"]["target_recall"]["ci95_pp"], [-100.0, 100.0])

    def test_identical_predictions_have_zero_interval(self):
        manifest, base, _ = fixture()
        result = paired_bootstrap(manifest, base, base, "en", "C1", replicates=300)
        for metric in result["metrics"].values():
            self.assertEqual(metric["delta_pp"], 0)
            self.assertEqual(metric["ci95_pp"], [0.0, 0.0])

    def test_seed_reproducible_across_multiple_batches(self):
        first = paired_bootstrap(*fixture(), "en", "C1", replicates=777, seed=91)
        self.assertEqual(first, paired_bootstrap(*fixture(), "en", "C1", replicates=777, seed=91))

    def test_missing_id_parent_and_single_parent_rejected(self):
        for problem in ("id", "parent", "single"):
            manifest, base, tuned = fixture()
            if problem == "id":
                base.pop("c")
            elif problem == "parent":
                manifest["c"].pop("parent_id")
            else:
                manifest["c"]["parent_id"] = "p1"
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                paired_bootstrap(manifest, base, tuned, "en", "C1", replicates=10)

    def test_wrong_metadata_even_outside_selected_slice_rejected(self):
        for problem in ("prediction", "parent_language", "case_identity", "outside_id"):
            manifest, base, tuned = fixture()
            manifest["d"] = dict(manifest["c"], example_id="d", case_id="d", condition="C0")
            base["d"] = dict(example_id="d", hypothesis="pear")
            tuned["d"] = dict(base["d"])
            if problem == "prediction":
                tuned["d"]["audio"] = "wrong.wav"
                manifest["d"]["audio"] = "right.wav"
            elif problem == "parent_language":
                manifest["d"]["language"] = "zh"
            elif problem == "case_identity":
                manifest["d"]["case_id"] = "a"
            else:
                tuned.pop("d")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                paired_bootstrap(manifest, base, tuned, "en", "C1", replicates=10)

    def test_invalid_parameters_rejected(self):
        for options in (dict(replicates=0), dict(seed=-1)):
            with self.assertRaises(ValueError):
                paired_bootstrap(*fixture(), "en", "C1", **options)

    def test_chinese_c3_uses_fixed_normalizer(self):
        manifest, base, tuned = fixture()
        for key in manifest:
            manifest[key].update(language="zh", condition="C3", reference_text="17个梨",
                                 branch_word="梨", explicit_cue_depth=2)
            base[key]["hypothesis"] = "十七个梨"
            tuned[key]["hypothesis"] = "17个梨"
        result = paired_bootstrap(manifest, base, tuned, "zh", "C3", replicates=20)
        self.assertEqual(result["metrics"]["CER"]["ci95_pp"], [0.0, 0.0])

    def test_cli_other_cwd_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in zip(("manifest", "base", "tuned"), fixture()):
                (root / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in values.values()))
            originals = {path: path.read_bytes() for path in root.glob("*.jsonl")}
            output = root / "outputs" / "result.json"
            command = [sys.executable, str(EVALUATION / "bootstrap.py"),
                       "--manifest", str(root / "manifest.jsonl"), "--base", str(root / "base.jsonl"),
                       "--tuned", str(root / "tuned.jsonl"), "--language", "en", "--condition", "C1",
                       "--replicates", "257", "--output", str(output)]
            first = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            content = output.read_bytes()
            result = json.loads(content)
            self.assertEqual(set(result["input_sha256"]), {"manifest", "base", "tuned"})
            self.assertNotIn("p_value", result)
            second = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(content, output.read_bytes())
            command[-1] = str(root / "manifest.jsonl")
            third = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(third.returncode, 0)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
