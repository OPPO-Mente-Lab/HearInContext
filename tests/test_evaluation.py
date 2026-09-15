# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable frozen-rule and CLI contracts; all examples are synthetic."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
from metrics import read_jsonl, score_benchmark
from normalization import normalize_pair, normalize_chinese_text


class EvaluationTests(unittest.TestCase):
    def test_local_chinese_script_boundaries(self):
        for raw, expected in (("中文ABC测试", "中 文 abc 测 试"),
                              ("甲，乙！A B", "甲 乙 ab"),
                              ("あカ한漢", "あ カ 한 漢"),
                              ("中🙂文", "中🙂文"),
                              ("A B Cs", "abcs"),
                              ("中9文", "中9文"),
                              ("甲」乙", "甲」乙")):
            self.assertEqual(normalize_chinese_text(raw), expected)

    def test_initialism_equivalence(self):
        for spaced, compact in (("F D I C", "FDIC"), ("F I C O", "FICO"),
                                ("I V", "IV"), ("N S A I Ds", "NSAIDs"),
                                ("x G", "xG"), ("e G F R", "eGFR"),
                                ("J. P.", "JP")):
            with self.subTest(spaced=spaced):
                pair = normalize_pair(spaced, compact, "en", [spaced])
                self.assertEqual(pair["reference_tokens"], pair["hypothesis_tokens"])
                self.assertEqual(pair["target_tokens"], [pair["reference_tokens"]])

    def test_initialisms_preserve_words_and_boundaries(self):
        for raw, expected in (("I A student", "i a student"),
                              ("a C B C panel", "a cbc panel"),
                              ("M R I. A medical student", "mri a medical student")):
            self.assertEqual(normalize_pair(raw, "", "en")["reference_tokens"],
                             expected.split())

    def test_chinese_numbers_and_ordinary_words(self):
        for raw, spoken in (("17", "十七"), ("4%", "百分之四"),
                            ("03", "零三"), ("合十动作", "合十动作")):
            pair = normalize_pair(raw, spoken, "zh", [raw])
            self.assertEqual(pair["reference_tokens"], list(spoken))
            self.assertEqual(pair["hypothesis_tokens"], list(spoken))
            self.assertEqual(pair["target_tokens"], [list(spoken)])

    def test_reference_normalization_is_independent(self):
        first = normalize_pair("F D I C twenty dollars", "wrong", "en", ["F D I C"])
        second = normalize_pair("F D I C twenty dollars", "$20", "en", ["F D I C"])
        self.assertEqual(first["reference_tokens"], ["fdic", "$20"])
        self.assertEqual(first["reference_tokens"], second["reference_tokens"])
        self.assertEqual(first["target_tokens"], second["target_tokens"])

    def test_known_spelling_collision_is_preserved(self):
        self.assertEqual(normalize_pair("bussing", "busing", "en", ["bussing", "busing"])
                         ["target_tokens"], [["busing"], ["busing"]])

    def manifest(self, **changes):
        return {"e": {"example_id": "e", "language": "en", "condition": "C1",
                      "reference_text": "a bear", "branch_word": "bear", **changes}}

    def test_word_error_and_target_recall(self):
        scores, rows = score_benchmark(self.manifest(), {"e": {"hypothesis": "a bare"}})
        self.assertEqual(scores["groups"]["en_C1"]["error_rate"], 0.5)
        self.assertEqual(scores["groups"]["en_C1"]["target_recall"], 0)
        self.assertEqual(rows[0]["errors"], 1)
        _, rows = score_benchmark(self.manifest(reference_text="oil", branch_word="oil"),
                                  {"e": {"hypothesis": "spoiled"}})
        self.assertEqual(rows[0]["target_hit"], 0)

    def test_chinese_cer_and_c3_depth(self):
        for depth in (0, 2, 4):
            scores, rows = score_benchmark(self.manifest(language="zh", condition="C3",
                reference_text="案件", branch_word="案件", explicit_cue_depth=depth),
                {"e": {"hypothesis": "按键"}})
            self.assertEqual(scores["c3_depth"][f"zh_C3_D{depth}"]["error_rate"], 1)
            self.assertEqual(rows[0]["target_hit"], 0)

    def test_empty_hypothesis_counts_as_deletions(self):
        scores, rows = score_benchmark(self.manifest(), {"e": {"hypothesis": ""}})
        self.assertEqual(rows[0]["errors"], 2)
        self.assertEqual(scores["groups"]["en_C1"]["empty_rate"], 1)

    def test_c3_overall_pools_counts_not_depth_percentages(self):
        examples = [(0, "red pear", "red pair"),
                    (0, "green pear", "green pair"),
                    (2, "we need a ripe pear today", "we need a ripe pear today"),
                    (4, "we would like a ripe pear from the market today",
                        "we would like a ripe pear from the market today")]
        manifest, predictions = {}, {}
        for index, (depth, reference, hypothesis) in enumerate(examples):
            key = str(index)
            manifest[key] = dict(example_id=key, language="en", condition="C3",
                                 reference_text=reference, branch_word="pear",
                                 explicit_cue_depth=depth)
            predictions[key] = dict(example_id=key, hypothesis=hypothesis)
        scores, _ = score_benchmark(manifest, predictions)
        self.assertEqual(scores["groups"]["en_C3"]["target_recall"], 0.5)
        self.assertEqual(scores["groups"]["en_C3"]["reference_units"], 20)
        self.assertAlmostEqual(scores["groups"]["en_C3"]["error_rate"], 2 / 20)
        macro = sum(v["target_recall"] for v in scores["c3_depth"].values()) / 3
        self.assertNotEqual(scores["groups"]["en_C3"]["target_recall"], macro)

    def test_invalid_inputs_rejected(self):
        for predictions in ({}, {"x": {"hypothesis": "a bear"}},
                            {"e": {"hypothesis": "a bear", "reference_text": "changed"}}):
            with self.assertRaises(ValueError):
                score_benchmark(self.manifest(), predictions)
        for metadata in ({"condition": "unknown"}, {"condition": "C3"},
                         {"condition": "C3", "explicit_cue_depth": 1}):
            with self.assertRaises(ValueError):
                score_benchmark(self.manifest(**metadata), {"e": {"hypothesis": "a bear"}})

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.jsonl"
            path.write_text('{"example_id":"e"}\n' * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl(path)

    def test_cli_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, hypotheses = root / "manifest.jsonl", root / "hypotheses.jsonl"
            output, observations = root / "score.json", root / "observations.jsonl"
            manifest.write_text(json.dumps(self.manifest()["e"]) + "\n", encoding="utf-8")
            hypotheses.write_text('{"example_id":"e","hypothesis":"a bear"}\n', encoding="utf-8")
            command = [sys.executable, str(EVALUATION / "evaluate.py"), "--manifest", str(manifest),
                       "--hypotheses", str(hypotheses), "--output", str(output),
                       "--observations", str(observations)]
            run = subprocess.run(command, capture_output=True, text=True, cwd=root)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(output.read_text())["groups"]["en_C1"]["target_recall"], 1)
            self.assertEqual(json.loads(observations.read_text())["target_hit"], 1)
            before = output.read_bytes(), observations.read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True, cwd=root)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual((output.read_bytes(), observations.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
