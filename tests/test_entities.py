# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EVAL = Path(__file__).resolve().parents[1] / 'evaluation'
sys.path.insert(0, str(EVAL))
from entities import entity_counts, score_entities
from metrics import score_benchmark


class EntityTests(unittest.TestCase):
    def test_multiple_repeated_overlapping_targets(self):
        self.assertEqual(entity_counts(list('甲乙甲'), list('甲甲甲丙'),
            [list('甲'), list('甲乙'), list('丙')]),
            dict(entity_targets=3, entity_hits=2, extra_entity_occurrences=2))

    def test_duplicate_labels_preserved(self):
        self.assertEqual(entity_counts(['a'], ['a', 'a'], [['a'], ['a']]),
            dict(entity_targets=2, entity_hits=2, extra_entity_occurrences=2))

    def test_no_reference_mentions(self):
        self.assertEqual(entity_counts([], ['a'], [[], ['a']]),
            dict(entity_targets=0, entity_hits=0, extra_entity_occurrences=1))

    def data(self):
        return {'a': dict(example_id='a', language='zh', condition='hotwords',
                          reference_text='甲乙甲', entity_list=['甲', '乙'])}

    def test_pooled_counts_and_empty_hypothesis(self):
        m = self.data()
        m['b'] = dict(example_id='b', language='zh', condition='hotwords',
                      reference_text='甲', entity_list=['甲'])
        result, rows = score_entities(m, {'a': {'hypothesis': '甲甲'}, 'b': {'hypothesis': ''}})
        g = result['groups']['zh_hotwords']
        self.assertEqual((g['entity_hits'], g['entity_targets']), (2, 4))
        self.assertEqual(g['entity_recall'], .5)
        self.assertEqual(g['error_rate'], .5)

    def test_shared_normalization(self):
        m = {'a': dict(example_id='a', language='en', condition='coarse',
                       reference_text='F D I C twenty dollars', entity_list=['FDIC', '$20'])}
        result, _ = score_entities(m, {'a': {'hypothesis': 'FDIC $20'}})
        self.assertEqual(result['groups']['en_coarse']['entity_recall'], 1)
        self.assertEqual(result['groups']['en_coarse']['error_rate'], 0)

    def test_zero_denominators(self):
        m = self.data(); m['a'].update(reference_text='', entity_list=[])
        result, _ = score_entities(m, {'a': {'hypothesis': '甲'}})
        g = result['groups']['zh_hotwords']
        self.assertEqual(g['errors'], 1)
        self.assertIsNone(g['error_rate']); self.assertIsNone(g['entity_recall'])

    def test_invalid_metadata(self):
        for hyp in ({}, {'a': {'hypothesis': 1}},
                    {'a': {'hypothesis': '', 'entity_list': ['wrong']}}):
            with self.assertRaises(ValueError): score_entities(self.data(), hyp)
        m = self.data();m['a']['entity_list']=[2]
        with self.assertRaises(ValueError): score_entities(m, {'a': {'hypothesis': ''}})

    def test_benchmark_multiple_candidates_still_presence(self):
        m = {'a': dict(example_id='a', language='en', condition='C1',
                       reference_text='morning', branch_word='morning')}
        result, _ = score_benchmark(m, {'a': {'hypothesis': 'morning mourning'}})
        self.assertEqual(result['groups']['en_C1']['target_recall'], 1)
        self.assertEqual(result['groups']['en_C1']['errors'], 1)

    def test_prepare_and_cli_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory);ref=p/'ref';words=p/'words';manifest=p/'manifest';hyp=p/'hyp';out=p/'out'
            ref.write_text('a 甲乙甲\n', encoding='utf-8');words.write_text('甲\n乙\n', encoding='utf-8')
            prep=[sys.executable, str(EVAL/'prepare_ne.py'), '--reference-text', str(ref),
                  '--hotwords', str(words), '--condition', 'hotwords', '--output', str(manifest)]
            subprocess.run(prep, cwd=p, capture_output=True, check=True)
            hyp.write_text(json.dumps(dict(example_id='a', hypothesis='甲甲'))+'\n')
            cmd=[sys.executable, str(EVAL/'evaluate.py'), '--mode', 'entities', '--manifest', str(manifest),
                 '--hypotheses', str(hyp), '--output', str(out)]
            subprocess.run(cmd,cwd=p,capture_output=True,check=True)
            saved=out.read_bytes();self.assertAlmostEqual(json.loads(saved)['groups']['zh_hotwords']['entity_recall'],2/3)
            self.assertNotEqual(subprocess.run(cmd,cwd=p,capture_output=True).returncode,0)
            self.assertEqual(out.read_bytes(),saved)


if __name__ == '__main__': unittest.main()
