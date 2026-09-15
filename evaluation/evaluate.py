#!/usr/bin/env python3
# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate HearInContext predictions with the fixed bilingual protocol."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from metrics import read_jsonl, score_benchmark, sha256
from entities import score_entities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--hypotheses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations', type=Path, help='Optional per-example counts')
    parser.add_argument('--mode', choices=('benchmark', 'entities'), default='benchmark')
    args = parser.parse_args()
    inputs = {args.manifest.resolve(), args.hypotheses.resolve()}
    outputs = [args.output] + ([args.observations] if args.observations else [])
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != len(resolved) or inputs.intersection(resolved):
        parser.error('Outputs must differ from inputs and each other')
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve outputs exclusively: two jobs cannot write the same destination.
    # Failed runs retain non-PASS files; a retry uses a new destination.
    with ExitStack() as stack:
        output = stack.enter_context(args.output.open('x', encoding='utf-8'))
        obs_file = (stack.enter_context(args.observations.open('x', encoding='utf-8'))
                    if args.observations else None)
        scorer = score_benchmark if args.mode == 'benchmark' else score_entities
        result, rows = scorer(read_jsonl(args.manifest), read_jsonl(args.hypotheses))
        result['input_sha256'] = {'manifest': sha256(args.manifest),
                                  'hypotheses': sha256(args.hypotheses)}
        if obs_file:
            for row in rows:
                obs_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({'status': 'PASS', 'rows': result['rows'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
