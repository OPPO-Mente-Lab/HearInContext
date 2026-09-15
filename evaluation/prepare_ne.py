# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an official ID/transcript list and shared hotword list to a manifest."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-text', type=Path, required=True)
    parser.add_argument('--hotwords', type=Path, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.reference_text.resolve(), args.hotwords.resolve()}:
        parser.error('Output must differ from inputs')
    words = args.hotwords.read_text(encoding='utf-8-sig').splitlines()
    words = [w.strip() for w in words if w.strip()]
    if not words or len(set(words)) != len(words):
        raise ValueError('Expected a nonempty, unique hotword list, one phrase per line')
    rows = {}
    for line in args.reference_text.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] in rows:
            raise ValueError('Expected unique utterance_id followed by reference text')
        uid, text = parts
        rows[uid] = dict(example_id=uid, reference_text=text, language='zh',
                         condition=args.condition, entity_list=words)
    if not rows or not args.condition.strip():
        raise ValueError('Expected nonempty references and condition')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(dict(rows=len(rows), hotwords=len(words))))


if __name__ == '__main__':
    main()
