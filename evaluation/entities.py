# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-entity counts sharing the benchmark normalization and edit distance."""
from collections import Counter, defaultdict
from metrics import edit_distance
from normalization import POLICY, normalize_pair


def entity_counts(reference, hypothesis, targets):
    def occurrences(tokens, target):
        width = len(target)
        return sum(tuple(tokens[i:i + width]) == target
                   for i in range(len(tokens) - width + 1))
    total = hits = extra = 0
    for target, multiplicity in Counter(tuple(t) for t in targets if t).items():
        r = occurrences(reference, target) * multiplicity
        h = occurrences(hypothesis, target) * multiplicity
        total += r
        hits += min(r, h)
        extra += max(0, h - r)
    return dict(entity_targets=total, entity_hits=hits, extra_entity_occurrences=extra)


def score_entities(manifest, predictions):
    if not manifest or manifest.keys() != predictions.keys():
        raise ValueError('Expected nonempty, identical manifest/hypothesis ID sets')
    rows = []
    for uid, row in manifest.items():
        hyp = predictions[uid]
        for field in ('reference_text', 'language', 'condition', 'entity_list', 'speaker'):
            if field in hyp and hyp[field] != row.get(field):
                raise ValueError(f'{uid}: conflicting prediction metadata {field}')
        if (row.get('language') not in ('zh', 'en')
                or not isinstance(row.get('condition'), str) or not row['condition']
                or not isinstance(row.get('reference_text'), str)
                or not isinstance(hyp.get('hypothesis'), str)
                or not isinstance(row.get('entity_list'), list)
                or any(not isinstance(t, str) for t in row['entity_list'])):
            raise ValueError(f'{uid}: invalid entity input')
        pair = normalize_pair(row['reference_text'], hyp['hypothesis'],
                              row['language'], row['entity_list'])
        ref, pred = pair['reference_tokens'], pair['hypothesis_tokens']
        rows.append(dict(example_id=uid, language=row['language'], condition=row['condition'],
            errors=edit_distance(ref, pred), reference_units=len(ref),
            speaker=row.get('speaker'), empty=int(not hyp['hypothesis'].strip()),
            **entity_counts(ref, pred, pair['target_tokens'])))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['language'] + '_' + row['condition']].append(row)
    groups = {}
    for key, group in sorted(grouped.items()):
        counts = {field: sum(r[field] for r in group) for field in
                  ('errors', 'reference_units', 'entity_targets', 'entity_hits',
                   'extra_entity_occurrences', 'empty')}
        groups[key] = dict(rows=len(group), **counts,
            error_rate=counts['errors'] / counts['reference_units'] if counts['reference_units'] else None,
            entity_recall=counts['entity_hits'] / counts['entity_targets'] if counts['entity_targets'] else None)
    return dict(status='PASS', rows=len(rows), metric_contract=POLICY +
                '; pooled entity occurrences capped by reference counts; overlaps retained', groups=groups), rows
