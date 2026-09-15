[English](entities.md) | [中文](entities_zh.md)

# Shared Multi-Entity Scoring

`evaluation/evaluate.py --mode entities` accepts the unified JSONL format converted from NE, ContextASR, and similar datasets. The default remains `--mode benchmark`, with unchanged main-benchmark outputs. Both modes share `normalization.py` and RapidFuzz edit distance, with no additional dependencies.

## Prepare AISHELL-1-NE

Obtain `text` (the `test/uttid` file in this revision also contains reference transcripts) and `hotword.txt` from the [SeACo repository](https://github.com/R1ckShi/SeACo-Paraformer/tree/855b0fb7cadd57f111e192dcd08682ebf391693b/data/test). Obtain AISHELL audio separately. These third-party datasets are not bundled.

```bash
python evaluation/prepare_ne.py --reference-text text --hotwords hotword.txt \
  --condition hotwords --output ne.jsonl
python evaluation/evaluate.py --mode entities --manifest ne.jsonl \
  --hypotheses predictions.jsonl --output scores.json --observations rows.jsonl
```

The official version contains 808 utterances and 400 hotwords. The preparation tool does not hard-code these counts and preserves the word-list order. Supply the same complete word list for every audio input; do not filter words using reference answers. Scoring does not invoke a model.

Prediction format: `{"example_id":"BAC009S0764W0179","hypothesis":"model transcription"}`. IDs must match the manifest exactly. If older predictions have an `aishell1::` prefix, map it explicitly during data adaptation; the scorer does not infer ID mappings.

## General Input

```json
{"example_id":"demo","language":"zh","condition":"coarse","reference_text":"甲方联系乙方，再联系甲方","entity_list":["甲方","乙方"]}
```

For ContextASR, map `uniq_id` to `example_id` and `text` to `reference_text`, retain `entity_list`, convert languages to `zh/en`, and specify the condition explicitly. When evaluating multiple conditions together, include the condition in each ID. Predictions use the same IDs and the `hypothesis` field.

## Counting Rules

- Apply the same language-specific normalization independently to REF, HYP, and every entity. Match complete, contiguous token sequences.
- For each entity, count its occurrences in the reference (R) and prediction (H). Hits equal min(R,H); aggregate hits are divided by total reference occurrences. For single-target benchmark samples with one reference occurrence, this reduces to per-sample hit counting.
- Count distinct nested labels separately, such as “许玮甯” and “玮甯”. Repeated annotations also retain their multiplicity; no implicit deduplication is applied. Labels that normalize to empty strings do not match.
- `extra_entity_occurrences` is sum(max(H-R,0)). It measures excess lexical occurrences, not aligned false positives or official Precision/F1.
- References without entities contribute zero hits. When the entire group's entity denominator is zero, recall is null. Empty references are retained, and nonempty predictions count as insertions; the error rate is null when total reference units are zero.
- Pool counts within each group rather than averaging sentence percentages. Output `error_rate` and `entity_recall` as proportions from 0 to 1.

Counting regression checks have been completed for the four NE configurations under both conditions. References without entities contribute zero hits, and recall is null when its denominator is zero.

This entrypoint uses our unified protocol, not SeACo's official scoring or alignment-based hotword F1 from external benchmarks. Sample-level speaker fields may be retained for further auditing. The existing bootstrap entrypoint supports main-benchmark `parent_id` groups, not NE speaker clustering.
