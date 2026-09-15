[English](evaluation.md) | [中文](evaluation_zh.md)

# Evaluation Protocol

Run all commands from the repository root.

## Input Format

Both inputs are UTF-8 JSONL files, with one JSON object per line.
Each file must have unique `example_id` values, and their ID sets must match exactly.

Reference example:

```json
{"example_id":"demo::C1::voice1","language":"en","condition":"C1","reference_text":"Please check the flour.","branch_word":"flour","parent_id":"demo","case_id":"demo_a","speaker":"voice1"}
```

Prediction example:

```json
{"example_id":"demo::C1::voice1","hypothesis":"Please check the flour."}
```

- Languages: `zh` (Chinese) and `en` (English).
- Conditions: `C0`, `C1`, `U`, `C3`, `USER_ONLY`, and `FULL_HISTORY`.
- C3 samples must provide `explicit_cue_depth`, with a value of `0`, `2`, or `4`.
- Empty predictions are valid inputs and are retained for scoring.

## Metrics

- **CER/WER**: total edit distance divided by the total number of normalized reference units. Chinese uses Chinese characters and Latin words; English uses words.
- **Target Recall**: the proportion of samples whose prediction contains the specified target. The normalized target token sequence must occur contiguously.
- **C3 depth groups**: the same metrics reported separately for D0, D2, and D4.

Target Recall measures presence: if a prediction contains both the target and a competitor, the target still counts as a hit. CER/WER captures additional output. This is not mutually exclusive candidate-selection accuracy.

For aggregate C3 results in the paper, use `groups.zh_C3` and `groups.en_C3`, rather than averaging percentages in `c3_depth`. Depth labels belong to different cases; differences between these groups are not a within-sample position ablation.

Use the dataset version corresponding to the predictions. Revised C3 changes the context input, not the scoring rules; matching IDs do not make predictions from different versions interchangeable.

References (REF), predictions (HYP), and target labels independently undergo the same language-specific normalization:

- Chinese: if the original text contains ASCII digits, first convert them to Chinese readings, then apply the local character and Latin-word segmentation.
- English: standardize initialisms, then apply Whisper `EnglishTextNormalizer` imported from Transformers 4.57.6.

The protocol identifier is `hearincontext`.
The output JSON fields `error_rate` and `target_recall` are proportions; multiply by 100 for percentages.

Spelling normalization merges `busing` and `bussing`, so normalized hits for this pair are not evidence of lexical disambiguation. Initialism handling has known word-boundary ambiguities; samples are not removed based on their scores.

## Tests

```bash
python -m unittest discover -s tests
```

## Paired Confidence Intervals

The statistical entrypoint uses the same root requirements as regular scoring:

```bash
python evaluation/bootstrap.py \
  --manifest test.jsonl --base base.jsonl --tuned tuned.jsonl \
  --language zh --condition C1 \
  --replicates 10000 --seed 20260908 \
  --output outputs/paired_ci.json
```

Both prediction files must cover the same manifest. Paired groups are sampled with replacement by `parent_id`, keeping their semantic branches and voices together. Each replicate computes metrics from total edits, reference units, and target hits, rather than averaging group percentages. Outputs report tuned-minus-base differences in percentage points and 95% percentile confidence intervals: negative error-rate differences and positive recall differences indicate improvement.

These intervals measure sample uncertainty for the given checkpoints, not variability across training seeds.

## Real-Speech Hotword Evaluation

The paper uses the AISHELL-1-NE test list of 808 utterances and the 400-word list released by SeACo. NE and other multi-entity inputs, including ContextASR, share `--mode entities`, reusing the main benchmark's normalization and edit distance. Only the annotation unit changes. See [multi-entity scoring](entities.md) for inputs, counting conventions, and NE preparation commands. This does not reproduce SeACo's official R/P/F or directly compare against its published scores.
