# Qwen3-ASR reproduction entrypoints

Run commands from this Git repository's root. Download the dataset and official
Qwen3-ASR base model separately. This release contains code and data, not
fine-tuned checkpoints. Train from the official base model, then decode with
your locally selected checkpoint.
Manifest `audio` paths are relative to the explicit `--data-root`, independent of
the manifest's own directory. No shared server directory is required.

## Environment

Use Python 3.12 on Linux with a compatible NVIDIA GPU. Install the observed
dependency pins and the bundled Qwen package in an isolated environment:

```bash
python -m pip install -r qwen/requirements-observed.txt
python -m pip install --no-deps -e qwen/vendor/Qwen3-ASR
```

The pins are based on the shared environment, not a complete historical
container lock. This release omits `soynlp`, Korean forced-alignment tokenization,
and its dictionary. Korean forced alignment raises an explicit unsupported error;
Chinese/English ASR training and decoding are unchanged. The remaining pins cover
the dependencies declared by the bundled `pyproject.toml`. These release
entrypoints have CPU manifest checks; no new GPU run was performed while packaging.

## Decode a local checkpoint or official base model

```bash
python qwen/decode.py \
  --manifest /path/to/dataset/test.jsonl \
  --data-root /path/to/dataset \
  --model /path/to/your-training-run/checkpoint-800 \
  --output-dir result/reproduced/qwen1p7b_m80_10_10
```

Baseline decoding uses the corresponding official base model directory.
`--validate-only` checks inputs
without importing the GPU model. `hypotheses.jsonl` retains all manifest rows and
adds `hypothesis`, `request_key`, and output-status fields for evaluation.

The protocol is vLLM, greedy temperature 0, maximum 512 generated tokens,
known Chinese/English language, 8192-token model limit, batch size 32,
tensor parallelism 1, GPU memory utilization 0.85, compiled execution, BF16.
Context is passed directly to Qwen's native `context` argument; no added instruction.
Identical audio/context/language requests are decoded once and expanded to all
example IDs. Request hashes use portable relative paths and therefore differ from
historical hashes containing private absolute paths.

Rerunning the identical command resumes complete saved requests. Empty transcripts
are valid results. Foreign IDs/configurations and malformed saved lines fail
explicitly. Each output directory has one advisory writer lock; keep every active
runner, dependency and checkpoint immutable. GPU/library/batching differences can
change newly generated text even under greedy decoding; the score calculator is deterministic for fixed input transcripts; historical hypotheses are not included in this package.

## Full-parameter training

```bash
python qwen/train.py \
  --base-model /path/to/Qwen3-ASR-1.7B \
  --train-manifest /path/to/dataset/train_mixtures/m80_10_10.jsonl \
  --dev-manifest /path/to/dataset/validation_conditions.jsonl \
  --data-root /path/to/dataset \
  --output-dir result/reproduced/train_qwen1p7b_m80_10_10
```

Use the official unfine-tuned model as `--base-model`. Training writes checkpoints
to your output directory; no experiment weights are distributed by this project.
The wrapper resolves relative audio into private runtime manifests under its output
directory and invokes the exact trainer snapshot. Both model sizes use 14,400
training rows, 50/50 Chinese/English row sampling, 4,800 frozen development rows,
full-parameter SFT, LR 1e-5, batch 1, accumulation 16, one epoch / 900 optimizer
steps, linear scheduler, warmup 0.02, and checkpoints/eval loss every 100 steps.
Mixture order is C1 / C0 / U. No C3 occurs in training. `--resume` uses the last
checkpoint of the same configuration; `--validate-only` performs CPU preflight.

The trainer retains upstream Transformers seed behavior (the wrapper does not
invent a new seed contract). Training reruns are not guaranteed bit-identical.
Select checkpoints with the frozen development protocol, never test scores.
The paper compares 1.7B checkpoints at steps 700, 800, and 400 for
80/20/0, 80/10/10, and 80/0/20, respectively.

For the reported 1.7B mixture comparison, selection uses the frozen development
decodes for checkpoints 100--900, rather than minimum validation loss. Within
each mixture, sum the ascending error-rate ranks and descending target-recall
ranks across Chinese/English × no-context/implicit/unrelated. Add 0.25 times
the summed ascending ranks of the corresponding across-speaker error-rate and
recall standard deviations. Prefer the lowest total, then the lowest performance
rank sum, then the earliest step. The resulting steps for 80/20/0,
80/10/10, and 80/0/20 are 700, 800, and 400. These historical selections are
retained when applying the final scoring normalization to test predictions.

## Source provenance

Vendor: `https://github.com/QwenLM/Qwen3-ASR`, upstream commit
`9567667698f195fa807b1581de5c03184e63d2b0`, package version 0.0.4, Apache-2.0.
The trainer's executable code matches the local experimental snapshot, SHA256
`3d66f86b01261773b3bb8fe4db1140a38d75451a1ece5021781b6fd8727d1953`.
Its local changes relative to upstream independently load train/dev Arrow schemas
and expose `load_best_model_at_end` (disabled for these experiments). The published
file additionally carries OPPO's modification notice; upstream copyright and
license are retained. `VENDOR_SHA256.json` records the current packaged bytes.
The release also removes the unused Korean forced-alignment dependency, tokenizer
path, and dictionary. Those packaging changes are marked in the aligner and
`pyproject.toml`; the ASR model implementation and trainer are unchanged by them.

For independent downloads, pass --data-root /path/to/dataset and explicit --dev-manifest /path/to/dataset/validation_conditions.jsonl. This package has CPU preflight only; no new GPU training or inference is claimed during packaging.
