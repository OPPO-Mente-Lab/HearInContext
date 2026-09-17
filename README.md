[English](README.md) | [中文](README_zh.md)

# HearInContext

### A Benchmark for Implicit Context in Speech Recognition

[![GitHub](https://img.shields.io/badge/GitHub-HearInContext-181717?logo=github&style=flat)](https://github.com/OPPO-Mente-Lab/HearInContext)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18680-b31b1b?style=flat)](https://arxiv.org/abs/2609.18680)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HearInContext-FFD21E?style=flat)](https://huggingface.co/datasets/OPPOer/HearInContext)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE)

![HearInContext overview](assets/overview.png)

*Illustrative example: the same spoken request is disambiguated as flour or flower by different assistant histories. The dialogue and waveform are illustrative.*

[Evaluation Protocol](docs/evaluation.md) · [Training & Decoding](qwen/README.md)

HearInContext evaluates how speech recognition models use implicit semantic cues from dialogue. Shared-audio comparisons distinguish contextual disambiguation from explicit word hints.

- Mandarin and English, with no-context, implicit, unrelated, and explicit conditions.
- Unified CER/WER and target recall, with breakdowns by condition and speaker.
- Qwen3-ASR fine-tuning and decoding entrypoints, plus paired bootstrap tools.
- CPU-only scoring; GPU-based model training and decoding.

## Installation

Run from the repository root using Python 3.12:

```bash
conda create -n hearincontext python=3.12 -y
conda activate hearincontext
python -m pip install -r requirements.txt
```

For the training and decoding environment, see the [Qwen guide](qwen/README.md).

## Dataset

Training and test data are distributed separately on [Hugging Face](https://huggingface.co/datasets/OPPOer/HearInContext). The dataset card covers data construction, speech synthesis, reference speakers, and data licensing.

## Quick Evaluation

Prepare a test manifest, `test.jsonl`, and model predictions, `predictions.jsonl`. Each prediction must retain the sample's full `example_id`:

```json
{"example_id":"demo::C1::voice1","hypothesis":"Please check the flour."}
```

Run scoring:

```bash
python evaluation/evaluate.py \
  --manifest test.jsonl \
  --hypotheses predictions.jsonl \
  --output outputs/metrics.json
```

IDs must be unique in each file and match exactly across files. Empty transcripts are scored normally. Use a new output path.

Results are grouped by language and condition. `error_rate` and `target_recall` are proportions; multiply by 100 for percentages. Add `--observations outputs/rows.jsonl` to save per-example statistics.

## Training & Decoding

Train from an official Qwen3-ASR base model, then decode with the selected checkpoint:

- [Training configuration and commands](qwen/README.md#full-parameter-training)
- [Decoding commands](qwen/README.md#decode-a-local-checkpoint-or-official-base-model)

We provide code and data, but do not distribute fine-tuned weights.

## Further Usage

- [Input format, normalization, and paired confidence intervals](docs/evaluation.md)
- [Multi-entity scoring and AISHELL-1-NE](docs/entities.md)

Run tests:

```bash
python -m unittest discover -s tests
```

## Citation

```bibtex
@misc{gao2026hearincontext,
  title  = {HearInContext: A Benchmark for Implicit Context in Speech Recognition},
  author = {Gao, Yifan and Tian, Yao and Suo, Hongbin},
  year   = {2026}
}
```

## Acknowledgements

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) for the model, training, and inference implementations.
- [Whisper](https://github.com/openai/whisper) / [Transformers](https://github.com/huggingface/transformers) for English text normalization.

## License

The code is licensed under [Apache-2.0](LICENSE). Third-party code retains its original copyright and license notices; see [NOTICE](NOTICE).
