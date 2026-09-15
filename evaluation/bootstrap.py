# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired parent-cluster bootstrap for HearInContext evaluation differences."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from metrics import observations, read_jsonl, sha256
from normalization import POLICY

METADATA_FIELDS = {"example_id", "reference_text", "language", "condition", "branch_word",
                   "parent_id", "case_id", "speaker", "audio", "audio_sha256", "context",
                   "gender", "tts_gender", "explicit_cue_depth"}


def paired_bootstrap(manifest, base, tuned, language, condition,
                     replicates=10000, seed=20260908):
    """Estimate tuned-minus-base differences, resampling intact parent groups."""
    if language not in ("zh", "en") or condition not in ("C0", "C1", "U", "C3"):
        raise ValueError("Unsupported language or condition")
    if not isinstance(replicates, int) or replicates < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError("replicates must be positive and seed must be nonnegative integers")
    if not manifest or set(manifest) != set(base) or set(manifest) != set(tuned):
        raise ValueError("Complete input ID coverage must match for manifest, base, and tuned")
    parent_languages = {}
    case_identity = {}
    for key, row in manifest.items():
        if row.get("example_id") != key:
            raise ValueError(f"{key}: conflicting manifest example_id")
        parent = row.get("parent_id")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError(f"{key}: parent_id must be nonempty")
        if row.get("language") not in ("zh", "en") or row.get("condition") not in (
                "C0", "C1", "U", "C3", "USER_ONLY", "FULL_HISTORY"):
            raise ValueError(f"{key}: unsupported manifest language or condition")
        if parent_languages.setdefault(parent, row["language"]) != row["language"]:
            raise ValueError(f"{parent}: parent spans multiple languages")
        if row["condition"] == "C3" and row.get("explicit_cue_depth") not in (0, 2, 4):
            raise ValueError(f"{key}: invalid C3 depth")
        for field in ("reference_text", "branch_word"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{key}: invalid {field}")
        if "case_id" in row:
            case = row["case_id"]
            if not isinstance(case, str) or not case.strip():
                raise ValueError(f"{key}: invalid case_id")
            identity = (parent, row["language"], row["reference_text"], row["branch_word"])
            if case_identity.setdefault(case, identity) != identity:
                raise ValueError(f"{case}: inconsistent case identity")
        for predictions in (base, tuned):
            prediction = predictions[key]
            if not isinstance(prediction.get("hypothesis"), str):
                raise ValueError(f"{key}: hypothesis must be a string")
            for field in prediction.keys() & (row.keys() | METADATA_FIELDS):
                if field != "hypothesis" and prediction[field] != row.get(field):
                    raise ValueError(f"{key}: conflicting prediction metadata {field}")

    selected = {key: row for key, row in manifest.items()
                if row["language"] == language and row["condition"] == condition}
    parents = sorted({row["parent_id"] for row in selected.values()})
    if len(parents) < 2:
        raise ValueError("Selected slice requires at least two parent groups")
    before = observations(selected, {key: base[key] for key in selected})
    after = observations(selected, {key: tuned[key] for key in selected})
    slots = {parent: index for index, parent in enumerate(parents)}
    # Per-parent columns: reference units, rows, base/tuned errors, base/tuned hits.
    totals = np.zeros((len(parents), 6), dtype=np.int64)
    for left, right in zip(before, after):
        if (left["example_id"], left["reference_units"], left["parent_id"]) != (
                right["example_id"], right["reference_units"], right["parent_id"]):
            raise ValueError("Paired observations have inconsistent identity or reference units")
        totals[slots[left["parent_id"]]] += (
            left["reference_units"], 1, left["errors"], right["errors"],
            left["target_hit"], right["target_hit"])
    pooled = totals.sum(axis=0)
    group_deltas = np.column_stack((totals[:, 0], totals[:, 1],
                                   totals[:, 3] - totals[:, 2], totals[:, 5] - totals[:, 4]))
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 2), dtype=np.float64)
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        multiplicities = rng.multinomial(len(parents), np.full(len(parents), 1 / len(parents)),
                                        size=end - start)
        draws = multiplicities @ group_deltas
        samples[start:end] = 100 * draws[:, 2:] / draws[:, :2]
    intervals = np.percentile(samples, [2.5, 97.5], axis=0, method="linear")
    results = {}
    for column, (name, denominator, base_col, tuned_col) in enumerate((
            ("CER" if language == "zh" else "WER", 0, 2, 3),
            ("target_recall", 1, 4, 5))):
        results[name] = {
            "base_pct": float(100 * pooled[base_col] / pooled[denominator]),
            "tuned_pct": float(100 * pooled[tuned_col] / pooled[denominator]),
            "delta_pp": float(100 * (pooled[tuned_col] - pooled[base_col]) / pooled[denominator]),
            "ci95_pp": intervals[:, column].tolist(),
        }
    return {
        "status": "PASS", "metric_contract": POLICY,
        "language": language, "condition": condition,
        "input_rows": len(manifest), "selected_rows": len(selected), "parent_groups": len(parents),
        "replicates": replicates, "seed": seed, "batch_size": 256,
        "numpy_version": np.__version__,
        "difference": "tuned minus base, percentage points",
        "algorithm": "Paired parent-cluster bootstrap: sample N parents with replacement; retain all selected branches and voices of each parent; compute pooled errors/reference units and hits/rows in each replicate; linear-interpolated 2.5th and 97.5th percentiles",
        "uncertainty": "Evaluation-sample uncertainty conditional on these two trained models; not variation across training seeds",
        "metrics": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "base", "tuned", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--language", choices=("zh", "en"), required=True)
    parser.add_argument("--condition", choices=("C0", "C1", "U", "C3"), required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    sources = {name: getattr(args, name) for name in ("manifest", "base", "tuned")}
    if args.output.exists() or args.output.resolve() in {path.resolve() for path in sources.values()}:
        raise ValueError("Output must be a new file and must not overwrite inputs")
    fingerprints = {name: sha256(path) for name, path in sources.items()}
    result = paired_bootstrap(*(read_jsonl(sources[name]) for name in ("manifest", "base", "tuned")),
                              args.language, args.condition, args.replicates, args.seed)
    if fingerprints != {name: sha256(path) for name, path in sources.items()}:
        raise ValueError("Input files changed during scoring")
    result["input_sha256"] = fingerprints
    directory = Path(__file__).resolve().parent
    result["implementation_sha256"] = {
        name: sha256(directory / name) for name in (
            "bootstrap.py", "metrics.py", "normalization.py",
            "resources/whisper_spelling.json")}
    result["transformers_version"] = version("transformers")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(args.output), "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
