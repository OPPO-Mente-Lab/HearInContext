# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HearInContext CER/WER, target recall and grouped statistics."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from normalization import POLICY, normalize_pair, contains_tokens

def read_jsonl(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("example_id")
            if not isinstance(key, str) or not key or key in rows:
                raise ValueError(f"{path}:{line_number}: missing/invalid/duplicate example_id")
            rows[key] = row
    if not rows:
        raise ValueError(f"{path}: empty input")
    return rows


def edit_distance(left, right):
    """Exact unit-cost token Levenshtein distance (RapidFuzz CPU implementation)."""
    return Levenshtein.distance(left, right)


def observations(manifest, predictions):
    missing = set(manifest) - set(predictions)
    extra = set(predictions) - set(manifest)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")
    result = []
    for example_id, row in manifest.items():
        hyp = predictions[example_id].get("hypothesis")
        if not isinstance(hyp, str):
            raise ValueError(f"{example_id}: hypothesis must be a string; empty string is valid")
        for field in ("reference_text", "language", "condition", "branch_word",
                      "parent_id", "case_id", "speaker"):
            if field in predictions[example_id] and predictions[example_id][field] != row.get(field):
                raise ValueError(f"{example_id}: conflicting prediction metadata {field}")
        language = row.get("language")
        if language not in ("zh", "en") or row.get("condition") not in ("C0", "C1", "U", "C3", "USER_ONLY", "FULL_HISTORY"):
            raise ValueError(f"{example_id}: unsupported language or condition")
        if not isinstance(row.get("reference_text"), str) or not isinstance(row.get("branch_word"), str) or not row["branch_word"]:
            raise ValueError(f"{example_id}: invalid reference or target")
        depth = row.get("explicit_cue_depth")
        if row["condition"] == "C3" and depth not in (0, 2, 4):
            raise ValueError(f"{example_id}: C3 requires explicit_cue_depth 0, 2, or 4")
        normalized = normalize_pair(row["reference_text"], hyp, language, [row["branch_word"]])
        ref_tokens = normalized["reference_tokens"]
        hyp_tokens = normalized["hypothesis_tokens"]
        if not ref_tokens:
            raise ValueError(f"{example_id}: empty normalized reference")
        result.append({
            "example_id": example_id, "language": language, "condition": row["condition"],
            "depth": depth, "parent_id": row.get("parent_id"),
            "speaker": row.get("speaker"), "gender": row.get("tts_gender", row.get("gender")),
            "errors": edit_distance(ref_tokens, hyp_tokens), "reference_units": len(ref_tokens),
            "target_hit": int(contains_tokens(hyp_tokens, normalized["target_tokens"][0])),
            "empty_normalized_target": int(not normalized["target_tokens"][0]),
            "normalized_target_missing_from_reference": int(not contains_tokens(ref_tokens, normalized["target_tokens"][0])),
            "digit_triggered": int(normalized["digit_triggered"]),
            "empty": int(not hyp.strip()), "exact": int(ref_tokens == hyp_tokens),
            "long": int(len(hyp_tokens) > max(2 * len(ref_tokens), len(ref_tokens) + 10)),
        })
    return result


def aggregate(rows):
    n = len(rows)
    errors = sum(row["errors"] for row in rows)
    units = sum(row["reference_units"] for row in rows)
    hits = sum(row["target_hit"] for row in rows)
    return {"rows": n, "errors": errors, "reference_units": units, "target_hits": hits,
            "error_rate": errors / units, "target_recall": hits / n,
            "empty_rate": sum(row["empty"] for row in rows) / n,
            "exact_rate": sum(row["exact"] for row in rows) / n,
            "long_output_rate": sum(row["long"] for row in rows) / n,
            "digit_triggered_rows": sum(row["digit_triggered"] for row in rows),
            "empty_normalized_target_rows": sum(row["empty_normalized_target"] for row in rows),
            "normalized_target_missing_from_reference_rows": sum(row["normalized_target_missing_from_reference"] for row in rows)}


def score_benchmark(manifest, predictions):
    rows = observations(manifest, predictions)
    groups = defaultdict(list)
    depths = defaultdict(list)
    speakers = defaultdict(list)
    genders = defaultdict(list)
    for row in rows:
        key = f"{row['language']}_{row['condition']}"
        groups[key].append(row)
        if row["condition"] == "C3":
            depths[f"{key}_D{row['depth']}"].append(row)
        if row["speaker"] is not None:
            speakers[f"{key}::{row['speaker']}"].append(row)
        if row["gender"] is not None:
            genders[f"{key}_{row['gender']}"].append(row)
    return {
        "status": "PASS", "rows": len(rows),
        "metric_contract": POLICY + "; labeled-target normalized token-span presence",
        "groups": {key: aggregate(value) for key, value in sorted(groups.items())},
        "c3_depth": {key: aggregate(value) for key, value in sorted(depths.items())},
        "by_speaker": {key: aggregate(value) for key, value in sorted(speakers.items())},
        "by_gender": {key: aggregate(value) for key, value in sorted(genders.items())},
    }, rows


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
