#!/usr/bin/env python3
# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode release manifests with the paper's Qwen3-ASR vLLM protocol."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from common import VENDOR, digest, read_rows, resolve_audio, writer_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    rows = read_rows(args.manifest)
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example_id in manifest")
    requests = {}
    keys = []
    for row in rows:
        if row["language"] not in ("zh", "en"):
            raise ValueError("Only zh and en are supported")
        context = row.get("context", row.get("prompt", ""))
        if not isinstance(context, str):
            raise ValueError("Context must be a string")
        # The portable key uses relative audio, so moving the release is harmless.
        value = [row["audio"], context, row["language"]]
        key = hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        keys.append(key)
        requests[key] = dict(audio=resolve_audio(row, args.data_root), context=context, language=row["language"])
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    config = dict(manifest_sha256=digest(args.manifest), model=str(args.model.resolve()),
                  batch_size=args.batch_size, temperature=0.0, max_new_tokens=512,
                  max_model_len=8192, tensor_parallel_size=1,
                  gpu_memory_utilization=0.85, enforce_eager=False, dtype="bfloat16",
                  forced_language=True, rows=len(rows), unique_requests=len(requests))
    if args.validate_only:
        print(json.dumps(dict(status="PASS", **config)))
        return
    with writer_lock(args.output_dir):
        meta = args.output_dir / "decode_config.json"
        if meta.exists() and json.loads(meta.read_text()) != config:
            raise ValueError("Output belongs to a different model, manifest or decode configuration")
        meta.write_text(json.dumps(config, indent=2) + "\n")
        raw = args.output_dir / "unique_hypotheses.jsonl"
        if raw.exists() and raw.stat().st_size and not raw.read_bytes().endswith(b"\n"):
            raise ValueError("Resume file ends without a newline; preserve and inspect the interrupted final record")
        prior = read_rows(raw) if raw.exists() and raw.stat().st_size else []
        completed = {row["request_key"]: row for row in prior}
        if len(completed) != len(prior) or set(completed) - set(requests):
            raise ValueError("Duplicate or foreign request in resume file")
        pending = [key for key in requests if key not in completed]
        if pending:
            sys.path.insert(0, str(VENDOR))
            from qwen_asr import Qwen3ASRModel
            model = Qwen3ASRModel.LLM(model=str(args.model), dtype="bfloat16",
                gpu_memory_utilization=0.85, tensor_parallel_size=1,
                max_inference_batch_size=args.batch_size, max_model_len=8192,
                enforce_eager=False, max_new_tokens=512)
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                values = model.transcribe(audio=[requests[k]["audio"] for k in batch],
                    context=[requests[k]["context"] for k in batch],
                    language=["Chinese" if requests[k]["language"] == "zh" else "English" for k in batch],
                    return_time_stamps=False)
                if len(values) != len(batch):
                    raise RuntimeError("Model response count mismatch")
                with raw.open("a", encoding="utf-8") as stream:
                    for key, value in zip(batch, values):
                        text = str(getattr(value, "text", "") or "").strip()
                        result = dict(request_key=key, hypothesis=text,
                            empty_output=not bool(text), detected_language=str(value.language or ""))
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        completed[key] = result
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(dict(done=len(completed), total=len(requests))), flush=True)
        with (args.output_dir / "hypotheses.jsonl").open("w", encoding="utf-8") as stream:
            for row, key in zip(rows, keys):
                stream.write(json.dumps({**row, **completed[key]}, ensure_ascii=False) + "\n")
        print(json.dumps(dict(status="PASS", rows=len(rows))))


if __name__ == "__main__":
    main()
