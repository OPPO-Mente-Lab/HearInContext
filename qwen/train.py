#!/usr/bin/env python3
# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen full-parameter SFT configuration on portable release data."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import VENDOR, digest, read_rows, resolve_audio, writer_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    datasets = {}
    for name, manifest, count in (("train", args.train_manifest, 14400), ("dev", args.dev_manifest, 4800)):
        rows = read_rows(manifest)
        if len(rows) != count:
            raise ValueError(f"{name}: expected {count} rows, found {len(rows)}")
        datasets[name] = [dict(audio=resolve_audio(row, args.data_root),
                               text=row["reference_text"] if name == "dev" else row["text"],
                               prompt=row["context"] if name == "dev" else row.get("prompt", ""))
                          for row in rows]
    config = dict(train_sha256=digest(args.train_manifest), dev_sha256=digest(args.dev_manifest),
        base_model=str(args.base_model.resolve()), learning_rate=1e-5, epochs=1,
        batch_size=1, gradient_accumulation=16, optimizer_steps=900, save_steps=100,
        warmup_ratio=0.02, scheduler="linear", load_best_model_at_end=False)
    if args.validate_only:
        print(json.dumps(dict(status="PASS", **config)))
        return
    with writer_lock(args.output_dir):
        meta = args.output_dir / "training_config.json"
        if meta.exists():
            if not args.resume or json.loads(meta.read_text()) != config:
                raise ValueError("Existing training output requires --resume and identical configuration")
        meta.write_text(json.dumps(config, indent=2) + "\n")
        for name, rows in datasets.items():
            with (args.output_dir / f"{name}.resolved.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        command = [sys.executable, str(VENDOR / "finetuning/qwen3_asr_sft.py"),
            "--model_path", str(args.base_model.resolve()),
            "--train_file", str((args.output_dir / "train.resolved.jsonl").resolve()),
            "--eval_file", str((args.output_dir / "dev.resolved.jsonl").resolve()),
            "--output_dir", str((args.output_dir / "checkpoints").resolve()),
            "--batch_size", "1", "--grad_acc", "16", "--lr", "1e-5", "--epochs", "1",
            "--log_steps", "10", "--save_strategy", "steps", "--save_steps", "100",
            "--save_total_limit", "10", "--load_best_model_at_end", "0", "--num_workers", "4"]
        if args.resume:
            command += ["--resume", "1"]
        env = {**os.environ, "PYTHONPATH": str(VENDOR), "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
