#!/usr/bin/env python3
"""Score all three checkpoints of one V170 OOF LoRA trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    bind_self_hash,
    environment_fingerprint,
    load_config,
    load_self_hashed_json,
    sha256_file,
    write_json_atomic,
)
from .data_contracts import load_ce_dataset, split_ce_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def prevalence_predictions(
    scores: Sequence[float],
    tie_keys: Sequence[str],
    *,
    prevalence: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ties = [str(value) for value in tie_keys]
    if (
        values.ndim != 1
        or len(values) != len(ties)
        or not np.isfinite(values).all()
        or len(set(ties)) != len(ties)
    ):
        raise RuntimeError("V170 prevalence rank inputs are invalid")
    count = max(0, min(len(values), int(math.floor(len(values) * prevalence + 0.5))))
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), ties[index]))
    prediction = np.zeros(len(values), dtype=np.int8)
    prediction[order[:count]] = 1
    return prediction


def _verify_checkpoint(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path / "checkpoint_manifest.json"
    manifest = load_self_hashed_json(manifest_path, context="V170 checkpoint")
    if (
        manifest.get("schema_version") != "v170.adapter_checkpoint.1"
        or manifest.get("status") != "COMPLETE_UNSCORED"
        or manifest.get("self_sha256") != expected["checkpoint_manifest_self_sha256"]
        or sha256_file(manifest_path) != expected["checkpoint_manifest_sha256"]
        or sha256_file(path / "adapter_config.json") != expected["adapter_config_sha256"]
        or sha256_file(path / "adapter_model.safetensors") != expected["adapter_model_sha256"]
    ):
        raise RuntimeError(f"V170 checkpoint binding drift: {path}")
    return manifest


def _score_one(
    *,
    config: Mapping[str, Any],
    model_path: Path,
    adapter_path: Path,
    query: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    import torch
    from peft import PeftModel

    from .modeling import binary_logits, load_base_model, load_processor, processor_batch

    processor = load_processor(model_path, config)
    base = load_base_model(model_path, config)
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path.resolve()),
        is_trainable=False,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for begin in range(0, len(query), batch_size):
            batch = query.iloc[begin : begin + batch_size]
            features = processor_batch(
                processor,
                batch,
                config=config,
                device=torch.device("cuda"),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = binary_logits(model, features, config)
            scores.extend((logits[:, 1] - logits[:, 0]).float().cpu().tolist())
    del model, base, processor
    gc.collect()
    torch.cuda.empty_cache()
    output = np.asarray(scores, dtype=np.float64)
    if len(output) != len(query) or not np.isfinite(output).all():
        raise RuntimeError("V170 checkpoint scoring lost rows or produced non-finite values")
    return output


def score(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from .modeling import verify_model_snapshot

    if not torch.cuda.is_available():
        raise RuntimeError("V170 checkpoint scoring requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError("V170 checkpoint scoring requires a fresh output directory")
    if args.batch_size <= 0:
        raise ValueError("V170 score batch size must be positive")
    config = load_config(args.config.resolve())
    rows, ce_binding = load_ce_dataset(args.ce_manifest.resolve(), config)
    fit_status_path = args.fit_dir.resolve() / "fit_status.json"
    fit_status = load_self_hashed_json(fit_status_path, context="V170 LoRA fit status")
    if (
        fit_status.get("schema_version") != "v170.lora_fit_status.1"
        or fit_status.get("status") != "FIT_COMPLETE_UNSCORED"
        or fit_status.get("full_fit") is not False
        or fit_status.get("query_fold") not in range(5)
        or fit_status.get("config_sha256") != sha256_file(args.config.resolve())
        or fit_status.get("ce_binding", {}).get("binding_sha256")
        != ce_binding["binding_sha256"]
    ):
        raise RuntimeError("V170 OOF scoring fit-status contract drift")
    train, query = split_ce_rows(
        rows,
        query_fold=int(fit_status["query_fold"]),
        development_folds=config["data"]["supplementary_development_folds"],
        forward_gate_fold=int(config["data"]["forward_gate_fold"]),
    )
    snapshot = verify_model_snapshot(
        args.model_path.resolve(),
        args.base_snapshot_manifest.resolve(),
        config,
    )
    if snapshot["files_sha256"] != fit_status["model_snapshot"]["files_sha256"]:
        raise RuntimeError("V170 scoring base snapshot differs from fit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_results: list[dict[str, Any]] = []
    train_prevalence = float(train.label.mean())
    for expected in fit_status["checkpoints"]:
        tag = str(expected["tag"])
        checkpoint = args.fit_dir.resolve() / tag
        checkpoint_manifest = _verify_checkpoint(checkpoint, expected)
        scores = _score_one(
            config=config,
            model_path=args.model_path.resolve(),
            adapter_path=checkpoint,
            query=query,
            batch_size=args.batch_size,
        )
        ties = [
            hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            for value in query.blind_uid
        ]
        prediction = prevalence_predictions(
            scores,
            ties,
            prevalence=train_prevalence,
        )
        frame = query[
            ["id", "blind_uid", "component_key", "fold", "label"]
        ].copy()
        frame["score"] = scores
        frame["prediction"] = prediction
        output_dir = args.output_dir / tag
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / "predictions.local_only.parquet"
        frame.to_parquet(output_path, index=False)
        manifest = bind_self_hash(
            {
                "schema_version": "v170.oof_predictions.1",
                "status": "COMPLETE",
                "arm": fit_status["arm"],
                "query_fold": int(fit_status["query_fold"]),
                "train_folds": fit_status["train_folds"],
                "checkpoint_tag": tag,
                "checkpoint_manifest_self_sha256": checkpoint_manifest["self_sha256"],
                "fit_status_sha256": sha256_file(fit_status_path),
                "fit_status_self_sha256": fit_status["self_sha256"],
                "rows": len(frame),
                "positives": int(frame.label.sum()),
                "predicted_positives": int(frame.prediction.sum()),
                "train_prevalence": train_prevalence,
                "all_scores_finite": True,
                "query_labels_used_for_scoring_or_threshold": False,
                "prediction_policy": "outer_train_prevalence_rank",
                "predictions_file": output_path.name,
                "predictions_sha256": sha256_file(output_path),
            }
        )
        manifest_path = output_dir / "predictions_manifest.json"
        write_json_atomic(manifest_path, manifest)
        checkpoint_results.append(
            {
                "tag": tag,
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_self_sha256": manifest["self_sha256"],
                "predictions_sha256": sha256_file(output_path),
            }
        )
    status = bind_self_hash(
        {
            "schema_version": "v170.oof_score_status.1",
            "status": "COMPLETE",
            "arm": fit_status["arm"],
            "query_fold": int(fit_status["query_fold"]),
            "fit_status_sha256": sha256_file(fit_status_path),
            "fit_status_self_sha256": fit_status["self_sha256"],
            "base_snapshot_files_sha256": snapshot["files_sha256"],
            "checkpoint_results": checkpoint_results,
            "environment": environment_fingerprint(include_cuda=True),
        }
    )
    write_json_atomic(args.output_dir / "score_status.json", status)
    return status


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = score(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
