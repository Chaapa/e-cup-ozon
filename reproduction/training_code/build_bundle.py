#!/usr/bin/env python3
"""Bind completed full fits, selector, code and final submission manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_self_hash,
    environment_fingerprint,
    load_config,
    load_self_hashed_json,
    sha256_file,
    write_json_atomic,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-code-dir", type=Path, required=True)
    parser.add_argument("--control-fit-dir", type=Path, required=True)
    parser.add_argument("--candidate-fit-dir", type=Path, required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_fit(path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    status_path = path / "fit_status.json"
    status = load_self_hashed_json(status_path, context=f"V170 {arm} full fit")
    if (
        status.get("schema_version") != "v170.lora_fit_status.1"
        or status.get("status") != "FIT_COMPLETE_UNSCORED"
        or status.get("arm") != arm
        or status.get("full_fit") is not True
        or status.get("query_fold") is not None
        or status.get("full_epoch_complete") is not True
        or [item.get("tag") for item in status.get("checkpoints", [])]
        != ["checkpoint_025", "checkpoint_050", "checkpoint_100"]
    ):
        raise RuntimeError(f"V170 {arm} full-fit status contract drift")
    files = {
        "fit_status.json": {
            "bytes": status_path.stat().st_size,
            "sha256": sha256_file(status_path),
        },
        "train_events.jsonl": {
            "bytes": (path / "train_events.jsonl").stat().st_size,
            "sha256": sha256_file(path / "train_events.jsonl"),
        },
    }
    for checkpoint in status["checkpoints"]:
        tag = checkpoint["tag"]
        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "checkpoint_manifest.json",
        ):
            source = path / tag / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            files[f"{tag}/{filename}"] = {
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
    return status, files


def _matched_contract(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    exact_same = (
        "config_sha256",
        "train_rows",
        "train_positives",
        "train_folds",
        "ce_binding",
        "r4_binding",
        "r5_binding",
        "r7_binding",
        "optimizer_steps",
        "model_snapshot",
        "adapter",
    )
    mismatched = [key for key in exact_same if control.get(key) != candidate.get(key)]
    if mismatched:
        raise RuntimeError(f"V170 matched full fits differ outside R7 gradient: {mismatched}")
    control_schedule = control["schedule"]
    candidate_schedule = candidate["schedule"]
    for key in (
        "ce_order_sha256",
        "r4_order_sha256",
        "r5_order_sha256",
        "optimizer_steps",
        "checkpoint_steps",
    ):
        if control_schedule.get(key) != candidate_schedule.get(key):
            raise RuntimeError(f"V170 matched full-fit schedule differs at {key}")
    if float(control.get("r7_channel_weight", -1)) != 0.0 or float(
        candidate.get("r7_channel_weight", -1)
    ) != 0.03:
        raise RuntimeError("V170 full-fit arm R7 weights are not 0.00/0.03")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("V170 bundle build requires a fresh output directory")
    config = load_config(args.config.resolve())
    control, control_files = _load_fit(args.control_fit_dir.resolve(), "control")
    candidate, candidate_files = _load_fit(args.candidate_fit_dir.resolve(), "candidate")
    _matched_contract(control, candidate)
    selector_path = args.selector_dir.resolve() / "selector_manifest.json"
    selector = load_self_hashed_json(selector_path, context="V170 selector")
    if (
        selector.get("schema_version") != "v170.selector_manifest.1"
        or selector.get("status") != "COMPLETE_UNSCORED"
        or selector.get("all_expert_inputs_oof") is not True
        or selector.get("feature_count") != 71
        or selector.get("rule_features_used") is not False
    ):
        raise RuntimeError("V170 selector bundle contract drift")
    code_manifest_path = args.training_code_dir.resolve() / "training_code_manifest.json"
    code = load_self_hashed_json(code_manifest_path, context="V170 training-code snapshot")
    if (
        code.get("schema_version") != "v170.training_code_snapshot.2"
        or code.get("status") != "COMPLETE"
        or any(
            code.get(key) is not False
            for key in (
                "contains_organizer_rows",
                "contains_supplementary_pair_rows",
                "contains_model_weights",
                "contains_predictions",
                "contains_credentials",
                "contains_external_eval_data",
            )
        )
    ):
        raise RuntimeError("V170 training-code snapshot safety contract drift")
    submission = load_self_hashed_json(
        args.submission_manifest.resolve(), context="final submission manifest"
    )
    if (
        submission.get("schema_version") != "v170.submission_manifest.1"
        or submission.get("status") != "FROZEN"
        or submission.get("contains_external_eval_rows") is not False
        or submission.get("contains_external_eval_labels") is not False
        or submission.get("organizer_train_only") is not True
    ):
        raise RuntimeError("Final submission manifest safety contract drift")
    selector_model = args.selector_dir.resolve() / str(selector["model_file"])
    if sha256_file(selector_model) != selector["model_sha256"]:
        raise RuntimeError("V170 selector model hash drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory = {
        "training_code_manifest": {
            "path": str(code_manifest_path),
            "sha256": sha256_file(code_manifest_path),
            "self_sha256": code["self_sha256"],
        },
        "control_fit": control_files,
        "candidate_fit": candidate_files,
        "selector": {
            "selector_manifest.json": {
                "bytes": selector_path.stat().st_size,
                "sha256": sha256_file(selector_path),
            },
            str(selector["model_file"]): {
                "bytes": selector_model.stat().st_size,
                "sha256": sha256_file(selector_model),
            },
        },
        "submission_manifest": {
            "path": str(args.submission_manifest.resolve()),
            "sha256": sha256_file(args.submission_manifest.resolve()),
            "self_sha256": submission["self_sha256"],
        },
    }
    write_json_atomic(args.output_dir / "artifact_inventory.json", inventory)
    manifest = bind_self_hash(
        {
            "schema_version": "v170.reproduction_bundle.1",
            "status": "COMPLETE_READY_FOR_PRIVATE_ARCHIVE",
            "experiment_id": config["experiment_id"],
            "config_sha256": sha256_file(args.config.resolve()),
            "control_fit_status_self_sha256": control["self_sha256"],
            "candidate_fit_status_self_sha256": candidate["self_sha256"],
            "matched_control_candidate": True,
            "only_training_difference": "candidate_adds_r7_pairwise_loss_weight_0.03",
            "selector_manifest_self_sha256": selector["self_sha256"],
            "selector_all_expert_inputs_oof": True,
            "selector_features": 71,
            "training_code_self_sha256": code["self_sha256"],
            "submission_manifest_self_sha256": submission["self_sha256"],
            "artifact_inventory_sha256": sha256_file(
                args.output_dir / "artifact_inventory.json"
            ),
            "contains_external_eval_rows": False,
            "contains_external_eval_labels": False,
            "contains_raw_organizer_rows": False,
            "archive_contents": "code, derived manifests, weights, OOF predictions, selector, environment and inference runtime",
            "environment": environment_fingerprint(include_cuda=False),
        }
    )
    write_json_atomic(args.output_dir / "reproduction_bundle_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
