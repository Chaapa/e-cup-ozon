#!/usr/bin/env python3
"""Validate all V170 data/schedules on CPU without loading or fitting a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .contracts import (
    bind_self_hash,
    environment_fingerprint,
    load_config,
    load_self_hashed_json,
    sha256_file,
    write_json_atomic,
)
from .data_contracts import (
    load_ce_dataset,
    load_r4_pairs,
    load_r5_pairs,
    load_r7_pairs,
    select_for_query_fold,
    select_r7_for_query_fold,
    split_ce_rows,
)
from .schedules import build_schedule, schedule_serializable_view


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--r5-manifest", type=Path, required=True)
    parser.add_argument("--r7-manifest", type=Path, required=True)
    parser.add_argument("--recipe-precommit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("V170 CPU dry-run output must be written once")
    config = load_config(args.config.resolve())
    manifest_paths = {
        "config": args.config.resolve(),
        "ce_manifest": args.ce_manifest.resolve(),
        "r4_manifest": args.r4_manifest.resolve(),
        "r5_manifest": args.r5_manifest.resolve(),
        "r7_manifest": args.r7_manifest.resolve(),
    }
    precommit = load_self_hashed_json(
        args.recipe_precommit.resolve(), context="V170 CPU dry-run recipe precommit"
    )
    expected_files = {key: sha256_file(path) for key, path in manifest_paths.items()}
    if (
        precommit.get("schema_version") != "v170.recipe_precommit.1"
        or precommit.get("status") != "FROZEN_BEFORE_OOF_AND_Q4_SCORE"
        or precommit.get("files") != expected_files
        or precommit.get("external_eval_feedback_used") is not False
        or precommit.get("contains_external_eval_rows") is not False
        or precommit.get("contains_external_eval_labels") is not False
    ):
        raise RuntimeError("V170 CPU dry-run recipe precommit drift")

    rows, ce_binding = load_ce_dataset(args.ce_manifest.resolve(), config)
    r4, r4_binding = load_r4_pairs(args.r4_manifest.resolve(), rows, config)
    r5, r5_binding = load_r5_pairs(args.r5_manifest.resolve(), rows, config)
    r7, r7_binding = load_r7_pairs(args.r7_manifest.resolve(), rows, config)
    data = config["data"]
    development = [int(value) for value in data["supplementary_development_folds"]]
    forward = int(data["forward_gate_fold"])
    folds = [int(value) for value in data["expected_folds"]]
    trajectories: list[dict[str, Any]] = []
    for query_fold in folds:
        train, query = split_ce_rows(
            rows,
            query_fold=query_fold,
            development_folds=development,
            forward_gate_fold=forward,
        )
        selected_r4 = select_for_query_fold(r4, query_fold)
        selected_r5 = select_for_query_fold(r5, query_fold)
        selected_r7 = select_r7_for_query_fold(r7, query_fold)
        arm_schedules = {
            arm: build_schedule(
                train,
                selected_r4,
                selected_r5,
                selected_r7,
                arm=arm,
                config=config,
            )
            for arm in ("control", "candidate")
        }
        for key in ("ce_order_sha256", "r4_order_sha256", "r5_order_sha256"):
            if arm_schedules["control"][key] != arm_schedules["candidate"][key]:
                raise RuntimeError(f"V170 CPU dry-run matched-arm drift at {key}")
        if (
            arm_schedules["control"]["r7_draws"] != 0
            or arm_schedules["candidate"]["r7_draws"] <= 0
            or (query_fold in development and forward in set(train.fold.astype(int)))
            or (query_fold == forward and set(train.fold.astype(int)) != set(development))
        ):
            raise RuntimeError("V170 CPU dry-run fold closure/R7 channel drift")
        for arm in ("control", "candidate"):
            trajectories.append(
                {
                    "arm": arm,
                    "query_fold": query_fold,
                    "train_folds": sorted(train.fold.astype(int).unique().tolist()),
                    "train_rows": len(train),
                    "query_rows": len(query),
                    "r4_eligible": len(selected_r4),
                    "r5_eligible": len(selected_r5),
                    "r7_eligible": len(selected_r7),
                    "schedule": schedule_serializable_view(arm_schedules[arm]),
                }
            )

    train, query = split_ce_rows(
        rows,
        query_fold=None,
        development_folds=development,
        forward_gate_fold=forward,
    )
    if len(query) != 0 or len(train) != len(rows):
        raise RuntimeError("V170 CPU dry-run full-fit split drift")
    fullfit_r7 = select_r7_for_query_fold(r7, None)
    fullfit_schedules = {
        arm: build_schedule(train, r4, r5, fullfit_r7, arm=arm, config=config)
        for arm in ("control", "candidate")
    }
    for key in ("ce_order_sha256", "r4_order_sha256", "r5_order_sha256"):
        if fullfit_schedules["control"][key] != fullfit_schedules["candidate"][key]:
            raise RuntimeError(f"V170 CPU dry-run full-fit matched-arm drift at {key}")

    report = bind_self_hash(
        {
            "schema_version": "v170.organizer_cpu_dry_run.1",
            "status": "PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT",
            "experiment_id": config["experiment_id"],
            "recipe_precommit_sha256": sha256_file(args.recipe_precommit.resolve()),
            "recipe_precommit_self_sha256": precommit["self_sha256"],
            "bindings": {
                "ce": ce_binding,
                "r4": r4_binding,
                "r5": r5_binding,
                "r7": r7_binding,
            },
            "counts": {
                "organizer_rows": len(rows),
                "organizer_positives": int(rows.label.sum()),
                "r4_pairs": len(r4),
                "r5_pairs": len(r5),
                "r5_training_pairs_including_reviewer_flags": len(r5),
                "r5_blind_usable": int((~r5.reviewer_flag).sum()),
                "r5_reviewer_flagged": int(r5.reviewer_flag.sum()),
                "r7_pairs": len(r7),
                "oof_trajectories": len(trajectories),
                "fullfit_trajectories": 2,
            },
            "r5_aggregate_gate_passed": True,
            "r5_reviewer_flags_preserved": True,
            "forward_fold_closed_during_development": True,
            "trajectories": trajectories,
            "fullfit_schedules": {
                arm: schedule_serializable_view(schedule)
                for arm, schedule in fullfit_schedules.items()
            },
            "gpu_loaded": False,
            "model_loaded": False,
            "fit_executed": False,
            "external_api_calls": 0,
            "network_calls": 0,
            "environment": environment_fingerprint(include_cuda=False),
            "next_step": "create immutable base snapshot and H100 network-disabled receipt; then execute the frozen DAG without changing data or recipe",
        }
    )
    write_json_atomic(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
