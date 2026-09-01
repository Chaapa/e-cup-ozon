#!/usr/bin/env python3
"""Apply the one-shot q4 transfer gate to the frozen V170 candidate recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from .contracts import (
    bind_self_hash,
    load_config,
    load_self_hashed_json,
    sha256_file,
    write_json_atomic,
)
from .data_contracts import load_ce_dataset
from .fit_selector import audit_oof, build_cpu_oof
from .score_lora import prevalence_predictions
from .selector_features import (
    build_augmented_features,
    build_base_features,
    component_class_weights,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--recipe-precommit", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _model(config: dict[str, Any]) -> Any:
    from catboost import CatBoostClassifier

    recipe = config["selector"]
    return CatBoostClassifier(
        iterations=int(recipe["iterations"]),
        depth=int(recipe["depth"]),
        learning_rate=float(recipe["learning_rate"]),
        l2_leaf_reg=float(recipe["l2_leaf_reg"]),
        random_strength=float(recipe["random_strength"]),
        bootstrap_type=str(recipe["bootstrap_type"]),
        loss_function=str(recipe["loss_function"]),
        random_seed=int(recipe["random_seed"]),
        thread_count=int(recipe["thread_count"]),
        verbose=False,
        allow_writing_files=False,
    )


def _metrics(truth: pd.Series, score: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = truth.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "rows": len(target),
        "positives": int(target.sum()),
        "predicted_positives": int(prediction.sum()),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(target, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("V170 q4 gate requires a fresh output directory")
    config = load_config(args.config.resolve())
    precommit = load_self_hashed_json(
        args.recipe_precommit.resolve(), context="V170 recipe precommit"
    )
    if (
        precommit.get("schema_version") != "v170.recipe_precommit.1"
        or precommit.get("status") != "FROZEN_BEFORE_OOF_AND_Q4_SCORE"
        or precommit.get("files", {}).get("config") != sha256_file(args.config.resolve())
        or precommit.get("files", {}).get("ce_manifest")
        != sha256_file(args.ce_manifest.resolve())
    ):
        raise RuntimeError("V170 q4 gate recipe precommit drift")
    rows, ce_binding = load_ce_dataset(args.ce_manifest.resolve(), config)
    rows = rows.copy()
    rows["component_size"] = rows.component_key.map(rows.groupby("component_key").size())
    signals, oof_audit = audit_oof(rows, args.oof_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output_dir / "oof_audit.json", oof_audit)
    cpu, per_fold = build_cpu_oof(rows, config)
    cpu_path = args.output_dir / "cpu_oof.local_only.parquet"
    cpu.to_parquet(cpu_path, index=False)
    cpu_signals = cpu[["id", "b1_score", "b1_prediction", "char_score", "char_prediction"]]
    control = cpu_signals.merge(signals["control"], on="id", validate="one_to_one")
    candidate = cpu_signals.merge(signals["candidate"], on="id", validate="one_to_one")
    control_matrix, control_columns = build_base_features(rows, control)
    candidate_matrix, candidate_columns = build_augmented_features(rows, control, candidate)
    train_mask = rows.fold.astype(int).ne(4).to_numpy()
    query_mask = ~train_mask
    if int(train_mask.sum()) == 0 or int(query_mask.sum()) == 0:
        raise RuntimeError("V170 q4 gate train/query split is empty")
    control_model = _model(config)
    control_model.fit(
        control_matrix.loc[train_mask, control_columns],
        control_matrix.loc[train_mask, "label"].astype(int),
        sample_weight=component_class_weights(control_matrix.loc[train_mask].reset_index(drop=True)),
    )
    candidate_model = _model(config)
    candidate_model.fit(
        candidate_matrix.loc[train_mask, candidate_columns],
        candidate_matrix.loc[train_mask, "label"].astype(int),
        sample_weight=component_class_weights(candidate_matrix.loc[train_mask].reset_index(drop=True)),
    )
    control_score = np.asarray(
        control_model.predict_proba(control_matrix.loc[query_mask, control_columns])[:, 1],
        dtype=np.float64,
    )
    candidate_score = np.asarray(
        candidate_model.predict_proba(candidate_matrix.loc[query_mask, candidate_columns])[:, 1],
        dtype=np.float64,
    )
    query = rows.loc[query_mask].reset_index(drop=True)
    prevalence = float(rows.loc[train_mask, "label"].mean())
    ties = [str(value) for value in query.blind_uid]
    control_prediction = prevalence_predictions(control_score, ties, prevalence=prevalence)
    candidate_prediction = prevalence_predictions(candidate_score, ties, prevalence=prevalence)
    control_metrics = _metrics(query.label, control_score, control_prediction)
    candidate_metrics = _metrics(query.label, candidate_score, candidate_prediction)
    truth = query.label.astype(int).to_numpy()
    control_correct = control_prediction == truth
    candidate_correct = candidate_prediction == truth
    comparison = {
        "delta_f1": candidate_metrics["f1"] - control_metrics["f1"],
        "delta_average_precision": candidate_metrics["average_precision"]
        - control_metrics["average_precision"],
        "rescues": int((candidate_correct & ~control_correct).sum()),
        "harms": int((~candidate_correct & control_correct).sum()),
    }
    passed = (
        comparison["delta_f1"] > 0.0
        and comparison["rescues"] > comparison["harms"]
        and comparison["delta_average_precision"] >= -0.01
    )
    report = bind_self_hash(
        {
            "schema_version": "v170.q4_gate.1",
            "status": "GO" if passed else "STOP",
            "train_folds": [0, 1, 2, 3],
            "query_fold": 4,
            "candidate_recipe_frozen_before_gate": True,
            "recipe_precommit_sha256": sha256_file(args.recipe_precommit.resolve()),
            "recipe_precommit_self_sha256": precommit["self_sha256"],
            "fold4_used_for_candidate_selection": False,
            "fold4_used_for_gate_only": True,
            "rule_features_used": False,
            "all_expert_inputs_oof": True,
            "control": control_metrics,
            "candidate": candidate_metrics,
            "comparison": comparison,
            "gates": {
                "delta_f1_gt_zero": comparison["delta_f1"] > 0.0,
                "rescues_gt_harms": comparison["rescues"] > comparison["harms"],
                "delta_average_precision_ge_minus_0_01": comparison[
                    "delta_average_precision"
                ]
                >= -0.01,
            },
            "ce_binding": ce_binding,
            "oof_audit_self_sha256": oof_audit["self_sha256"],
            "cpu_oof_sha256": sha256_file(cpu_path),
            "cpu_per_fold": per_fold,
            "external_eval_rows_read": 0,
            "external_eval_labels_read": 0,
        }
    )
    write_json_atomic(args.output_dir / "q4_gate_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
