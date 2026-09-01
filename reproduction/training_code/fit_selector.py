#!/usr/bin/env python3
"""Audit OOF trajectories or fit the fixed 71-feature selector."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

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
from .score_lora import prevalence_predictions
from .selector_features import (
    build_augmented_features,
    component_class_weights,
)


TAGS = ("checkpoint_025", "checkpoint_050", "checkpoint_100")
TAG_NAMES = {
    "checkpoint_025": "lora025",
    "checkpoint_050": "lora050",
    "checkpoint_100": "lora100",
}
ARMS = ("control", "candidate")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "fit"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _feature_text(frame: pd.DataFrame) -> list[str]:
    return [
        f"title {name}\n description {description}"
        for name, description in zip(
            frame.normalized_name.fillna("").astype(str),
            frame.normalized_description.fillna("").astype(str),
            strict=True,
        )
    ]


def _ties(frame: pd.DataFrame) -> list[str]:
    return [
        hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        for value in frame.blind_uid
    ]


def _fit_b1(train: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    train_text = _feature_text(train)
    query_text = _feature_text(query)
    train_matrix = hstack(
        [char.fit_transform(train_text), word.fit_transform(train_text)], format="csr"
    )
    query_matrix = hstack(
        [char.transform(query_text), word.transform(query_text)], format="csr"
    )
    model = LinearSVC(
        C=1.0,
        class_weight=None,
        dual="auto",
        max_iter=20_000,
        random_state=20260822,
    )
    model.fit(
        train_matrix,
        train.label.astype(int),
        sample_weight=component_class_weights(train.reset_index(drop=True)),
    )
    score = np.asarray(model.decision_function(query_matrix), dtype=np.float64)
    prediction = prevalence_predictions(
        score,
        _ties(query),
        prevalence=float(train.label.mean()),
    )
    del char, word, train_matrix, query_matrix, model
    gc.collect()
    return score, prediction


def _fit_char(train: pd.DataFrame, query: pd.DataFrame, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import expit
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=160_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(_feature_text(train)).tocsr()
    query_matrix = vectorizer.transform(_feature_text(query)).tocsr()
    model = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    model.fit(
        train_matrix,
        train.label.astype(int),
        sample_weight=component_class_weights(train.reset_index(drop=True)),
    )
    decision = np.asarray(model.decision_function(query_matrix), dtype=np.float64)
    del vectorizer, train_matrix, query_matrix, model
    gc.collect()
    return expit(decision), (decision >= 0).astype(np.int8)


def build_cpu_oof(
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    per_fold: dict[str, Any] = {}
    for fold in sorted(rows.fold.unique()):
        train, query = split_ce_rows(
            rows,
            query_fold=int(fold),
            development_folds=config["data"]["supplementary_development_folds"],
            forward_gate_fold=int(config["data"]["forward_gate_fold"]),
        )
        if set(train.component_key) & set(query.component_key):
            raise RuntimeError(f"V170 selector CPU component leakage at fold {fold}")
        b1_score, b1_prediction = _fit_b1(train, query)
        char_score, char_prediction = _fit_char(train, query, seed=20260823 + int(fold))
        part = query[["id", "fold", "label"]].copy()
        part["b1_score"] = b1_score
        part["b1_prediction"] = b1_prediction
        part["char_score"] = char_score
        part["char_prediction"] = char_prediction
        parts.append(part)
        per_fold[str(int(fold))] = {
            "train_rows": len(train),
            "train_positives": int(train.label.sum()),
            "query_rows": len(query),
            "query_positives": int(query.label.sum()),
            "component_overlap": 0,
            "b1_predicted_positives": int(b1_prediction.sum()),
            "char_predicted_positives": int(char_prediction.sum()),
        }
    result = pd.concat(parts, ignore_index=True).sort_values("id", kind="stable")
    if len(result) != len(rows) or result.id.astype(str).duplicated().any():
        raise RuntimeError("V170 CPU OOF coverage drift")
    return result.reset_index(drop=True), per_fold


def _load_prediction_file(
    path: Path,
    manifest_path: Path,
    *,
    arm: str,
    fold: int,
    tag: str,
    expected_rows: pd.DataFrame,
    expected_train_folds: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_self_hashed_json(
        manifest_path,
        context=f"V170 {arm} q{fold} {tag} predictions",
    )
    if (
        manifest.get("schema_version") != "v170.oof_predictions.1"
        or manifest.get("status") != "COMPLETE"
        or manifest.get("arm") != arm
        or int(manifest.get("query_fold", -1)) != fold
        or manifest.get("train_folds") != expected_train_folds
        or manifest.get("checkpoint_tag") != tag
        or manifest.get("query_labels_used_for_scoring_or_threshold") is not False
        or manifest.get("prediction_policy") != "outer_train_prevalence_rank"
        or manifest.get("predictions_sha256") != sha256_file(path)
    ):
        raise RuntimeError(f"V170 OOF prediction manifest drift: {manifest_path}")
    frame = pd.read_parquet(path)
    required = {"id", "blind_uid", "component_key", "fold", "label", "score", "prediction"}
    if (
        set(frame.columns) != required
        or len(frame) != len(expected_rows)
        or frame.id.astype(str).duplicated().any()
        or not frame.fold.astype(int).eq(fold).all()
        or not frame.sort_values("id").id.astype(str).reset_index(drop=True).equals(
            expected_rows.sort_values("id").id.astype(str).reset_index(drop=True)
        )
        or not frame.sort_values("id").label.astype(int).reset_index(drop=True).equals(
            expected_rows.sort_values("id").label.astype(int).reset_index(drop=True)
        )
        or not np.isfinite(frame.score.astype(float)).all()
        or not set(frame.prediction.astype(int)).issubset({0, 1})
    ):
        raise RuntimeError(f"V170 OOF prediction schema/identity drift: {path}")
    return frame, {
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "predictions_sha256": sha256_file(path),
        "fit_status_sha256": manifest["fit_status_sha256"],
        "fit_status_self_sha256": manifest["fit_status_self_sha256"],
    }


def load_arm_oof(
    root: Path,
    rows: pd.DataFrame,
    *,
    arm: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals = rows[["id"]].copy()
    hashes: dict[str, Any] = {}
    folds = sorted(rows.fold.unique().tolist())
    forward_fold = max(folds)
    development_folds = [value for value in folds if value != forward_fold]
    parent_fit_hashes: dict[str, set[str]] = {}
    for tag in TAGS:
        parts = []
        for fold in folds:
            directory = root / arm / f"fold_{fold}" / tag
            path = directory / "predictions.local_only.parquet"
            manifest_path = directory / "predictions_manifest.json"
            if not path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Incomplete V170 OOF path: {directory}")
            query = rows.loc[rows.fold.eq(fold)].copy()
            frame, binding = _load_prediction_file(
                path,
                manifest_path,
                arm=arm,
                fold=int(fold),
                tag=tag,
                expected_rows=query,
                expected_train_folds=(
                    development_folds
                    if int(fold) == forward_fold
                    else [value for value in development_folds if value != int(fold)]
                ),
            )
            hashes[str(directory.resolve())] = binding
            parent_fit_hashes.setdefault(str(fold), set()).add(binding["fit_status_sha256"])
            parts.append(
                frame[["id", "score", "prediction"]].rename(
                    columns={
                        "score": f"{TAG_NAMES[tag]}_score",
                        "prediction": f"{TAG_NAMES[tag]}_prediction",
                    }
                )
            )
        trajectory = pd.concat(parts, ignore_index=True)
        signals = signals.merge(trajectory, on="id", validate="one_to_one")
    if len(signals) != len(rows):
        raise RuntimeError(f"V170 {arm} OOF signal merge lost rows")
    if any(len(values) != 1 for values in parent_fit_hashes.values()):
        raise RuntimeError(f"V170 {arm} checkpoints within a fold came from different fits")
    return signals, {
        "artifact_bindings": hashes,
        "per_fold_fit_status_sha256": {
            fold: next(iter(values)) for fold, values in parent_fit_hashes.items()
        },
    }


def audit_oof(
    rows: pd.DataFrame,
    root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    signals: dict[str, pd.DataFrame] = {}
    bindings: dict[str, Any] = {}
    for arm in ARMS:
        signals[arm], bindings[arm] = load_arm_oof(root, rows, arm=arm)
    if not signals["control"].id.astype(str).equals(signals["candidate"].id.astype(str)):
        raise RuntimeError("V170 control/candidate OOF rosters differ")
    report = bind_self_hash(
        {
            "schema_version": "v170.selector_oof_audit.1",
            "status": "AUDIT_PASS",
            "rows": len(rows),
            "positives": int(rows.label.sum()),
            "folds": sorted(rows.fold.unique().tolist()),
            "arms": list(ARMS),
            "checkpoint_tags": list(TAGS),
            "all_expert_inputs_oof": True,
            "external_eval_rows_read": 0,
            "external_eval_labels_read": 0,
            "bindings": bindings,
        }
    )
    return signals, report


def fit_selector(args: argparse.Namespace) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    if args.output_dir.exists():
        raise FileExistsError("V170 selector fit requires a fresh output directory")
    config = load_config(args.config.resolve())
    rows, ce_binding = load_ce_dataset(args.ce_manifest.resolve(), config)
    rows = rows.copy()
    component_sizes = rows.groupby("component_key").size()
    rows["component_size"] = rows.component_key.map(component_sizes).astype(int)
    signals, audit = audit_oof(rows, args.oof_root.resolve())
    if args.phase == "audit":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(args.output_dir / "oof_audit.json", audit)
        return audit
    if args.gate_report is None:
        raise RuntimeError("V170 selector fit requires the precommitted q4 GO report")
    gate = load_self_hashed_json(
        args.gate_report.resolve(), context="V170 q4 gate report"
    )
    if (
        gate.get("schema_version") != "v170.q4_gate.1"
        or gate.get("status") != "GO"
        or gate.get("query_fold") != 4
        or gate.get("train_folds") != [0, 1, 2, 3]
        or gate.get("fold4_used_for_candidate_selection") is not False
        or gate.get("candidate_recipe_frozen_before_gate") is not True
        or float(gate.get("comparison", {}).get("delta_f1", -1.0)) <= 0.0
        or int(gate.get("comparison", {}).get("rescues", -1))
        <= int(gate.get("comparison", {}).get("harms", 10**9))
        or float(gate.get("comparison", {}).get("delta_average_precision", -1.0)) < -0.01
    ):
        raise RuntimeError("V170 full selector requires a passing q4 gate")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output_dir / "oof_audit.json", audit)
    cpu, per_fold = build_cpu_oof(rows, config)
    cpu_path = args.output_dir / "cpu_oof.local_only.parquet"
    cpu.to_parquet(cpu_path, index=False)
    control = cpu[["id", "b1_score", "b1_prediction", "char_score", "char_prediction"]].merge(
        signals["control"], on="id", validate="one_to_one"
    )
    candidate = cpu[["id", "b1_score", "b1_prediction", "char_score", "char_prediction"]].merge(
        signals["candidate"], on="id", validate="one_to_one"
    )
    matrix, feature_columns = build_augmented_features(rows, control, candidate)
    matrix_path = args.output_dir / "selector_feature_matrix.local_only.parquet"
    matrix.to_parquet(matrix_path, index=False)
    recipe = config["selector"]
    model = CatBoostClassifier(
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
    model.fit(
        matrix[feature_columns],
        matrix.label.astype(int),
        sample_weight=component_class_weights(matrix),
    )
    model_path = args.output_dir / "selector_augmented_fold01234.cbm"
    model.save_model(model_path)
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise RuntimeError("V170 selector model was not saved")
    manifest = bind_self_hash(
        {
            "schema_version": "v170.selector_manifest.1",
            "status": "COMPLETE_UNSCORED",
            "purpose": "production_selector_after_matched_oof_reproduction",
            "fit_rows": len(matrix),
            "fit_positives": int(matrix.label.sum()),
            "fit_folds": sorted(matrix.fold.unique().tolist()),
            "all_expert_inputs_oof": True,
            "control_arm": "control",
            "candidate_arm": "candidate",
            "rule_features_used": False,
            "recipe_frozen_before_fold4": True,
            "fold4_used_for_final_meta_fit": True,
            "fold4_used_for_candidate_selection": False,
            "q4_gate_passed": True,
            "q4_gate_report_sha256": sha256_file(args.gate_report.resolve()),
            "q4_gate_report_self_sha256": gate["self_sha256"],
            "q4_gate_delta_f1": float(gate["comparison"]["delta_f1"]),
            "q4_gate_delta_average_precision": float(
                gate["comparison"]["delta_average_precision"]
            ),
            "q4_gate_rescues": int(gate["comparison"]["rescues"]),
            "q4_gate_harms": int(gate["comparison"]["harms"]),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "ce_binding": ce_binding,
            "oof_audit_self_sha256": audit["self_sha256"],
            "oof_audit_sha256": sha256_file(args.output_dir / "oof_audit.json"),
            "cpu_oof": {
                "path": cpu_path.name,
                "bytes": cpu_path.stat().st_size,
                "sha256": sha256_file(cpu_path),
                "per_fold": per_fold,
            },
            "feature_matrix": {
                "path": matrix_path.name,
                "bytes": matrix_path.stat().st_size,
                "sha256": sha256_file(matrix_path),
                "contains_only_organizer_train_labels": True,
            },
            "model_file": model_path.name,
            "model_bytes": model_path.stat().st_size,
            "model_sha256": sha256_file(model_path),
            "environment": environment_fingerprint(include_cuda=False),
            "selector_source_sha256": sha256_file(Path(__file__)),
        }
    )
    write_json_atomic(args.output_dir / "selector_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = fit_selector(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
