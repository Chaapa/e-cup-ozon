from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training_code.contracts import (
    bind_self_hash,
    load_config,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from training_code.data_contracts import (
    CE_SCHEMA,
    R4_SCHEMA,
    R5_SCHEMA,
    R7_SCHEMA,
    load_ce_dataset,
    load_r4_pairs,
    load_r5_pairs,
    load_r7_pairs,
    select_for_query_fold,
    select_r7_for_query_fold,
    split_ce_rows,
)
from training_code.freeze_training_code import freeze
from training_code.fit_selector import audit_oof
from training_code.organizer_cpu_dry_run import (
    parse_args as parse_cpu_dry_run_args,
    run as run_cpu_dry_run,
)
from training_code.schedules import build_schedule
from training_code.selector_features import (
    build_augmented_features,
)
from training_code.train_lora import main as train_lora_main


BASE_CONFIG = Path(__file__).resolve().parents[1] / "config.json"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _train_only() -> dict:
    return {
        "source_partition": "organizer_train",
        "contains_external_eval_rows": False,
        "contains_external_eval_labels": False,
        "contains_external_eval_predictions": False,
        "contains_private_rows": False,
        "contains_private_labels": False,
        "contains_hidden_labels": False,
        "external_eval_feedback_used": False,
    }


def _bound_file(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _fixture(tmp_path: Path) -> dict:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["data"].update(
        {
            "expected_full_rows": 200,
            "expected_full_positives": 5,
            "expected_r4_pairs": 5,
            "expected_r5_pairs": 200,
            "minimum_r5_style_frames": 40,
            "maximum_r5_pairs_per_style_frame": 5,
            "expected_r7_relation_instances": 5,
            "expected_r7_mass_keys": 5,
            "expected_r7_query_counts": {"0": 2, "1": 1, "2": 1, "3": 1},
        }
    )
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, config)
    records = []
    for fold in range(5):
        for offset in range(40):
            identifier = fold * 40 + offset
            label = int(offset == 0)
            records.append(
                {
                    "id": identifier,
                    "blind_uid": f"uid-{identifier}",
                    "label": label,
                    "fold": fold,
                    "component_key": f"component-{identifier}",
                    "clean_name": f"Товар {identifier}",
                    "clean_description": (
                        f"Описание положительного товара {identifier}"
                        if label
                        else f"Описание обычного товара {identifier}"
                    ),
                    "normalized_name": f"товар {identifier}",
                    "normalized_description": f"описание {identifier}",
                    "image_count": offset,
                }
            )
    rows = pd.DataFrame(records)
    rows_path = tmp_path / "rows.parquet"
    rows.to_parquet(rows_path, index=False)
    ce_manifest_path = tmp_path / "ce_manifest.json"
    ce = bind_self_hash(
        {
            "schema_version": CE_SCHEMA,
            "status": "PASS",
            **_train_only(),
            "rows_file": _bound_file(rows_path),
            "rows": 200,
            "positives": 5,
            "folds": [0, 1, 2, 3, 4],
            "components": 200,
        }
    )
    write_json_atomic(ce_manifest_path, ce)

    r4_records = []
    r7_records = []
    pair_specs = [
        (0, 0, 1),
        (1, 40, 41),
        (2, 80, 81),
        (3, 120, 121),
        (0, 0, 3),
    ]
    for index, (fold, positive, negative) in enumerate(pair_specs):
        eligible = [value for value in range(5) if value != fold]
        r4_records.append(
            {
                "pair_id": f"r4-{index}",
                "positive_id": positive,
                "negative_id": negative,
                "boundary_code": f"boundary-{fold}",
                "review_status": "ACCEPT",
                "eligible_query_folds": eligible,
            }
        )
        r7_records.append(
            {
                "pair_id": f"r7-{index}",
                "mass_key": f"mass-{index}",
                "query_fold": fold,
                "positive_id": positive,
                "negative_id": negative + 1,
                "boundary_code": f"r7-boundary-{index}",
                "review_status": "ACCEPT",
                "eligible_query_folds": [fold],
            }
        )
    r4_path = tmp_path / "r4.jsonl"
    r7_path = tmp_path / "r7.jsonl"
    _write_jsonl(r4_path, r4_records)
    _write_jsonl(r7_path, r7_records)
    r4_manifest_path = tmp_path / "r4_manifest.json"
    r7_manifest_path = tmp_path / "r7_manifest.json"
    write_json_atomic(
        r4_manifest_path,
        bind_self_hash(
            {
                "schema_version": R4_SCHEMA,
                "status": "PASS",
                **_train_only(),
                "pairs_file": _bound_file(r4_path),
                "pairs": 5,
                "positive_components": 4,
                "negative_components": 5,
                "boundaries": 4,
            }
        ),
    )
    write_json_atomic(
        r7_manifest_path,
        bind_self_hash(
            {
                "schema_version": R7_SCHEMA,
                "status": "PASS",
                **_train_only(),
                "pairs_file": _bound_file(r7_path),
                "pairs": 5,
                "mass_keys": 5,
                "positive_components": 4,
                "negative_components": 5,
                "boundaries": 5,
                "query_counts": {"0": 2, "1": 1, "2": 1, "3": 1},
            }
        ),
    )
    r5_records = []
    r5_reviews = []
    r5_structural = []
    r5_cards = []
    for index in range(200):
        pair_id = f"r5-{index:03d}"
        positive_span = f"горючий слот присутствует {index}"
        negative_span = f"горючий слот отсутствует {index}"
        positive = {
            "name": f"Синтетический товар {index}",
            "description": f"Карточка {index}. Решающий факт: {positive_span}",
            "decisive_span": positive_span,
        }
        negative = {
            "name": f"Синтетический товар {index}",
            "description": f"Карточка {index}. Решающий факт: {negative_span}",
            "decisive_span": negative_span,
        }
        positive_first = index % 2 == 0
        r5_records.append(
            {
                "pair_id": pair_id,
                "positive": positive,
                "negative": negative,
                "boundary": f"synthetic-boundary-{index % 40}",
                "style_frame_id": f"style-{index % 40}",
                "changed_slot": "flammable_slot_present",
                "changed_field": "description",
                "invariant_fact_frame": f"Синтетический товар {index} | Карточка {index}",
                "positive_first": positive_first,
                "variant_index": index,
            }
        )
        usable = index < 192
        blind_swap = int(sha256_text(pair_id), 16) % 2 == 1
        if not blind_swap:
            label_a, label_b = 1, 0
            evidence_a, evidence_b = positive_span, negative_span
        else:
            label_a, label_b = 0, 1
            evidence_a, evidence_b = negative_span, positive_span
        r5_reviews.append(
            {
                "pair_id": pair_id,
                "label_a": label_a,
                "label_b": label_b,
                "evidence_a": evidence_a,
                "evidence_b": evidence_b,
                "observed_slot": "flammable_slot_present",
                "pair_usable": usable,
                "reject_reason": "NONE" if usable else "reviewer ambiguity retained",
                "same_product_frame": usable,
                "single_causal_boundary": usable,
            }
        )
        r5_structural.append(
            {
                "pair_id": pair_id,
                "boundary": f"synthetic-boundary-{index % 40}",
                "style_frame_id": f"style-{index % 40}",
                "pass": True,
                "checks": {"single_slot": True, "surface_bound": True},
            }
        )
        ordered = [(1, positive), (0, negative)] if positive_first else [(0, negative), (1, positive)]
        for position, (label, side) in enumerate(ordered):
            r5_cards.append(
                {
                    "pair_id": pair_id,
                    "pair_position": position,
                    "organizer_label": label,
                    "name": side["name"],
                    "description": side["description"],
                    "decisive_span": side["decisive_span"],
                    "fully_synthetic": True,
                    "source_id": None,
                    "source_component": None,
                }
            )
    r5_path = tmp_path / "rendered_pairs.jsonl"
    r5_reviews_path = tmp_path / "blind_reviews.jsonl"
    r5_structural_path = tmp_path / "structural_audit.jsonl"
    r5_cards_path = tmp_path / "training_cards.jsonl"
    r5_metrics_path = tmp_path / "metrics.json"
    r5_artifact_manifest_path = tmp_path / "artifact_manifest.json"
    _write_jsonl(r5_path, r5_records)
    _write_jsonl(r5_reviews_path, r5_reviews)
    _write_jsonl(r5_structural_path, r5_structural)
    _write_jsonl(r5_cards_path, r5_cards)
    r5_metrics = {
        "schema_version": "v170.r5.production.metrics.1",
        "pairs": 200,
        "cards": 400,
        "structural": {"pairs": 200, "passed": 200, "pass_rate": 1.0},
        "blind_direction_correct": 200,
        "blind_direction_accuracy": 1.0,
        "blind_usable": 192,
        "blind_usable_rate": 0.96,
        "exact_train_overlap": 0,
        "copied_13_token_train_spans": 0,
        "population_audit": {
            "exact_duplicate_cards": 0,
            "exact_duplicate_masked_pairs": 0,
            "cross_frame_near_duplicate_edges_ge_0_94": 0,
        },
        "checks": {"aggregate_gate": True},
        "preregistration_checks": {"frozen_before_generation": True},
        "verdict": "PASS",
    }
    write_json_atomic(r5_metrics_path, r5_metrics)
    r5_artifact_manifest = {
        "schema_version": "v170.r5.production.artifacts.1",
        "verdict": "PASS",
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                r5_path,
                r5_reviews_path,
                r5_structural_path,
                r5_cards_path,
                r5_metrics_path,
            )
        },
    }
    write_json_atomic(r5_artifact_manifest_path, r5_artifact_manifest)
    r5_manifest_path = tmp_path / "r5_manifest.json"
    write_json_atomic(
        r5_manifest_path,
        bind_self_hash(
            {
                "schema_version": R5_SCHEMA,
                "status": "PASS",
                **{**_train_only(), "source_partition": "openrouter_synthetic_no_row_source"},
                "pairs_file": _bound_file(r5_path),
                "reviews_file": _bound_file(r5_reviews_path),
                "structural_audit_file": _bound_file(r5_structural_path),
                "training_cards_file": _bound_file(r5_cards_path),
                "metrics_file": _bound_file(r5_metrics_path),
                "artifact_manifest_file": _bound_file(r5_artifact_manifest_path),
                "pairs": 200,
                "boundaries": 40,
                "style_frames": 40,
                "max_pairs_per_style_frame": 5,
                "structural_passed": 200,
                "blind_direction_correct": 200,
                "blind_usable": 192,
                "reviewer_flagged_pairs": 8,
                "reviewer_flags_preserved": True,
                "training_pairs_including_reviewer_flags": 200,
                "aggregate_blind_usable_gate": 0.95,
            }
        ),
    )
    recipe_path = tmp_path / "recipe_precommit.json"
    write_json_atomic(
        recipe_path,
        bind_self_hash(
            {
                "schema_version": "v170.recipe_precommit.1",
                "status": "FROZEN_BEFORE_OOF_AND_Q4_SCORE",
                "files": {
                    "config": sha256_file(config_path),
                    "ce_manifest": sha256_file(ce_manifest_path),
                    "r4_manifest": sha256_file(r4_manifest_path),
                    "r5_manifest": sha256_file(r5_manifest_path),
                    "r7_manifest": sha256_file(r7_manifest_path),
                },
                "external_eval_feedback_used": False,
                "contains_external_eval_rows": False,
                "contains_external_eval_labels": False,
            }
        ),
    )
    return {
        "config": config_path,
        "rows": rows,
        "ce": ce_manifest_path,
        "r4": r4_manifest_path,
        "r4_file": r4_path,
        "r4_records": r4_records,
        "r5": r5_manifest_path,
        "r5_file": r5_path,
        "r5_reviews_file": r5_reviews_path,
        "r5_structural_file": r5_structural_path,
        "r5_cards_file": r5_cards_path,
        "r5_metrics_file": r5_metrics_path,
        "r5_artifact_manifest_file": r5_artifact_manifest_path,
        "r5_records": r5_records,
        "r7": r7_manifest_path,
        "recipe": recipe_path,
    }


def test_v170_loaders_and_matched_schedules(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture["config"])
    rows, _ = load_ce_dataset(fixture["ce"], config)
    r4, _ = load_r4_pairs(fixture["r4"], rows, config)
    r5, r5_binding = load_r5_pairs(fixture["r5"], rows, config)
    r7, _ = load_r7_pairs(fixture["r7"], rows, config)
    train, query = split_ce_rows(rows, query_fold=0)
    selected_r4 = select_for_query_fold(r4, 0)
    selected_r5 = select_for_query_fold(r5, 0)
    selected_r7 = select_r7_for_query_fold(r7, 0)
    control = build_schedule(
        train,
        selected_r4,
        selected_r5,
        selected_r7,
        arm="control",
        config=config,
    )
    candidate = build_schedule(
        train,
        selected_r4,
        selected_r5,
        selected_r7,
        arm="candidate",
        config=config,
    )
    assert len(query) == 40
    assert len(selected_r4) == 3
    assert len(selected_r7) == 2
    assert len(r5) == len(selected_r5) == 200
    assert int(r5.reviewer_flag.sum()) == 8
    assert int(r5.usable.sum()) == 192
    assert r5_binding["blind_direction_correct"] == 200
    assert r5_binding["training_pairs_including_reviewer_flags"] == 200
    assert r5_binding["reviewer_flags_preserved_in_training_frame"] is True
    assert control["ce_order_sha256"] == candidate["ce_order_sha256"]
    assert control["r4_order_sha256"] == candidate["r4_order_sha256"]
    assert control["r5_order_sha256"] == candidate["r5_order_sha256"]
    assert control["r7_draws"] == 0
    assert candidate["r7_draws"] > 0


def test_v170_rejects_external_eval_provenance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture["config"])
    manifest = json.loads(fixture["ce"].read_text(encoding="utf-8"))
    manifest.pop("self_sha256")
    manifest["contains_external_eval_rows"] = True
    write_json_atomic(fixture["ce"], bind_self_hash(manifest))
    with pytest.raises(RuntimeError, match="contains_external_eval_rows"):
        load_ce_dataset(fixture["ce"], config)


def test_v170_rejects_wrong_real_pair_direction(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture["config"])
    rows, _ = load_ce_dataset(fixture["ce"], config)
    records = fixture["r4_records"]
    records[0]["positive_id"] = 1
    _write_jsonl(fixture["r4_file"], records)
    manifest = json.loads(fixture["r4"].read_text(encoding="utf-8"))
    manifest.pop("self_sha256")
    manifest["pairs_file"] = _bound_file(fixture["r4_file"])
    write_json_atomic(fixture["r4"], bind_self_hash(manifest))
    with pytest.raises(RuntimeError, match="label direction"):
        load_r4_pairs(fixture["r4"], rows, config)


def test_v170_keeps_forward_fold_closed_during_development(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture["config"])
    rows, _ = load_ce_dataset(fixture["ce"], config)
    train, query = split_ce_rows(rows, query_fold=0)
    assert set(train.fold.astype(int)) == {1, 2, 3}
    assert set(query.fold.astype(int)) == {0}
    forward_train, forward_query = split_ce_rows(rows, query_fold=4)
    assert set(forward_train.fold.astype(int)) == {0, 1, 2, 3}
    assert set(forward_query.fold.astype(int)) == {4}

    records = fixture["r4_records"]
    records[-1]["positive_id"] = 160
    records[-1]["negative_id"] = 161
    records[-1]["eligible_query_folds"] = [0, 1, 2, 3]
    _write_jsonl(fixture["r4_file"], records)
    manifest = json.loads(fixture["r4"].read_text(encoding="utf-8"))
    manifest.pop("self_sha256")
    manifest["pairs_file"] = _bound_file(fixture["r4_file"])
    write_json_atomic(fixture["r4"], bind_self_hash(manifest))
    with pytest.raises(RuntimeError, match="forward-gate fold"):
        load_r4_pairs(fixture["r4"], rows, config)


def test_v170_rejects_synthetic_exact_train_copy(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture["config"])
    rows, _ = load_ce_dataset(fixture["ce"], config)
    records = fixture["r5_records"]
    records[0]["positive"]["name"] = str(rows.iloc[0].clean_name)
    records[0]["positive"]["description"] = str(rows.iloc[0].clean_description)
    records[0]["positive"]["decisive_span"] = str(rows.iloc[0].clean_name)
    _write_jsonl(fixture["r5_file"], records)
    artifact_manifest = json.loads(
        fixture["r5_artifact_manifest_file"].read_text(encoding="utf-8")
    )
    artifact_manifest["files"][fixture["r5_file"].name] = {
        "sha256": sha256_file(fixture["r5_file"]),
        "bytes": fixture["r5_file"].stat().st_size,
    }
    write_json_atomic(fixture["r5_artifact_manifest_file"], artifact_manifest)
    manifest = json.loads(fixture["r5"].read_text(encoding="utf-8"))
    manifest.pop("self_sha256")
    manifest["pairs_file"] = _bound_file(fixture["r5_file"])
    manifest["artifact_manifest_file"] = _bound_file(
        fixture["r5_artifact_manifest_file"]
    )
    write_json_atomic(fixture["r5"], bind_self_hash(manifest))
    with pytest.raises(RuntimeError, match="exactly copies"):
        load_r5_pairs(fixture["r5"], rows, config)


def test_v170_organizer_cpu_dry_run_preserves_r5_flags(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "organizer_cpu_dry_run.json"
    args = parse_cpu_dry_run_args(
        [
            "--config",
            str(fixture["config"]),
            "--ce-manifest",
            str(fixture["ce"]),
            "--r4-manifest",
            str(fixture["r4"]),
            "--r5-manifest",
            str(fixture["r5"]),
            "--r7-manifest",
            str(fixture["r7"]),
            "--recipe-precommit",
            str(fixture["recipe"]),
            "--output",
            str(output),
        ]
    )
    report = run_cpu_dry_run(args)
    assert report["status"] == "PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT"
    assert report["counts"]["r5_pairs"] == 200
    assert report["counts"]["r5_training_pairs_including_reviewer_flags"] == 200
    assert report["counts"]["r5_blind_usable"] == 192
    assert report["counts"]["r5_reviewer_flagged"] == 8
    assert report["r5_aggregate_gate_passed"] is True
    assert report["r5_reviewer_flags_preserved"] is True
    assert report["fit_executed"] is False
    assert report["gpu_loaded"] is False


def test_v170_rejects_r5_reviewer_flag_policy_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture["config"])
    rows, _ = load_ce_dataset(fixture["ce"], config)
    manifest = json.loads(fixture["r5"].read_text(encoding="utf-8"))
    manifest.pop("self_sha256")
    manifest["reviewer_flags_preserved"] = False
    write_json_atomic(fixture["r5"], bind_self_hash(manifest))
    with pytest.raises(RuntimeError, match="aggregate production gate"):
        load_r5_pairs(fixture["r5"], rows, config)


def test_v170_selector_feature_contract_is_exact_71(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"].copy()
    rows["component_size"] = 1
    base = rows[["id"]].copy()
    for index, name in enumerate(("b1", "char", "lora025", "lora050", "lora100")):
        base[f"{name}_score"] = np.arange(len(rows), dtype=float) * (index + 1) / 17.0
        base[f"{name}_prediction"] = (
            (np.arange(len(rows)) + index) % 3 == 0
        ).astype(np.int8)
    candidate = base.copy()
    for name in ("lora025", "lora050", "lora100"):
        candidate[f"{name}_score"] += np.linspace(-0.2, 0.2, len(rows))
        candidate[f"{name}_prediction"] = (
            candidate[f"{name}_score"] > candidate[f"{name}_score"].median()
        ).astype(np.int8)
    matrix, columns = build_augmented_features(rows, base, candidate)
    assert len(columns) == 71
    assert not any(column.startswith("rule_") for column in columns)
    assert matrix[columns].notna().all().all()
    assert np.isfinite(matrix[columns].to_numpy(dtype=float)).all()


def test_v170_training_code_snapshot_is_data_free(tmp_path: Path) -> None:
    manifest = freeze(tmp_path / "training_code")
    assert manifest["status"] == "COMPLETE"
    assert manifest["contains_organizer_rows"] is False
    assert manifest["contains_supplementary_pair_rows"] is False
    assert manifest["contains_model_weights"] is False
    assert manifest["contains_predictions"] is False
    assert manifest["contains_credentials"] is False


def test_v170_train_audit_cli_requires_no_gpu_or_fit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "audit"
    train_lora_main(
        [
            "--phase",
            "audit",
            "--arm",
            "candidate",
            "--query-fold",
            "2",
            "--config",
            str(fixture["config"]),
            "--ce-manifest",
            str(fixture["ce"]),
            "--r4-manifest",
            str(fixture["r4"]),
            "--r5-manifest",
            str(fixture["r5"]),
            "--r7-manifest",
            str(fixture["r7"]),
            "--recipe-precommit",
            str(fixture["recipe"]),
            "--output-dir",
            str(output),
        ]
    )
    report = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "AUDIT_PASS"
    assert report["query_fold"] == 2
    assert report["images_read"] == 0
    assert report["external_eval_rows_read"] == 0
    assert report["r7_channel_weight"] == 0.03


def test_v170_oof_audit_requires_all_30_bound_artifacts(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"].copy()
    root = tmp_path / "oof"
    for arm_index, arm in enumerate(("control", "candidate")):
        for fold in range(5):
            query = rows.loc[rows.fold.eq(fold)].copy()
            for tag_index, tag in enumerate(
                ("checkpoint_025", "checkpoint_050", "checkpoint_100")
            ):
                directory = root / arm / f"fold_{fold}" / tag
                directory.mkdir(parents=True, exist_ok=True)
                predictions = query[
                    ["id", "blind_uid", "component_key", "fold", "label"]
                ].copy()
                predictions["score"] = (
                    np.arange(len(query), dtype=float)
                    + arm_index * 0.1
                    + tag_index * 0.01
                )
                predictions["prediction"] = (
                    np.arange(len(query)) < int(round(len(query) * 0.025))
                ).astype(np.int8)
                prediction_path = directory / "predictions.local_only.parquet"
                predictions.to_parquet(prediction_path, index=False)
                manifest = bind_self_hash(
                    {
                        "schema_version": "v170.oof_predictions.1",
                        "status": "COMPLETE",
                        "arm": arm,
                        "query_fold": fold,
                        "train_folds": (
                            [0, 1, 2, 3]
                            if fold == 4
                            else [value for value in range(4) if value != fold]
                        ),
                        "checkpoint_tag": tag,
                        "query_labels_used_for_scoring_or_threshold": False,
                        "prediction_policy": "outer_train_prevalence_rank",
                        "predictions_sha256": sha256_file(prediction_path),
                        "fit_status_sha256": f"{fold + arm_index:064x}"[-64:],
                        "fit_status_self_sha256": f"{fold + arm_index + 10:064x}"[-64:],
                    }
                )
                write_json_atomic(directory / "predictions_manifest.json", manifest)
    signals, report = audit_oof(rows, root)
    assert report["status"] == "AUDIT_PASS"
    assert report["all_expert_inputs_oof"] is True
    assert set(signals) == {"control", "candidate"}
    assert all(len(frame) == len(rows) for frame in signals.values())
