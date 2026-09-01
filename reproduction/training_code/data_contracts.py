"""Fail-closed loaders for regenerated organizer-train CE, R4, R5 and R7 data."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    ALLOWED_SOURCE_PARTITION,
    SYNTHETIC_SOURCE_PARTITION,
    assert_allowed_training_provenance,
    canonical_sha256,
    load_self_hashed_json,
    read_jsonl,
    resolve_bound_file,
    sha256_file,
    sha256_text,
)


CE_SCHEMA = "v170.ce_dataset.1"
R4_SCHEMA = "v170.r4_real_pairs.1"
R5_SCHEMA = "v170.r5_synthetic_pairs.2"
R7_SCHEMA = "v170.r7_real_pairs.1"
REQUIRED_CE_COLUMNS = (
    "id",
    "blind_uid",
    "label",
    "fold",
    "component_key",
    "clean_name",
    "clean_description",
    "normalized_name",
    "normalized_description",
    "image_count",
)
FORBIDDEN_DATA_COLUMN_PATTERN = re.compile(
    r"(?:^|_)(?:external_eval|private|hidden|solution|answer|gold|truth)(?:_|$)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[\wё]+", re.IGNORECASE)


def _validate_manifest(
    path: Path,
    *,
    schema: str,
    kind: str,
    allowed_source_partitions: Sequence[str] = (ALLOWED_SOURCE_PARTITION,),
) -> dict[str, Any]:
    manifest = load_self_hashed_json(path, context=f"V170 {kind} manifest")
    if manifest.get("schema_version") != schema or manifest.get("status") != "PASS":
        raise RuntimeError(f"V170 {kind} manifest identity/status drift")
    assert_allowed_training_provenance(
        manifest,
        context=f"V170 {kind} manifest",
        allowed_source_partitions=allowed_source_partitions,
    )
    return manifest


def _id_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise RuntimeError("V170 ID cannot be null")
    return str(value)


def _normalize_surface(name: Any, description: Any) -> str:
    return " ".join(f"{name or ''}\n{description or ''}".lower().split())


def _tokens(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.lower())


def _ngrams(value: str, width: int) -> set[tuple[str, ...]]:
    tokens = _tokens(value)
    return {
        tuple(tokens[index : index + width])
        for index in range(max(0, len(tokens) - width + 1))
    }


def load_ce_dataset(
    manifest_path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _validate_manifest(
        manifest_path,
        schema=CE_SCHEMA,
        kind="CE dataset",
    )
    rows_path = resolve_bound_file(
        manifest_path,
        manifest.get("rows_file", {}),
        context="V170 CE rows",
    )
    if rows_path.suffix.lower() != ".parquet":
        raise RuntimeError("V170 CE rows must use Parquet")
    rows = pd.read_parquet(rows_path)
    missing = sorted(set(REQUIRED_CE_COLUMNS) - set(rows.columns))
    forbidden = sorted(
        column for column in rows.columns if FORBIDDEN_DATA_COLUMN_PATTERN.search(str(column))
    )
    if missing or forbidden:
        raise RuntimeError(
            f"V170 CE schema invalid: missing={missing}, forbidden={forbidden}"
        )
    rows = rows.loc[:, list(REQUIRED_CE_COLUMNS)].copy()
    rows["id_key"] = [_id_key(value) for value in rows.id]
    rows["blind_uid"] = rows.blind_uid.astype(str)
    rows["component_key"] = rows.component_key.astype(str)
    rows["label"] = rows.label.astype(int)
    rows["fold"] = rows.fold.astype(int)
    if (
        rows.empty
        or rows.id_key.duplicated().any()
        or rows.blind_uid.duplicated().any()
        or set(rows.label) != {0, 1}
        or rows[list(REQUIRED_CE_COLUMNS)].isna().any().any()
        or (rows.image_count.astype(int) < 0).any()
        or rows.groupby("component_key").fold.nunique().max() != 1
    ):
        raise RuntimeError("V170 CE row identity/target/component contract failed")
    data_config = config["data"]
    expected_folds = [int(value) for value in data_config["expected_folds"]]
    observed = {
        "rows": len(rows),
        "positives": int(rows.label.sum()),
        "folds": sorted(rows.fold.unique().tolist()),
        "components": int(rows.component_key.nunique()),
    }
    declared = {
        "rows": int(manifest.get("rows", -1)),
        "positives": int(manifest.get("positives", -1)),
        "folds": [int(value) for value in manifest.get("folds", [])],
        "components": int(manifest.get("components", -1)),
    }
    if observed != declared:
        raise RuntimeError(f"V170 CE manifest inventory drift: {observed} != {declared}")
    if (
        observed["rows"] != int(data_config["expected_full_rows"])
        or observed["positives"] != int(data_config["expected_full_positives"])
        or observed["folds"] != expected_folds
    ):
        raise RuntimeError("V170 CE organizer-train inventory differs from frozen method")
    rows = rows.sort_values("id_key", kind="stable").reset_index(drop=True)
    binding = {
        "schema_version": "v170.ce_binding.1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256_file(rows_path),
        **observed,
        "contains_external_eval_data": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return rows, binding


def _by_id(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.set_index("id_key", drop=False)


def _validate_pair_common(
    record: Mapping[str, Any],
    *,
    line_number: int,
    kind: str,
) -> None:
    if record.get("review_status") != "ACCEPT":
        raise RuntimeError(f"V170 {kind} line {line_number} is not accepted")
    if not str(record.get("pair_id", "")).strip():
        raise RuntimeError(f"V170 {kind} line {line_number} has no pair_id")
    forbidden = sorted(
        key
        for key in record
        if str(key).lower() in {
            "external_eval_label",
            "private_label",
            "hidden_label",
            "external_eval_label",
            "external_eval_score",
        }
    )
    if forbidden:
        raise RuntimeError(f"V170 {kind} line {line_number} has forbidden keys {forbidden}")


def _real_pair_frame(
    records: Sequence[Mapping[str, Any]],
    rows: pd.DataFrame,
    *,
    kind: str,
    require_same_fold: bool,
    allowed_endpoint_folds: set[int],
) -> pd.DataFrame:
    by_id = _by_id(rows)
    output: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    component_relations: set[tuple[str, str]] = set()
    for line_number, record in enumerate(records, 1):
        _validate_pair_common(record, line_number=line_number, kind=kind)
        pair_id = str(record["pair_id"])
        if pair_id in pair_ids:
            raise RuntimeError(f"V170 {kind} duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        positive_id = _id_key(record.get("positive_id"))
        negative_id = _id_key(record.get("negative_id"))
        if positive_id not in by_id.index or negative_id not in by_id.index:
            raise RuntimeError(f"V170 {kind} endpoint is outside organizer train")
        positive = by_id.loc[positive_id]
        negative = by_id.loc[negative_id]
        if isinstance(positive, pd.DataFrame) or isinstance(negative, pd.DataFrame):
            raise RuntimeError(f"V170 {kind} endpoint ID is ambiguous")
        if int(positive.label) != 1 or int(negative.label) != 0:
            raise RuntimeError(f"V170 {kind} endpoint label direction is not 1>0")
        if str(positive.component_key) == str(negative.component_key):
            raise RuntimeError(f"V170 {kind} endpoints share one duplicate component")
        if require_same_fold and int(positive.fold) != int(negative.fold):
            raise RuntimeError(f"V170 {kind} endpoints must be in the same fold")
        if (
            int(positive.fold) not in allowed_endpoint_folds
            or int(negative.fold) not in allowed_endpoint_folds
        ):
            raise RuntimeError(
                f"V170 {kind} endpoint touches the frozen forward-gate fold"
            )
        relation = (str(positive.component_key), str(negative.component_key))
        if relation in component_relations:
            raise RuntimeError(f"V170 {kind} duplicates a component relation")
        component_relations.add(relation)
        eligible = sorted(
            set(int(value) for value in rows.fold.unique())
            - {int(positive.fold), int(negative.fold)}
        )
        declared_eligible = sorted(int(value) for value in record.get("eligible_query_folds", []))
        if declared_eligible != eligible:
            raise RuntimeError(
                f"V170 {kind} eligible_query_folds drift for pair {pair_id}: "
                f"{declared_eligible} != {eligible}"
            )
        output.append(
            {
                "pair_id": pair_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "positive_component_key": str(positive.component_key),
                "negative_component_key": str(negative.component_key),
                "positive_name": str(positive.clean_name),
                "positive_description": str(positive.clean_description),
                "negative_name": str(negative.clean_name),
                "negative_description": str(negative.clean_description),
                "boundary_code": str(record.get("boundary_code", "")).strip(),
                "eligible_query_folds": eligible,
                "mass_key": str(record.get("mass_key", pair_id)).strip(),
            }
        )
    frame = pd.DataFrame(output)
    if frame.empty or frame.boundary_code.eq("").any() or frame.mass_key.eq("").any():
        raise RuntimeError(f"V170 {kind} contains empty required fields")
    return frame


def _r7_pair_frame(
    records: Sequence[Mapping[str, Any]],
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve query-specific R7 ID relations against organizer train.

    R7 contains 100 independently accepted query-specific instances. The same
    endpoint relation can occur for more than one development query and is
    identified by one shared ``mass_key``. Endpoint folds in the supplied CE
    file are not used as R7 authority; ``query_fold`` is part of the bundled
    annotation and was fixed when the relation was produced.
    """

    by_id = _by_id(rows)
    output: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    query_mass_keys: set[tuple[int, str]] = set()
    mass_endpoints: dict[str, tuple[str, str]] = {}
    for line_number, record in enumerate(records, 1):
        _validate_pair_common(record, line_number=line_number, kind="R7")
        pair_id = str(record["pair_id"])
        if pair_id in pair_ids:
            raise RuntimeError(f"V170 R7 duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        try:
            query_fold = int(record["query_fold"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"V170 R7 line {line_number} has no valid query_fold") from error
        if query_fold not in range(4):
            raise RuntimeError(f"V170 R7 line {line_number} query_fold must be 0..3")
        declared_eligible = sorted(
            int(value) for value in record.get("eligible_query_folds", [])
        )
        if declared_eligible != [query_fold]:
            raise RuntimeError(
                f"V170 R7 eligible_query_folds must contain only its source query: "
                f"{pair_id}"
            )
        positive_id = _id_key(record.get("positive_id"))
        negative_id = _id_key(record.get("negative_id"))
        if positive_id not in by_id.index or negative_id not in by_id.index:
            raise RuntimeError("V170 R7 endpoint is outside organizer train")
        positive = by_id.loc[positive_id]
        negative = by_id.loc[negative_id]
        if isinstance(positive, pd.DataFrame) or isinstance(negative, pd.DataFrame):
            raise RuntimeError("V170 R7 endpoint ID is ambiguous")
        if int(positive.label) != 1 or int(negative.label) != 0:
            raise RuntimeError("V170 R7 endpoint label direction is not 1>0")
        if str(positive.component_key) == str(negative.component_key):
            raise RuntimeError("V170 R7 endpoints share one duplicate component")
        mass_key = str(record.get("mass_key", "")).strip()
        boundary_code = str(record.get("boundary_code", "")).strip()
        if not mass_key or not boundary_code:
            raise RuntimeError("V170 R7 contains empty required fields")
        query_mass_key = (query_fold, mass_key)
        if query_mass_key in query_mass_keys:
            raise RuntimeError("V170 R7 duplicates a mass key within one query fold")
        query_mass_keys.add(query_mass_key)
        endpoints = (positive_id, negative_id)
        if mass_key in mass_endpoints and mass_endpoints[mass_key] != endpoints:
            raise RuntimeError("V170 R7 mass key maps to conflicting endpoints")
        mass_endpoints[mass_key] = endpoints
        output.append(
            {
                "pair_id": pair_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "positive_component_key": str(positive.component_key),
                "negative_component_key": str(negative.component_key),
                "positive_name": str(positive.clean_name),
                "positive_description": str(positive.clean_description),
                "negative_name": str(negative.clean_name),
                "negative_description": str(negative.clean_description),
                "boundary_code": boundary_code,
                "eligible_query_folds": [query_fold],
                "source_query_fold": query_fold,
                "mass_key": mass_key,
            }
        )
    frame = pd.DataFrame(output)
    if frame.empty:
        raise RuntimeError("V170 R7 contains no accepted relation instances")
    return frame


def load_r4_pairs(
    manifest_path: Path,
    rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _validate_manifest(manifest_path, schema=R4_SCHEMA, kind="R4")
    path = resolve_bound_file(
        manifest_path,
        manifest.get("pairs_file", {}),
        context="V170 R4 pairs",
    )
    allowed = {int(value) for value in config["data"]["supplementary_development_folds"]}
    frame = _real_pair_frame(
        read_jsonl(path),
        rows,
        kind="R4",
        require_same_fold=True,
        allowed_endpoint_folds=allowed,
    )
    observed = {
        "pairs": len(frame),
        "positive_components": int(frame.positive_component_key.nunique()),
        "negative_components": int(frame.negative_component_key.nunique()),
        "boundaries": int(frame.boundary_code.nunique()),
    }
    if observed != {
        key: int(manifest.get(key, -1)) for key in observed
    }:
        raise RuntimeError("V170 R4 manifest inventory drift")
    if len(frame) != int(config["data"]["expected_r4_pairs"]):
        raise RuntimeError("V170 R4 pair count differs from frozen method")
    binding = {
        "schema_version": "v170.r4_binding.1",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "pairs_sha256": sha256_file(path),
        **observed,
        "pairwise_only": True,
        "ce_rows": 0,
        "contains_external_eval_data": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return frame, binding


def load_r5_pairs(
    manifest_path: Path,
    rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _validate_manifest(
        manifest_path,
        schema=R5_SCHEMA,
        kind="R5",
        allowed_source_partitions=(SYNTHETIC_SOURCE_PARTITION,),
    )
    pairs_path = resolve_bound_file(
        manifest_path,
        manifest.get("pairs_file", {}),
        context="V170 R5 pairs",
    )
    reviews_path = resolve_bound_file(
        manifest_path,
        manifest.get("reviews_file", {}),
        context="V170 R5 blind reviews",
    )
    structural_path = resolve_bound_file(
        manifest_path,
        manifest.get("structural_audit_file", {}),
        context="V170 R5 structural audit",
    )
    cards_path = resolve_bound_file(
        manifest_path,
        manifest.get("training_cards_file", {}),
        context="V170 R5 training cards",
    )
    metrics_path = resolve_bound_file(
        manifest_path,
        manifest.get("metrics_file", {}),
        context="V170 R5 metrics",
    )
    artifact_manifest_path = resolve_bound_file(
        manifest_path,
        manifest.get("artifact_manifest_file", {}),
        context="V170 R5 artifact manifest",
    )
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    expected_artifact_files = {
        "rendered_pairs.jsonl": pairs_path,
        "blind_reviews.jsonl": reviews_path,
        "structural_audit.jsonl": structural_path,
        "training_cards.jsonl": cards_path,
        "metrics.json": metrics_path,
    }
    if (
        not isinstance(artifact_manifest, dict)
        or artifact_manifest.get("schema_version") != "v170.r5.production.artifacts.1"
        or artifact_manifest.get("verdict") != "PASS"
    ):
        raise RuntimeError("V170 R5 production artifact manifest did not pass")
    artifact_files = artifact_manifest.get("files")
    if not isinstance(artifact_files, dict):
        raise RuntimeError("V170 R5 artifact manifest has no file inventory")
    for filename, path in expected_artifact_files.items():
        expected = artifact_files.get(filename)
        observed = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if expected != observed or path.name != filename:
            raise RuntimeError(f"V170 R5 artifact binding drift: {filename}")

    records = read_jsonl(pairs_path)
    reviews = read_jsonl(reviews_path)
    structural = read_jsonl(structural_path)
    cards = read_jsonl(cards_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    folds = sorted(rows.fold.unique().tolist())
    organizer_surfaces = {
        _normalize_surface(row.clean_name, row.clean_description)
        for row in rows.itertuples(index=False)
    }
    organizer_ngrams: set[tuple[str, ...]] = set()
    for surface in organizer_surfaces:
        organizer_ngrams.update(_ngrams(surface, 13))
    reviews_by_id = {str(record.get("pair_id", "")): record for record in reviews}
    structural_by_id = {str(record.get("pair_id", "")): record for record in structural}
    cards_by_id: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        cards_by_id.setdefault(str(card.get("pair_id", "")), []).append(card)
    if (
        len(reviews_by_id) != len(reviews)
        or len(structural_by_id) != len(structural)
        or any(not pair_id for pair_id in reviews_by_id | structural_by_id | cards_by_id)
    ):
        raise RuntimeError("V170 R5 review/audit/card IDs are missing or duplicated")

    output: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    card_surfaces: set[str] = set()
    direction_correct = 0
    for line_number, record in enumerate(records, 1):
        pair_id = str(record.get("pair_id", "")).strip()
        if not pair_id or pair_id in pair_ids:
            raise RuntimeError(f"V170 R5 duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if any(
            key in record
            for key in (
                "positive_id",
                "negative_id",
                "label",
                "source_id",
                "source_component",
            )
        ):
            raise RuntimeError("V170 R5 synthetic records cannot carry organizer IDs/labels")
        positive = record.get("positive")
        negative = record.get("negative")
        if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
            raise RuntimeError(f"V170 R5 line {line_number} has no positive/negative side")
        values = {
            "positive_name": str(positive.get("name", "")).strip(),
            "positive_description": str(positive.get("description", "")).strip(),
            "negative_name": str(negative.get("name", "")).strip(),
            "negative_description": str(negative.get("description", "")).strip(),
            "positive_decisive_span": str(positive.get("decisive_span", "")).strip(),
            "negative_decisive_span": str(negative.get("decisive_span", "")).strip(),
            "boundary_code": str(record.get("boundary", "")).strip(),
            "style_frame": str(record.get("style_frame_id", "")).strip(),
            "changed_slot": str(record.get("changed_slot", "")).strip(),
            "changed_field": str(record.get("changed_field", "")).strip(),
            "invariant_fact_frame": str(record.get("invariant_fact_frame", "")).strip(),
        }
        if any(not value for value in values.values()):
            raise RuntimeError(f"V170 R5 line {line_number} has an empty surface/stratum")
        if values["changed_field"] not in {"name", "description"}:
            raise RuntimeError(f"V170 R5 line {line_number} has invalid changed_field")
        if values["positive_decisive_span"] not in (
            values["positive_name"] + "\n" + values["positive_description"]
        ) or values["negative_decisive_span"] not in (
            values["negative_name"] + "\n" + values["negative_description"]
        ):
            raise RuntimeError(f"V170 R5 line {line_number} decisive span is not bound")
        positive_surface = _normalize_surface(values["positive_name"], values["positive_description"])
        negative_surface = _normalize_surface(values["negative_name"], values["negative_description"])
        if positive_surface == negative_surface:
            raise RuntimeError(f"V170 R5 line {line_number} has identical endpoints")
        if positive_surface in card_surfaces or negative_surface in card_surfaces:
            raise RuntimeError(f"V170 R5 line {line_number} duplicates a synthetic card")
        card_surfaces.update((positive_surface, negative_surface))
        if positive_surface in organizer_surfaces or negative_surface in organizer_surfaces:
            raise RuntimeError(f"V170 R5 line {line_number} exactly copies organizer train")
        for surface in (positive_surface, negative_surface):
            if _ngrams(surface, 13) & organizer_ngrams:
                raise RuntimeError(
                    f"V170 R5 line {line_number} copies a 13-token organizer-train span"
                )

        audit = structural_by_id.get(pair_id)
        checks = audit.get("checks") if isinstance(audit, Mapping) else None
        if (
            not isinstance(audit, Mapping)
            or audit.get("pass") is not True
            or audit.get("boundary") != values["boundary_code"]
            or audit.get("style_frame_id") != values["style_frame"]
            or not isinstance(checks, Mapping)
            or not checks
            or any(value is not True for value in checks.values())
        ):
            raise RuntimeError(f"V170 R5 structural audit failed for {pair_id}")

        review = reviews_by_id.get(pair_id)
        if not isinstance(review, Mapping):
            raise RuntimeError(f"V170 R5 has no blind review for {pair_id}")
        label_a = int(review.get("label_a", -1))
        label_b = int(review.get("label_b", -1))
        evidence_a = str(review.get("evidence_a", ""))
        evidence_b = str(review.get("evidence_b", ""))
        positive_span = values["positive_decisive_span"]
        negative_span = values["negative_decisive_span"]
        blind_swap = int(sha256_text(pair_id), 16) % 2 == 1
        if blind_swap:
            expected_labels = (0, 1)
            expected_surfaces = (negative_surface, positive_surface)
        else:
            expected_labels = (1, 0)
            expected_surfaces = (positive_surface, negative_surface)
        normalized_evidence = (
            _normalize_surface(evidence_a, ""),
            _normalize_surface(evidence_b, ""),
        )
        review_direction_correct = (
            (label_a, label_b) == expected_labels
            and all(normalized_evidence)
            and all(
                evidence in surface
                for evidence, surface in zip(normalized_evidence, expected_surfaces)
            )
        )
        if not review_direction_correct:
            raise RuntimeError(f"V170 R5 blind direction failed for {pair_id}")
        direction_correct += 1
        usable = review.get("pair_usable")
        reject_reason = str(review.get("reject_reason", "")).strip()
        if not isinstance(usable, bool) or (
            usable and reject_reason != "NONE"
        ) or (not usable and reject_reason in {"", "NONE"}):
            raise RuntimeError(f"V170 R5 blind usability evidence drift for {pair_id}")

        pair_cards = cards_by_id.get(pair_id)
        if not isinstance(pair_cards, list) or len(pair_cards) != 2:
            raise RuntimeError(f"V170 R5 training-card roster drift for {pair_id}")
        card_by_label: dict[int, Mapping[str, Any]] = {}
        for card in pair_cards:
            label = int(card.get("organizer_label", -1))
            if (
                label not in {0, 1}
                or label in card_by_label
                or card.get("fully_synthetic") is not True
                or card.get("source_id") is not None
                or card.get("source_component") is not None
            ):
                raise RuntimeError(f"V170 R5 training card provenance drift for {pair_id}")
            card_by_label[label] = card
        expected_cards = {1: positive, 0: negative}
        for label, side in expected_cards.items():
            card = card_by_label.get(label)
            if not isinstance(card, Mapping) or any(
                str(card.get(key, "")) != str(side.get(key, ""))
                for key in ("name", "description", "decisive_span")
            ):
                raise RuntimeError(f"V170 R5 rendered/training card drift for {pair_id}")
        positive_first = record.get("positive_first")
        if not isinstance(positive_first, bool) or any(
            int(card["organizer_label"])
            != (1 if (int(card.get("pair_position", -1)) == 0) == positive_first else 0)
            for card in pair_cards
        ):
            raise RuntimeError(f"V170 R5 pair-order balance drift for {pair_id}")

        output.append(
            {
                "pair_id": pair_id,
                **values,
                "sampling_frame": f"{values['boundary_code']}::{values['style_frame']}",
                "source_component_keys": [],
                "eligible_query_folds": folds,
                "reviewer_blind_direction_correct": True,
                "usable": usable,
                "reviewer_flag": not usable,
                "reviewer_reject_reason": reject_reason,
                "reviewer_same_product_frame": bool(review.get("same_product_frame")),
                "reviewer_single_causal_boundary": bool(
                    review.get("single_causal_boundary")
                ),
            }
        )
    frame = pd.DataFrame(output)
    if set(pair_ids) != set(reviews_by_id) or set(pair_ids) != set(structural_by_id) or set(
        pair_ids
    ) != set(cards_by_id):
        raise RuntimeError("V170 R5 pair/review/audit/card roster mismatch")
    data_config = config["data"]
    frame_counts = Counter(frame.sampling_frame.astype(str))
    reviewer_flagged = int(frame.reviewer_flag.sum())
    blind_usable = len(frame) - reviewer_flagged
    structural_passed = sum(bool(record.get("pass")) for record in structural)
    observed = {
        "pairs": len(frame),
        "boundaries": int(frame.boundary_code.nunique()),
        "style_frames": int(frame.sampling_frame.nunique()),
        "max_pairs_per_style_frame": max(frame_counts.values(), default=0),
        "structural_passed": structural_passed,
        "blind_direction_correct": direction_correct,
        "blind_usable": blind_usable,
        "reviewer_flagged_pairs": reviewer_flagged,
    }
    if observed != {key: int(manifest.get(key, -1)) for key in observed}:
        raise RuntimeError("V170 R5 manifest inventory drift")
    structural_rate = structural_passed / len(frame)
    direction_rate = direction_correct / len(frame)
    blind_usable_rate = blind_usable / len(frame)
    population = metrics.get("population_audit", {})
    checks = metrics.get("checks", {})
    preregistration_checks = metrics.get("preregistration_checks", {})
    if (
        len(frame) != int(data_config["expected_r5_pairs"])
        or observed["style_frames"] < int(data_config["minimum_r5_style_frames"])
        or observed["max_pairs_per_style_frame"]
        > int(data_config["maximum_r5_pairs_per_style_frame"])
        or structural_rate < float(data_config["minimum_r5_structural_pass_rate"])
        or direction_rate < float(data_config["minimum_r5_direction_accuracy"])
        or blind_usable_rate < float(data_config["minimum_r5_blind_usable_rate"])
        or metrics.get("schema_version") != "v170.r5.production.metrics.1"
        or metrics.get("verdict") != "PASS"
        or int(metrics.get("pairs", -1)) != len(frame)
        or int(metrics.get("cards", -1)) != 2 * len(frame)
        or int(metrics.get("blind_direction_correct", -1)) != direction_correct
        or float(metrics.get("blind_direction_accuracy", -1.0)) != direction_rate
        or int(metrics.get("blind_usable", -1)) != blind_usable
        or float(metrics.get("blind_usable_rate", -1.0)) != blind_usable_rate
        or manifest.get("reviewer_flags_preserved") is not True
        or int(manifest.get("training_pairs_including_reviewer_flags", -1))
        != len(frame)
        or float(manifest.get("aggregate_blind_usable_gate", -1.0))
        != float(data_config["minimum_r5_blind_usable_rate"])
        or int(metrics.get("structural", {}).get("passed", -1)) != structural_passed
        or float(metrics.get("structural", {}).get("pass_rate", -1.0))
        != structural_rate
        or int(metrics.get("exact_train_overlap", -1)) != 0
        or int(metrics.get("copied_13_token_train_spans", -1)) != 0
        or int(population.get("exact_duplicate_cards", -1)) != 0
        or int(population.get("exact_duplicate_masked_pairs", -1)) != 0
        or int(population.get("cross_frame_near_duplicate_edges_ge_0_94", -1)) != 0
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(preregistration_checks, Mapping)
        or not preregistration_checks
        or any(value is not True for value in preregistration_checks.values())
    ):
        raise RuntimeError("V170 R5 aggregate production gate failed")
    flagged_ids = sorted(frame.loc[frame.reviewer_flag, "pair_id"].astype(str))
    flag_reasons = dict(
        sorted(Counter(frame.loc[frame.reviewer_flag, "reviewer_reject_reason"]).items())
    )
    binding = {
        "schema_version": "v170.r5_binding.2",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "pairs_sha256": sha256_file(pairs_path),
        "reviews_sha256": sha256_file(reviews_path),
        "structural_audit_sha256": sha256_file(structural_path),
        "training_cards_sha256": sha256_file(cards_path),
        "metrics_sha256": sha256_file(metrics_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        **observed,
        "structural_pass_rate": structural_rate,
        "blind_direction_accuracy": direction_rate,
        "blind_usable_rate": blind_usable_rate,
        "minimum_blind_usable_rate": float(
            data_config["minimum_r5_blind_usable_rate"]
        ),
        "reviewer_flagged_pair_ids_sha256": canonical_sha256(flagged_ids),
        "reviewer_flag_reason_counts": flag_reasons,
        "reviewer_flags_preserved_in_training_frame": True,
        "aggregate_gate_policy": "all_200_pairs_train_when_corpus_blind_usable_rate_ge_0.95",
        "training_pairs_including_reviewer_flags": len(frame),
        "pairwise_only": True,
        "ce_rows": 0,
        "exact_train_overlap": 0,
        "train_13_token_overlap": 0,
        "contains_external_eval_data": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return frame, binding


def _attach_r7_mass(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.mass_key.duplicated().any():
        raise RuntimeError("V170 selected R7 view requires one row per mass_key")
    positive_reuse = Counter(frame.positive_component_key.astype(str))
    negative_reuse = Counter(frame.negative_component_key.astype(str))
    output = frame.copy()
    output["raw_fullfit_mass"] = [
        min(1.0 / positive_reuse[str(positive)], 1.0 / negative_reuse[str(negative)])
        for positive, negative in zip(
            output.positive_component_key,
            output.negative_component_key,
            strict=True,
        )
    ]
    output["normalized_pair_weight"] = (
        output.raw_fullfit_mass / float(output.raw_fullfit_mass.sum())
    )
    if (
        abs(float(output.normalized_pair_weight.sum()) - 1.0) > 1e-12
        or output.groupby("positive_component_key").raw_fullfit_mass.sum().max() > 1.0 + 1e-12
        or output.groupby("negative_component_key").raw_fullfit_mass.sum().max() > 1.0 + 1e-12
    ):
        raise RuntimeError("V170 R7 component mass cap failed")
    return output


def load_r7_pairs(
    manifest_path: Path,
    rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _validate_manifest(manifest_path, schema=R7_SCHEMA, kind="R7")
    path = resolve_bound_file(
        manifest_path,
        manifest.get("pairs_file", {}),
        context="V170 R7 pairs",
    )
    frame = _r7_pair_frame(read_jsonl(path), rows)
    observed = {
        "pairs": len(frame),
        "mass_keys": int(frame.mass_key.nunique()),
        "positive_components": int(frame.positive_component_key.nunique()),
        "negative_components": int(frame.negative_component_key.nunique()),
        "boundaries": int(frame.boundary_code.nunique()),
        "query_counts": {
            str(key): int(value)
            for key, value in frame.source_query_fold.value_counts().sort_index().items()
        },
    }
    declared = {
        **{
            key: int(manifest.get(key, -1))
            for key in (
                "pairs",
                "mass_keys",
                "positive_components",
                "negative_components",
                "boundaries",
            )
        },
        "query_counts": {
            str(key): int(value)
            for key, value in manifest.get("query_counts", {}).items()
        },
    }
    if observed != declared:
        raise RuntimeError("V170 R7 manifest inventory drift")
    data_config = config["data"]
    if (
        len(frame) != int(data_config["expected_r7_relation_instances"])
        or frame.mass_key.nunique() != int(data_config["expected_r7_mass_keys"])
        or observed["query_counts"]
        != {
            str(key): int(value)
            for key, value in data_config["expected_r7_query_counts"].items()
        }
    ):
        raise RuntimeError("V170 R7 relation inventory differs from the bundled contract")
    binding = {
        "schema_version": "v170.r7_binding.1",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "pairs_sha256": sha256_file(path),
        **observed,
        "mass_policy": "query_specific_instances_and_deduplicated_fullfit_mass_keys",
        "pairwise_only": True,
        "ce_rows": 0,
        "contains_external_eval_data": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return frame, binding


def select_for_query_fold(frame: pd.DataFrame, query_fold: int | None) -> pd.DataFrame:
    if query_fold is None:
        return frame.copy().reset_index(drop=True)
    selected = frame.loc[
        frame.eligible_query_folds.map(lambda values: int(query_fold) in values)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"V170 pair inventory has no rows eligible for q{query_fold}")
    return selected.reset_index(drop=True)


def select_r7_for_query_fold(
    frame: pd.DataFrame, query_fold: int | None
) -> pd.DataFrame:
    """Select R7 query instances or the 62-key q4/full-fit view."""

    if query_fold is None or int(query_fold) == 4:
        selected = (
            frame.sort_values(
                ["mass_key", "source_query_fold", "pair_id"], kind="stable"
            )
            .drop_duplicates("mass_key", keep="first")
            .reset_index(drop=True)
        )
    elif int(query_fold) in range(4):
        selected = frame.loc[
            frame.source_query_fold.astype(int).eq(int(query_fold))
        ].copy().reset_index(drop=True)
    else:
        raise RuntimeError(f"V170 unknown R7 query fold: {query_fold}")
    if selected.empty:
        raise RuntimeError(f"V170 R7 inventory has no rows eligible for q{query_fold}")
    return _attach_r7_mass(selected)


def split_ce_rows(
    rows: pd.DataFrame,
    *,
    query_fold: int | None,
    development_folds: Sequence[int] = (0, 1, 2, 3),
    forward_gate_fold: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = {int(value) for value in development_folds}
    forward = int(forward_gate_fold)
    if (
        not development
        or forward in development
        or set(rows.fold.astype(int).unique()) != development | {forward}
    ):
        raise RuntimeError("V170 CE development/forward-fold topology drift")
    if query_fold is None:
        train = rows.copy()
        query = rows.iloc[0:0].copy()
    elif int(query_fold) == forward:
        train = rows.loc[rows.fold.astype(int).isin(development)].copy()
        query = rows.loc[rows.fold.astype(int).eq(forward)].copy()
    elif int(query_fold) in development:
        train_folds = development - {int(query_fold)}
        train = rows.loc[rows.fold.astype(int).isin(train_folds)].copy()
        query = rows.loc[rows.fold.astype(int).eq(int(query_fold))].copy()
    else:
        raise RuntimeError(f"V170 unknown query fold: {query_fold}")
    if train.empty or (query_fold is not None and query.empty):
        raise RuntimeError("V170 CE train/query split is empty")
    if set(train.component_key) & set(query.component_key):
        raise RuntimeError("V170 CE duplicate component crosses train/query")
    return train.reset_index(drop=True), query.reset_index(drop=True)
