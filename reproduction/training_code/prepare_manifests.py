#!/usr/bin/env python3
"""Create V170 self-hashed data, base-snapshot, and execution manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .contracts import (
    SYNTHETIC_SOURCE_PARTITION,
    bind_self_hash,
    canonical_sha256,
    file_reference,
    load_config,
    package_versions,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from .data_contracts import (
    CE_SCHEMA,
    R4_SCHEMA,
    R5_SCHEMA,
    R7_SCHEMA,
    REQUIRED_CE_COLUMNS,
    _real_pair_frame,
    _r7_pair_frame,
    load_ce_dataset,
    load_r4_pairs,
    load_r5_pairs,
    load_r7_pairs,
)


TRAIN_ONLY_DECLARATIONS = {
    "source_partition": "organizer_train",
    "contains_external_eval_rows": False,
    "contains_external_eval_labels": False,
    "contains_external_eval_predictions": False,
    "contains_private_rows": False,
    "contains_private_labels": False,
    "contains_hidden_labels": False,
    "external_eval_feedback_used": False,
}
SYNTHETIC_TRAIN_ONLY_DECLARATIONS = {
    **TRAIN_ONLY_DECLARATIONS,
    "source_partition": SYNTHETIC_SOURCE_PARTITION,
}


def _require_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Input file must live below manifest directory: {path}") from error


def _write_once(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    payload = bind_self_hash(value)
    write_json_atomic(path, payload)
    return payload


def _write_after_validation(
    path: Path,
    value: dict[str, Any],
    validator: Any,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    payload = bind_self_hash(value)
    temporary = path.with_name(f".{path.name}.validation.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    write_json_atomic(temporary, payload)
    try:
        validator(temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def build_ce(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    rows_path = args.rows.resolve()
    output = args.output.resolve()
    _require_under(rows_path, output.parent)
    rows = pd.read_parquet(rows_path)
    if not set(REQUIRED_CE_COLUMNS).issubset(rows.columns):
        raise RuntimeError("V170 CE input is missing required columns")
    payload_value = {
        "schema_version": CE_SCHEMA,
        "status": "PASS",
        **TRAIN_ONLY_DECLARATIONS,
        "rows_file": file_reference(rows_path, relative_to=output.parent),
        "rows": len(rows),
        "positives": int(rows.label.astype(int).sum()),
        "folds": sorted(rows.fold.astype(int).unique().tolist()),
        "components": int(rows.component_key.astype(str).nunique()),
        "columns": list(rows.columns),
        "organizer_train_labels_used": True,
    }
    payload = _write_after_validation(
        output,
        payload_value,
        lambda temporary: load_ce_dataset(temporary, config),
    )
    return payload


def build_pairs(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    rows, _ = load_ce_dataset(args.ce_manifest.resolve(), config)
    pairs_path = args.pairs.resolve()
    output = args.output.resolve()
    _require_under(pairs_path, output.parent)
    records = read_jsonl(pairs_path)
    reference = file_reference(pairs_path, relative_to=output.parent)
    if args.kind == "r4":
        allowed = {
            int(value)
            for value in config["data"]["supplementary_development_folds"]
        }
        frame = _real_pair_frame(
            records,
            rows,
            kind="R4",
            require_same_fold=True,
            allowed_endpoint_folds=allowed,
        )
        schema = R4_SCHEMA
        inventory = {
            "pairs": len(frame),
            "positive_components": int(frame.positive_component_key.nunique()),
            "negative_components": int(frame.negative_component_key.nunique()),
            "boundaries": int(frame.boundary_code.nunique()),
        }
        loader = load_r4_pairs
    else:
        frame = _r7_pair_frame(records, rows)
        schema = R7_SCHEMA
        inventory = {
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
        loader = load_r7_pairs
    payload = _write_after_validation(
        output,
        {
            "schema_version": schema,
            "status": "PASS",
            **TRAIN_ONLY_DECLARATIONS,
            "pairs_file": reference,
            **inventory,
            "pairwise_only": True,
            "ce_rows": 0,
        },
        lambda temporary: loader(temporary, rows, config),
    )
    return payload


def build_r5_production(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    rows, _ = load_ce_dataset(args.ce_manifest.resolve(), config)
    root = args.artifact_dir.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        "pairs_file": root / "rendered_pairs.jsonl",
        "reviews_file": root / "blind_reviews.jsonl",
        "structural_audit_file": root / "structural_audit.jsonl",
        "training_cards_file": root / "training_cards.jsonl",
        "metrics_file": root / "metrics.json",
        "artifact_manifest_file": root / "artifact_manifest.json",
    }
    for path in paths.values():
        _require_under(path, output.parent)
        if not path.is_file():
            raise FileNotFoundError(path)
    pairs = read_jsonl(paths["pairs_file"])
    reviews = read_jsonl(paths["reviews_file"])
    structural = read_jsonl(paths["structural_audit_file"])
    metrics = json.loads(paths["metrics_file"].read_text(encoding="utf-8"))
    frame = pd.DataFrame(pairs)
    if frame.empty or not {"pair_id", "boundary", "style_frame_id"}.issubset(
        frame.columns
    ):
        raise RuntimeError("V170 R5 production pair inventory is invalid")
    sampling = frame.boundary.astype(str) + "::" + frame.style_frame_id.astype(str)
    reviewer_flagged = sum(record.get("pair_usable") is False for record in reviews)
    payload = _write_after_validation(
        output,
        {
            "schema_version": R5_SCHEMA,
            "status": "PASS",
            **SYNTHETIC_TRAIN_ONLY_DECLARATIONS,
            **{
                key: file_reference(path, relative_to=output.parent)
                for key, path in paths.items()
            },
            "pairs": len(frame),
            "boundaries": int(frame.boundary.astype(str).nunique()),
            "style_frames": int(sampling.nunique()),
            "max_pairs_per_style_frame": int(sampling.value_counts().max()),
            "structural_passed": sum(
                record.get("pass") is True for record in structural
            ),
            "blind_direction_correct": int(
                metrics.get("blind_direction_correct", -1)
            ),
            "blind_usable": sum(
                record.get("pair_usable") is True for record in reviews
            ),
            "reviewer_flagged_pairs": reviewer_flagged,
            "reviewer_flags_preserved": True,
            "training_pairs_including_reviewer_flags": len(frame),
            "aggregate_blind_usable_gate": float(
                config["data"]["minimum_r5_blind_usable_rate"]
            ),
            "pairwise_only": True,
            "ce_rows": 0,
        },
        lambda temporary: load_r5_pairs(temporary, rows, config),
    )
    return payload


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    root = args.model_dir.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    excluded_parts = {".cache", ".git", "__pycache__"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or set(path.relative_to(root).parts) & excluded_parts:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files or not any(item["path"].endswith(".safetensors") for item in files):
        raise RuntimeError("V170 base snapshot has no model tensor files")
    return _write_once(
        output,
        {
            "schema_version": "v170.base_snapshot.1",
            "status": "PASS",
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "files": files,
            "files_sha256": canonical_sha256(files),
        },
    )


def build_execution_receipt(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = load_config(args.config.resolve())
    output = args.output.resolve()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V170 execution receipt requires exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    expected = str(config["container"]["hardware_class"])
    if expected.lower() not in name.lower():
        raise RuntimeError(f"V170 expected {expected}, observed {name}")
    properties = torch.cuda.get_device_properties(0)
    return _write_once(
        output,
        {
            "schema_version": "v170.execution_receipt.1",
            "status": "PASS",
            "hardware_class": expected,
            "cuda_device_name": name,
            "cuda_total_memory_bytes": int(properties.total_memory),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda_version": torch.version.cuda,
            "container_immutable_reference": config["container"]["immutable_reference"],
            "network_disabled_during_fit": bool(args.network_disabled_during_fit),
            "packages": package_versions(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ce = sub.add_parser("ce")
    ce.add_argument("--config", type=Path, required=True)
    ce.add_argument("--rows", type=Path, required=True)
    ce.add_argument("--output", type=Path, required=True)
    pairs = sub.add_parser("pairs")
    pairs.add_argument("--kind", choices=("r4", "r7"), required=True)
    pairs.add_argument("--config", type=Path, required=True)
    pairs.add_argument("--ce-manifest", type=Path, required=True)
    pairs.add_argument("--pairs", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True)
    r5 = sub.add_parser("r5-production")
    r5.add_argument("--config", type=Path, required=True)
    r5.add_argument("--ce-manifest", type=Path, required=True)
    r5.add_argument("--artifact-dir", type=Path, required=True)
    r5.add_argument("--output", type=Path, required=True)
    snapshot = sub.add_parser("base-snapshot")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--model-dir", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    receipt = sub.add_parser("execution-receipt")
    receipt.add_argument("--config", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)
    receipt.add_argument("--network-disabled-during-fit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "ce":
        result = build_ce(args)
    elif args.command == "pairs":
        result = build_pairs(args)
    elif args.command == "r5-production":
        result = build_r5_production(args)
    elif args.command == "base-snapshot":
        result = build_snapshot(args)
    else:
        result = build_execution_receipt(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
