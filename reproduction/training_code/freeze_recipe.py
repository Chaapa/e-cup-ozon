#!/usr/bin/env python3
"""Freeze regenerated R4/R5/R7 and training recipe before OOF/q4 scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .contracts import bind_self_hash, load_config, sha256_file, write_json_atomic
from .data_contracts import load_ce_dataset, load_r4_pairs, load_r5_pairs, load_r7_pairs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--r5-manifest", type=Path, required=True)
    parser.add_argument("--r7-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def freeze(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("V170 recipe precommit must be written once")
    config = load_config(args.config.resolve())
    rows, ce = load_ce_dataset(args.ce_manifest.resolve(), config)
    r4, r4_binding = load_r4_pairs(args.r4_manifest.resolve(), rows, config)
    r5, r5_binding = load_r5_pairs(args.r5_manifest.resolve(), rows, config)
    r7, r7_binding = load_r7_pairs(args.r7_manifest.resolve(), rows, config)
    payload = bind_self_hash(
        {
            "schema_version": "v170.recipe_precommit.1",
            "status": "FROZEN_BEFORE_OOF_AND_Q4_SCORE",
            "experiment_id": config["experiment_id"],
            "files": {
                "config": sha256_file(args.config.resolve()),
                "ce_manifest": sha256_file(args.ce_manifest.resolve()),
                "r4_manifest": sha256_file(args.r4_manifest.resolve()),
                "r5_manifest": sha256_file(args.r5_manifest.resolve()),
                "r7_manifest": sha256_file(args.r7_manifest.resolve()),
            },
            "bindings": {
                "ce": ce,
                "r4": r4_binding,
                "r5": r5_binding,
                "r7": r7_binding,
            },
            "inventories": {
                "ce_rows": len(rows),
                "r4_pairs": len(r4),
                "r5_pairs": len(r5),
                "r7_pairs": len(r7),
            },
            "candidate_axis": "R7 pairwise loss weight 0.03 versus matched zero-mass control",
            "q4_labels_used_for_recipe_selection": False,
            "external_eval_feedback_used": False,
            "contains_external_eval_rows": False,
            "contains_external_eval_labels": False,
        }
    )
    write_json_atomic(args.output.resolve(), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = freeze(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

