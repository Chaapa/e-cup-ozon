#!/usr/bin/env python3
"""Write the exact, non-executing V170 OOF/full-fit command DAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .contracts import bind_self_hash, load_self_hashed_json, sha256_file, write_json_atomic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--r5-manifest", type=Path, required=True)
    parser.add_argument("--r7-manifest", type=Path, required=True)
    parser.add_argument("--recipe-precommit", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _base(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training_code.train_lora",
        "--config",
        str(args.config.resolve()),
        "--ce-manifest",
        str(args.ce_manifest.resolve()),
        "--r4-manifest",
        str(args.r4_manifest.resolve()),
        "--r5-manifest",
        str(args.r5_manifest.resolve()),
        "--r7-manifest",
        str(args.r7_manifest.resolve()),
        "--recipe-precommit",
        str(args.recipe_precommit.resolve()),
    ]


def create(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("V170 execution plan must be frozen once")
    precommit = load_self_hashed_json(
        args.recipe_precommit.resolve(), context="V170 recipe precommit"
    )
    if precommit.get("status") != "FROZEN_BEFORE_OOF_AND_Q4_SCORE":
        raise RuntimeError("V170 execution plan requires a frozen recipe")
    root = args.run_root.resolve()
    stages: list[dict[str, Any]] = []
    for fold in range(5):
        for arm in ("control", "candidate"):
            stem = f"{arm}_q{fold}"
            audit_dir = root / "audits" / stem
            fit_dir = root / "fits" / arm / f"fold_{fold}"
            score_dir = root / "oof" / arm / f"fold_{fold}"
            stages.append(
                {
                    "stage_id": f"audit_{stem}",
                    "kind": "cpu_audit",
                    "depends_on": [],
                    "command": _base(args)
                    + [
                        "--phase",
                        "audit",
                        "--arm",
                        arm,
                        "--query-fold",
                        str(fold),
                        "--output-dir",
                        str(audit_dir),
                    ],
                    "expected": str(audit_dir / "preflight.json"),
                }
            )
            stages.append(
                {
                    "stage_id": f"fit_{stem}",
                    "kind": "gpu_fit",
                    "depends_on": [f"audit_{stem}"],
                    "command": _base(args)
                    + [
                        "--phase",
                        "fit",
                        "--arm",
                        arm,
                        "--query-fold",
                        str(fold),
                        "--model-path",
                        str(args.model_path.resolve()),
                        "--base-snapshot-manifest",
                        str(args.base_snapshot_manifest.resolve()),
                        "--execution-receipt",
                        str(args.execution_receipt.resolve()),
                        "--output-dir",
                        str(fit_dir),
                    ],
                    "expected": str(fit_dir / "fit_status.json"),
                }
            )
            stages.append(
                {
                    "stage_id": f"score_{stem}",
                    "kind": "gpu_score_three_checkpoints",
                    "depends_on": [f"fit_{stem}"],
                    "command": [
                        sys.executable,
                        "-m",
                        "training_code.score_lora",
                        "--config",
                        str(args.config.resolve()),
                        "--ce-manifest",
                        str(args.ce_manifest.resolve()),
                        "--fit-dir",
                        str(fit_dir),
                        "--model-path",
                        str(args.model_path.resolve()),
                        "--base-snapshot-manifest",
                        str(args.base_snapshot_manifest.resolve()),
                        "--output-dir",
                        str(score_dir),
                    ],
                    "expected": str(score_dir / "score_status.json"),
                }
            )
    all_scores = [f"score_{arm}_q{fold}" for fold in range(5) for arm in ("control", "candidate")]
    gate_dir = root / "q4_gate"
    stages.append(
        {
            "stage_id": "q4_gate",
            "kind": "cpu_transfer_gate",
            "depends_on": all_scores,
            "command": [
                sys.executable,
                "-m",
                "training_code.evaluate_q4_gate",
                "--config",
                str(args.config.resolve()),
                "--ce-manifest",
                str(args.ce_manifest.resolve()),
                "--recipe-precommit",
                str(args.recipe_precommit.resolve()),
                "--oof-root",
                str(root / "oof"),
                "--output-dir",
                str(gate_dir),
            ],
            "expected": str(gate_dir / "q4_gate_report.json"),
            "stop_on_status": "STOP",
        }
    )
    selector_dir = root / "selector"
    stages.append(
        {
            "stage_id": "selector_full_oof",
            "kind": "cpu_selector_fit",
            "depends_on": ["q4_gate"],
            "command": [
                sys.executable,
                "-m",
                "training_code.fit_selector",
                "--phase",
                "fit",
                "--config",
                str(args.config.resolve()),
                "--ce-manifest",
                str(args.ce_manifest.resolve()),
                "--oof-root",
                str(root / "oof"),
                "--gate-report",
                str(gate_dir / "q4_gate_report.json"),
                "--output-dir",
                str(selector_dir),
            ],
            "expected": str(selector_dir / "selector_manifest.json"),
        }
    )
    for arm in ("control", "candidate"):
        audit_dir = root / "audits" / f"{arm}_fullfit"
        fit_dir = root / "fullfit" / arm
        stages.append(
            {
                "stage_id": f"audit_{arm}_fullfit",
                "kind": "cpu_audit",
                "depends_on": ["q4_gate"],
                "command": _base(args)
                + [
                    "--phase",
                    "audit",
                    "--arm",
                    arm,
                    "--full-fit",
                    "--output-dir",
                    str(audit_dir),
                ],
                "expected": str(audit_dir / "preflight.json"),
            }
        )
        stages.append(
            {
                "stage_id": f"fit_{arm}_fullfit",
                "kind": "gpu_fit",
                "depends_on": [f"audit_{arm}_fullfit", "selector_full_oof"],
                "command": _base(args)
                + [
                    "--phase",
                    "fit",
                    "--arm",
                    arm,
                    "--full-fit",
                    "--model-path",
                    str(args.model_path.resolve()),
                    "--base-snapshot-manifest",
                    str(args.base_snapshot_manifest.resolve()),
                    "--execution-receipt",
                    str(args.execution_receipt.resolve()),
                    "--output-dir",
                    str(fit_dir),
                ],
                "expected": str(fit_dir / "fit_status.json"),
            }
        )
    payload = bind_self_hash(
        {
            "schema_version": "v170.execution_plan.1",
            "status": "FROZEN_NOT_EXECUTED",
            "recipe_precommit_sha256": sha256_file(args.recipe_precommit.resolve()),
            "recipe_precommit_self_sha256": precommit["self_sha256"],
            "run_root": str(root),
            "counts": {
                "oof_lora_fits": 10,
                "full_lora_fits": 2,
                "checkpoint_score_passes": 30,
                "q4_gates": 1,
                "selector_fits": 1,
            },
            "stages": stages,
            "automatic_execution": False,
            "stop_after_q4_stop": True,
            "external_eval_data_used": False,
        }
    )
    write_json_atomic(args.output.resolve(), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = create(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

