#!/usr/bin/env python3
"""Audit or fit one matched V170 control/candidate LoRA trajectory."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    append_jsonl_fsync,
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
from .schedules import (
    build_schedule,
    component_class_weights,
    schedule_serializable_view,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "fit"), required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--r5-manifest", type=Path, required=True)
    parser.add_argument("--r7-manifest", type=Path, required=True)
    parser.add_argument("--recipe-precommit", type=Path, required=True)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--query-fold", type=int, choices=range(5))
    split.add_argument("--full-fit", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--base-snapshot-manifest", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _verify_execution_receipt(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = load_self_hashed_json(path, context="V170 execution receipt")
    if (
        receipt.get("schema_version") != "v170.execution_receipt.1"
        or receipt.get("status") != "PASS"
        or receipt.get("container_immutable_reference")
        != config["container"]["immutable_reference"]
        or receipt.get("hardware_class") != config["container"]["hardware_class"]
        or receipt.get("network_disabled_during_fit") is not True
    ):
        raise RuntimeError("V170 execution receipt contract drift")
    return {
        **receipt,
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
    }


def load_contract(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    rows, ce_binding = load_ce_dataset(args.ce_manifest.resolve(), config)
    r4, r4_binding = load_r4_pairs(args.r4_manifest.resolve(), rows, config)
    r5, r5_binding = load_r5_pairs(args.r5_manifest.resolve(), rows, config)
    r7, r7_binding = load_r7_pairs(args.r7_manifest.resolve(), rows, config)
    precommit = load_self_hashed_json(
        args.recipe_precommit.resolve(), context="V170 recipe precommit"
    )
    expected_precommit_files = {
        "config": sha256_file(args.config.resolve()),
        "ce_manifest": sha256_file(args.ce_manifest.resolve()),
        "r4_manifest": sha256_file(args.r4_manifest.resolve()),
        "r5_manifest": sha256_file(args.r5_manifest.resolve()),
        "r7_manifest": sha256_file(args.r7_manifest.resolve()),
    }
    if (
        precommit.get("schema_version") != "v170.recipe_precommit.1"
        or precommit.get("status") != "FROZEN_BEFORE_OOF_AND_Q4_SCORE"
        or precommit.get("files") != expected_precommit_files
        or precommit.get("external_eval_feedback_used") is not False
    ):
        raise RuntimeError("V170 recipe precommit contract drift")
    query_fold = None if args.full_fit else int(args.query_fold)
    train, query = split_ce_rows(
        rows,
        query_fold=query_fold,
        development_folds=config["data"]["supplementary_development_folds"],
        forward_gate_fold=int(config["data"]["forward_gate_fold"]),
    )
    selected_r4 = select_for_query_fold(r4, query_fold)
    selected_r5 = select_for_query_fold(r5, query_fold)
    selected_r7 = select_r7_for_query_fold(r7, query_fold)
    schedule = build_schedule(
        train,
        selected_r4,
        selected_r5,
        selected_r7,
        arm=args.arm,
        config=config,
    )
    audit = bind_self_hash(
        {
            "schema_version": "v170.lora_preflight.1",
            "status": "AUDIT_PASS",
            "arm": args.arm,
            "query_fold": query_fold,
            "full_fit": query_fold is None,
            "config_sha256": sha256_file(args.config.resolve()),
            "recipe_precommit_sha256": sha256_file(args.recipe_precommit.resolve()),
            "recipe_precommit_self_sha256": precommit["self_sha256"],
            "train_rows": len(train),
            "train_positives": int(train.label.sum()),
            "query_rows": len(query),
            "query_positives": int(query.label.sum()) if len(query) else 0,
            "train_folds": sorted(train.fold.unique().tolist()),
            "query_components_excluded": True,
            "text_only": True,
            "images_read": 0,
            "external_eval_rows_read": 0,
            "external_eval_labels_read": 0,
            "external_eval_feedback_used": False,
            "r4_inventory_total": len(r4),
            "r4_inventory_eligible": len(selected_r4),
            "r5_inventory_total": len(r5),
            "r5_inventory_eligible": len(selected_r5),
            "r7_inventory_total": len(r7),
            "r7_inventory_eligible": len(selected_r7),
            "r7_channel_weight": float(
                config["training"][
                    "r7_candidate_weight" if args.arm == "candidate" else "r7_control_weight"
                ]
            ),
            "matched_control_candidate_axes": ["r7_channel_weight_and_gradient_only"],
            "ce_binding": ce_binding,
            "r4_binding": r4_binding,
            "r5_binding": r5_binding,
            "r7_binding": r7_binding,
            "schedule": schedule_serializable_view(schedule),
        }
    )
    return {
        "config": config,
        "rows": rows,
        "train": train,
        "query": query,
        "r4": selected_r4,
        "r5": selected_r5,
        "r7": selected_r7,
        "schedule": schedule,
        "audit": audit,
    }


def _checkpoint_tag(fraction: float) -> str:
    return f"checkpoint_{int(round(100 * fraction)):03d}"


def fit(args: argparse.Namespace, contract: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    from .modeling import (
        attach_lora,
        binary_logits,
        cosine_multiplier,
        frozen_sentinel,
        load_base_model,
        load_processor,
        pairwise_backward,
        processor_batch,
        save_adapter_checkpoint,
        verify_model_snapshot,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("V170 LoRA fit requires CUDA")
    if args.model_path is None or args.base_snapshot_manifest is None:
        raise RuntimeError("V170 fit requires model path and base snapshot manifest")
    if args.execution_receipt is None:
        raise RuntimeError("V170 fit requires a self-hashed execution receipt")
    if args.output_dir.exists():
        raise FileExistsError("V170 fit requires a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output_dir / "preflight.json", contract["audit"])

    config = contract["config"]
    training = config["training"]
    receipt = _verify_execution_receipt(args.execution_receipt.resolve(), config)
    snapshot = verify_model_snapshot(
        args.model_path.resolve(),
        args.base_snapshot_manifest.resolve(),
        config,
    )
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(training["tf32"])

    processor = load_processor(args.model_path.resolve(), config)
    model = load_base_model(args.model_path.resolve(), config)
    model, adapter = attach_lora(model, config)
    model = model.to("cuda")
    sentinel_before = frozen_sentinel(model, seed=str(seed))
    device = torch.device("cuda")

    train = contract["train"].copy().reset_index(drop=True)
    train["loss_weight"] = component_class_weights(train)
    r4 = contract["r4"]
    r5 = contract["r5"]
    r7 = contract["r7"]
    schedule = contract["schedule"]
    total_steps = int(schedule["optimizer_steps"])
    checkpoint_map = {
        int(step): float(fraction)
        for step, fraction in schedule["checkpoint_steps"].items()
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_multiplier(
            step,
            total_steps,
            int(schedule["warmup_steps"]),
        ),
    )
    events_path = args.output_dir / "train_events.jsonl"
    checkpoints: list[dict[str, Any]] = []
    rows_consumed = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    effective = int(training["effective_batch_size"])
    micro = int(training["micro_batch_size"])
    r4_batch_size = int(training["r4_batch_size"])
    r5_batch_size = int(training["r5_batch_size"])
    r7_batch_size = int(training["r7_batch_size"])
    for step_index in range(total_steps):
        step = step_index + 1
        block = schedule["ce_order"][step_index * effective : (step_index + 1) * effective]
        if not block:
            raise RuntimeError("V170 CE schedule produced an empty optimizer step")
        optimizer.zero_grad(set_to_none=True)
        ce_numerator = 0.0
        for begin in range(0, len(block), micro):
            batch = train.iloc[block[begin : begin + micro]]
            features = processor_batch(processor, batch, config=config, device=device)
            targets = torch.tensor(batch.label.tolist(), dtype=torch.long, device=device)
            weights = torch.tensor(
                batch.loss_weight.tolist(), dtype=torch.float32, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = binary_logits(model, features, config)
                per_row = functional.cross_entropy(logits, targets, reduction="none")
                loss = (per_row * weights).sum() / len(block)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V170 CE loss is non-finite")
            loss.backward()
            ce_numerator += float((per_row.detach() * weights).sum().cpu())
        r4_batch = r4.iloc[
            schedule["r4_order"][step_index * r4_batch_size : (step_index + 1) * r4_batch_size]
        ]
        r4_stats = pairwise_backward(
            model,
            processor,
            r4_batch,
            channel_weight=float(training["r4_weight"]),
            config=config,
            device=device,
        )
        r5_batch = r5.iloc[
            schedule["r5_order"][step_index * r5_batch_size : (step_index + 1) * r5_batch_size]
        ]
        r5_stats = pairwise_backward(
            model,
            processor,
            r5_batch,
            channel_weight=float(training["r5_weight"]),
            config=config,
            device=device,
        )
        r7_stats = {"raw_loss": 0.0, "mean_margin": 0.0, "accuracy": 0.0}
        if args.arm == "candidate":
            r7_batch = r7.iloc[
                schedule["r7_order"][
                    step_index * r7_batch_size : (step_index + 1) * r7_batch_size
                ]
            ]
            r7_stats = pairwise_backward(
                model,
                processor,
                r7_batch,
                channel_weight=float(training["r7_candidate_weight"]),
                config=config,
                device=device,
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["max_grad_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("V170 adapter gradient norm is non-finite")
        if not any(
            parameter.grad is not None and bool(parameter.grad.detach().abs().max() > 0)
            for parameter in trainable
        ):
            raise RuntimeError("V170 adapter gradient is identically zero")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        rows_consumed += len(block)
        append_jsonl_fsync(
            events_path,
            {
                "schema_version": "v170.optimizer_step.1",
                "event": "optimizer_step",
                "step": step,
                "arm": args.arm,
                "query_fold": contract["audit"]["query_fold"],
                "rows_consumed": rows_consumed,
                "block_rows": len(block),
                "ce_loss": ce_numerator / len(block),
                "r4_pairwise_raw_loss": r4_stats["raw_loss"],
                "r4_pairwise_weight": float(training["r4_weight"]),
                "r5_pairwise_raw_loss": r5_stats["raw_loss"],
                "r5_pairwise_weight": float(training["r5_weight"]),
                "r7_pairwise_raw_loss": r7_stats["raw_loss"],
                "r7_pairwise_weight": float(
                    training[
                        "r7_candidate_weight"
                        if args.arm == "candidate"
                        else "r7_control_weight"
                    ]
                ),
                "gradient_norm": float(grad_norm.detach().float().cpu()),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        if step in checkpoint_map:
            fraction = checkpoint_map[step]
            tag = _checkpoint_tag(fraction)
            saved = save_adapter_checkpoint(
                model,
                args.output_dir / tag,
                step=step,
                fraction=fraction,
            )
            checkpoints.append({"tag": tag, **saved})

    sentinel_after = frozen_sentinel(model, seed=str(seed))
    if sentinel_after != sentinel_before:
        raise RuntimeError("V170 frozen base-model sentinel changed")
    if rows_consumed != len(train) or [item["tag"] for item in checkpoints] != [
        "checkpoint_025",
        "checkpoint_050",
        "checkpoint_100",
    ]:
        raise RuntimeError("V170 full-epoch/checkpoint completion contract failed")
    status = bind_self_hash(
        {
            **{
                key: value
                for key, value in contract["audit"].items()
                if key != "self_sha256"
            },
            "schema_version": "v170.lora_fit_status.1",
            "status": "FIT_COMPLETE_UNSCORED",
            "preflight_self_sha256": contract["audit"]["self_sha256"],
            "optimizer_steps": total_steps,
            "rows_consumed": rows_consumed,
            "full_epoch_complete": True,
            "checkpoints": checkpoints,
            "train_events_sha256": sha256_file(events_path),
            "model_snapshot": snapshot,
            "execution_receipt": receipt,
            "adapter": adapter,
            "frozen_sentinel": sentinel_after,
            "environment": environment_fingerprint(include_cuda=True),
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "trainer_sha256": sha256_file(Path(__file__)),
        }
    )
    write_json_atomic(args.output_dir / "fit_status.json", status)
    del optimizer, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return status


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("V170 audit/fit requires a fresh output directory")
    contract = load_contract(args)
    if args.phase == "audit":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(args.output_dir / "preflight.json", contract["audit"])
        print(json.dumps(contract["audit"], ensure_ascii=False, indent=2, sort_keys=True))
        return
    result = fit(args, contract)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
