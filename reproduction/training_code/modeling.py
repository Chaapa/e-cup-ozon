"""Lazy GPU/model primitives for the V170 LoRA trajectory."""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contracts import (
    bind_self_hash,
    canonical_sha256,
    load_self_hashed_json,
    sha256_file,
    write_json_atomic,
)


def verify_model_snapshot(
    model_path: Path,
    snapshot_manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_self_hashed_json(
        snapshot_manifest_path,
        context="V170 base snapshot manifest",
    )
    if (
        manifest.get("schema_version") != "v170.base_snapshot.1"
        or manifest.get("status") != "PASS"
        or manifest.get("base_model") != config["base_model"]
        or manifest.get("base_revision") != config["base_revision"]
    ):
        raise RuntimeError("V170 base snapshot manifest identity drift")
    root = model_path.resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("V170 base snapshot manifest has no files")
    observed: list[dict[str, Any]] = []
    for expected in files:
        relative = Path(str(expected.get("path", "")))
        if relative.is_absolute() or not str(relative):
            raise RuntimeError("V170 base snapshot file path must be relative")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("V170 base snapshot file escapes model root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        item = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if item != expected:
            raise RuntimeError(f"V170 base snapshot file drift: {relative}")
        observed.append(item)
    if canonical_sha256(observed) != manifest.get("files_sha256"):
        raise RuntimeError("V170 base snapshot aggregate hash drift")
    return {
        "manifest_path": str(snapshot_manifest_path.resolve()),
        "manifest_sha256": sha256_file(snapshot_manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "base_model": config["base_model"],
        "base_revision": config["base_revision"],
        "files": observed,
        "files_sha256": manifest["files_sha256"],
    }


def load_processor(model_path: Path, config: Mapping[str, Any]) -> Any:
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(
        str(model_path.resolve()),
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    zero_id = processor.tokenizer("0", add_special_tokens=False)["input_ids"]
    one_id = processor.tokenizer("1", add_special_tokens=False)["input_ids"]
    if zero_id != [int(config["binary_token_ids"]["0"])] or one_id != [
        int(config["binary_token_ids"]["1"])
    ]:
        raise RuntimeError(f"V170 atomic token drift: zero={zero_id}, one={one_id}")
    return processor


def load_base_model(model_path: Path, config: Mapping[str, Any]) -> Any:
    import torch
    import transformers

    model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if model_class is None:
        raise RuntimeError(
            "V170 requires transformers.Qwen3_5ForConditionalGeneration"
        )
    model = model_class.from_pretrained(
        str(model_path.resolve()),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(config["training"]["attention_implementation"]),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def discover_lora_targets(model: Any, config: Mapping[str, Any]) -> list[str]:
    import torch

    suffixes = tuple(str(value) for value in config["training"]["lora_target_suffixes"])
    targets: list[str] = []
    for name, module in model.named_modules():
        if ".language_model." not in name:
            continue
        if not any(name.endswith("." + suffix) for suffix in suffixes):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            targets.append(name)
    targets = sorted(set(targets))
    if not targets or any(
        marker in name.lower() for name in targets for marker in ("visual", "merger")
    ):
        raise RuntimeError("V170 language-only LoRA target discovery failed")
    return targets


def attach_lora(model: Any, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model

    training = config["training"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = discover_lora_targets(model, config)
    adapter_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(training["rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        target_modules=targets,
        bias="none",
        use_rslora=False,
        use_dora=False,
    )
    model = get_peft_model(model, adapter_config)
    if bool(training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("V170 exposed non-LoRA trainable parameters")
    return model, {
        "rank": int(training["rank"]),
        "alpha": int(training["lora_alpha"]),
        "dropout": float(training["lora_dropout"]),
        "target_modules": targets,
        "target_modules_sha256": canonical_sha256(targets),
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": int(sum(value.numel() for _, value in trainable)),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "visual_targets": 0,
        "merger_targets": 0,
    }


def _sentinel_positions(*, numel: int, width: int, device: Any) -> Any:
    import torch

    positions = torch.arange(width, dtype=torch.int64, device=device)
    return torch.div(positions * (numel - 1), width - 1, rounding_mode="floor")


def frozen_sentinel(model: Any, *, seed: str) -> dict[str, Any]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" not in name and parameter.numel() > 1
    ]
    candidates.sort(
        key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest()
    )
    result = []
    for name, parameter in candidates[:16]:
        flat = parameter.detach().reshape(-1)
        width = min(4096, flat.numel())
        sample = (
            flat
            if flat.numel() <= width
            else flat.index_select(
                0,
                _sentinel_positions(
                    numel=flat.numel(),
                    width=width,
                    device=flat.device,
                ),
            )
        )
        payload = sample.to(dtype=__import__("torch").float32, device="cpu").numpy().tobytes()
        result.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if len(result) != 16:
        raise RuntimeError("V170 base sentinel could not sample 16 tensors")
    return {"tensors": result, "contract_sha256": canonical_sha256(result)}


def render_user_prompt(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    return str(config["prompt"]["user_template"]).format(
        title=str(row.get("clean_name", "")),
        description=str(row.get("clean_description", "")),
    )


def processor_batch(
    processor: Any,
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    conversations = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": str(config["prompt"]["system"])}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": render_user_prompt(record, config)}
                ],
            },
        ]
        for record in frame.to_dict("records")
    ]
    features = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        enable_thinking=False,
    )
    if "input_ids" not in features or int(features["input_ids"].shape[0]) != len(frame):
        raise RuntimeError("V170 processor lost text rows")
    if "pixel_values" in features or "image_grid_thw" in features:
        raise RuntimeError("V170 text-only trajectory unexpectedly produced image tensors")
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in features.items()
    }


def binary_logits(model: Any, features: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
    output = model(**features, use_cache=False)
    ids = [int(config["binary_token_ids"]["0"]), int(config["binary_token_ids"]["1"])]
    logits = output.logits[:, -1, ids].float()
    if logits.ndim != 2 or int(logits.shape[1]) != 2:
        raise RuntimeError("V170 binary-logit shape drift")
    return logits


def pairwise_backward(
    model: Any,
    processor: Any,
    frame: pd.DataFrame,
    *,
    channel_weight: float,
    config: Mapping[str, Any],
    device: Any,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    positive = pd.DataFrame(
        {
            "clean_name": frame.positive_name.astype(str).tolist(),
            "clean_description": frame.positive_description.astype(str).tolist(),
        }
    )
    negative = pd.DataFrame(
        {
            "clean_name": frame.negative_name.astype(str).tolist(),
            "clean_description": frame.negative_description.astype(str).tolist(),
        }
    )
    features = processor_batch(
        processor,
        pd.concat([positive, negative], ignore_index=True),
        config=config,
        device=device,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = binary_logits(model, features, config)
        scores = logits[:, 1] - logits[:, 0]
        pair_count = len(frame)
        margins = scores[:pair_count] - scores[pair_count:]
        raw_loss = functional.softplus(-margins).mean()
        weighted_loss = raw_loss * float(channel_weight)
    if not bool(torch.isfinite(weighted_loss)) or not bool(torch.isfinite(margins).all()):
        raise RuntimeError("V170 pair channel produced a non-finite loss/margin")
    weighted_loss.backward()
    return {
        "raw_loss": float(raw_loss.detach().float().cpu()),
        "mean_margin": float(margins.detach().float().mean().cpu()),
        "accuracy": float(margins.detach().gt(0).float().mean().cpu()),
    }


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def save_adapter_checkpoint(
    model: Any,
    directory: Path,
    *,
    step: int,
    fraction: float,
) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    config_path = directory / "adapter_config.json"
    model_path = directory / "adapter_model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise RuntimeError("V170 adapter checkpoint is incomplete")
    status = bind_self_hash(
        {
            "schema_version": "v170.adapter_checkpoint.1",
            "status": "COMPLETE_UNSCORED",
            "step": int(step),
            "fraction": float(fraction),
            "adapter_config": {
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            "adapter_model": {
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            },
        }
    )
    write_json_atomic(directory / "checkpoint_manifest.json", status)
    return {
        "step": int(step),
        "fraction": float(fraction),
        "directory": directory.name,
        "checkpoint_manifest_sha256": sha256_file(directory / "checkpoint_manifest.json"),
        "checkpoint_manifest_self_sha256": status["self_sha256"],
        "adapter_config_sha256": status["adapter_config"]["sha256"],
        "adapter_model_sha256": status["adapter_model"]["sha256"],
        "adapter_bytes": status["adapter_model"]["bytes"],
    }


def release_model(*values: Any) -> None:
    import torch

    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
