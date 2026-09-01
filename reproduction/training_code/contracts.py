"""Portable integrity, provenance, and environment contracts for V170."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "v170.reproduction_training.1"
ALLOWED_SOURCE_PARTITION = "organizer_train"
SYNTHETIC_SOURCE_PARTITION = "openrouter_synthetic_no_row_source"
FORBIDDEN_PROVENANCE_KEYS = frozenset(
    {
        "external_eval_label",
        "external_eval_labels",
        "external_eval_rows",
        "external_eval_predictions",
        "private_label",
        "private_labels",
        "private_rows",
        "hidden_label",
        "hidden_labels",
        "external_eval_label",
        "external_eval_labels",
        "external_eval_score",
        "external_eval_scores",
    }
)
LOCKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "pandas",
    "pyarrow",
    "numpy",
    "scipy",
    "scikit-learn",
    "catboost",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_rank(seed: str, namespace: str, value: str) -> str:
    return hmac.new(
        str(seed).encode("utf-8"),
        f"{namespace}\0{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def bind_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    if "self_sha256" in output:
        raise ValueError("Cannot bind a payload that already contains self_sha256")
    output["self_sha256"] = canonical_sha256(output)
    return output


def verify_self_hash(payload: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    observed = payload.get("self_sha256")
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "self_sha256"}
    )
    if observed != expected:
        raise RuntimeError(f"{context} self-hash drift")
    return dict(payload)


def load_self_hashed_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain one JSON object")
    return verify_self_hash(value, context=context)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl_fsync(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_bound_file(
    manifest_path: Path,
    reference: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    relative = reference.get("path")
    expected_sha = reference.get("sha256")
    expected_bytes = reference.get("bytes")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{context} path must be nonempty and relative")
    root = manifest_path.resolve().parent
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{context} escapes its manifest directory") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = sha256_file(path)
    observed_bytes = path.stat().st_size
    if expected_sha != observed_sha or int(expected_bytes or -1) != observed_bytes:
        raise RuntimeError(
            f"{context} binding drift: sha={observed_sha}, bytes={observed_bytes}"
        )
    return path


def file_reference(path: Path, *, relative_to: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(relative_to.resolve())
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL line {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).strip().lower()
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def assert_allowed_training_provenance(
    manifest: Mapping[str, Any],
    *,
    context: str,
    allowed_source_partitions: Sequence[str] = (ALLOWED_SOURCE_PARTITION,),
) -> None:
    keys = set(_walk_keys(manifest))
    forbidden = sorted(keys & FORBIDDEN_PROVENANCE_KEYS)
    if forbidden:
        raise RuntimeError(f"{context} declares forbidden provenance keys: {forbidden}")
    required_false = (
        "contains_external_eval_rows",
        "contains_external_eval_labels",
        "contains_external_eval_predictions",
        "contains_private_rows",
        "contains_private_labels",
        "contains_hidden_labels",
        "external_eval_feedback_used",
    )
    for key in required_false:
        if manifest.get(key) is not False:
            raise RuntimeError(f"{context} must explicitly declare {key}=false")
    if manifest.get("source_partition") not in set(allowed_source_partitions):
        raise RuntimeError(
            f"{context} source_partition must be one of "
            f"{sorted(set(allowed_source_partitions))!r}"
        )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("V170 config schema drift")
    if config.get("experiment_id") != "quality_flv_openrouter_reproduction":
        raise RuntimeError("V170 config experiment identity drift")
    if config.get("base_model") != "Qwen/Qwen3.5-4B":
        raise RuntimeError("V170 base model drift")
    revision = str(config.get("base_revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("V170 base revision must be one immutable git SHA")
    if config.get("binary_token_ids") != {"0": 15, "1": 16}:
        raise RuntimeError("V170 atomic label-token contract drift")
    data = config.get("data")
    if (
        not isinstance(data, Mapping)
        or data.get("expected_folds") != [0, 1, 2, 3, 4]
        or data.get("supplementary_development_folds") != [0, 1, 2, 3]
        or data.get("forward_gate_fold") != 4
        or float(data.get("minimum_r5_structural_pass_rate", -1.0)) != 1.0
        or float(data.get("minimum_r5_direction_accuracy", -1.0)) != 1.0
        or float(data.get("minimum_r5_blind_usable_rate", -1.0)) != 0.95
        or int(data.get("expected_full_rows", -1)) <= 0
        or int(data.get("expected_full_positives", -1)) <= 0
        or int(data.get("expected_r7_relation_instances", -1)) <= 0
        or int(data.get("expected_r7_mass_keys", -1)) <= 0
        or set(data.get("expected_r7_query_counts", {})) != {"0", "1", "2", "3"}
    ):
        raise RuntimeError("V170 organizer-train/frozen-forward data contract drift")
    training = config.get("training")
    expected = {
        "seed": 20260822,
        "rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "epochs": 1.0,
        "effective_batch_size": 32,
        "micro_batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "r4_batch_size": 4,
        "r4_weight": 0.02,
        "r5_batch_size": 4,
        "r5_weight": 0.02,
        "r7_batch_size": 2,
        "r7_candidate_weight": 0.03,
        "r7_control_weight": 0.0,
        "checkpoint_fractions": [0.25, 0.5, 1.0],
        "text_only": True,
        "synthetic_ce_rows": 0,
    }
    if not isinstance(training, Mapping) or {
        key: training.get(key) for key in expected
    } != expected:
        raise RuntimeError("V170 matched LoRA training recipe drift")
    selector = config.get("selector")
    if not isinstance(selector, Mapping) or selector.get("all_expert_inputs_oof") is not True:
        raise RuntimeError("V170 selector must require all expert inputs OOF")
    if selector.get("rule_features_used") is not False:
        raise RuntimeError("V170 selector must exclude target-authored rule features")
    return dict(config)


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("V170 config must contain one JSON object")
    return validate_config(value)


def package_versions(names: Sequence[str] = LOCKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_fingerprint(*, include_cuda: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }
    if include_cuda:
        import torch

        result["torch_cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["cuda_device_count"] = int(torch.cuda.device_count())
            result["cuda_devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    return result


def source_inventory(paths: Sequence[Path], *, root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted({item.resolve() for item in paths}):
        result.append(
            {
                "path": path.relative_to(root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result
