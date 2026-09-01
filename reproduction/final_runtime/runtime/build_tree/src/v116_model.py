from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v115_round7_runtime_features import (
    attach_runtime_component_size, augment_runtime_features,
)

from .constants import (
    ARTIFACTS,
    BASE_MODEL,
    BASE_REVISION,
    MAX_PIXELS,
    MIN_PIXELS,
    ONE_TOKEN_ID,
    QWEN_MODEL,
    QUOTA_DENOMINATOR,
    QUOTA_NUMERATOR,
    SCORE_BATCH_SIZE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    ZERO_TOKEN_ID,
)


BASES = ("b1", "char", "lora025", "lora050", "lora100")
CHECKPOINTS = (
    ("checkpoint_025", "lora025"),
    ("checkpoint_050", "lora050"),
    ("checkpoint_100", "lora100"),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_quota(row_count: int) -> int:
    if row_count < 0:
        raise ValueError("Negative V116 row count")
    return min(
        row_count,
        (2 * row_count * QUOTA_NUMERATOR + QUOTA_DENOMINATOR) // (2 * QUOTA_DENOMINATOR),
    )


def rank_predictions(scores: Sequence[float], ties: Sequence[str]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    keys = [str(value) for value in ties]
    if values.ndim != 1 or len(values) != len(keys) or not np.isfinite(values).all():
        raise ValueError("V116 ranking requires one finite score per row")
    if len(set(keys)) != len(keys):
        raise ValueError("V116 ranking requires unique deterministic ties")
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), keys[index]))
    prediction = np.zeros(len(values), dtype=np.int8)
    prediction[order[: positive_quota(len(values))]] = 1
    return prediction


def production_manifest() -> dict[str, Any]:
    path = ARTIFACTS / "production_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing V116 production manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_self = sha256_bytes(
        canonical({key: item for key, item in value.items() if key != "self_sha256"}).encode("utf-8")
    )
    if (
        not isinstance(value, dict)
        or value.get("self_sha256") != expected_self
        or value.get("status") != "PRODUCTION_ARTIFACTS_COMPLETE"
        or value.get("package_id") != "flv_control_candidate_runtime"
        or value.get("base_model") != BASE_MODEL
        or value.get("base_revision") != BASE_REVISION
        or value.get("training_rows") != 5_502
        or value.get("training_positives") != 198
        or value.get("control_rank") != 16
        or value.get("candidate_rank") not in (8, 16, 32, 64)
        or value.get("text_only") is not True
        or value.get("quota_numerator") != QUOTA_NUMERATOR
        or value.get("quota_denominator") != QUOTA_DENOMINATOR
        or value.get("contains_organizer_rows") is not False
        or value.get("contains_images") is not False
        or value.get("contains_predictions") is not False
        or value.get("meta_fit_rows") != 5_502
        or value.get("meta_fit_folds") != [0, 1, 2, 3, 4]
        or value.get("meta_fold4_used") is not True
        or value.get("meta_all_expert_inputs_oof") is not True
        or value.get("recipe_frozen_before_fold4") is not True
        or value.get("fold4_used_for_candidate_selection") is not False
        or value.get("q4_gate_passed") is not True
        or not isinstance(value.get("feature_columns"), list)
        or len(value["feature_columns"]) != 71
    ):
        raise RuntimeError("V116 production manifest contract drift")
    expected_artifacts = value.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        raise RuntimeError("V116 manifest artifact inventory missing")
    for relative, expected in expected_artifacts.items():
        artifact = ARTIFACTS / relative
        observed = {
            "bytes": artifact.stat().st_size if artifact.is_file() else -1,
            "sha256": sha256_file(artifact) if artifact.is_file() else None,
        }
        if observed != expected:
            raise RuntimeError(f"V116 artifact hash drift: {relative}")
    if value.get("model_snapshot_files_sha256") != sha256_bytes(
        canonical(value.get("base_file_contract")).encode("utf-8")
    ):
        raise RuntimeError("V116 base-file contract digest drift")
    return value

def verify_base_snapshot(manifest: Mapping[str, Any]) -> None:
    if not QWEN_MODEL.is_dir():
        raise FileNotFoundError(f"Shared V116 base model is missing: {QWEN_MODEL}")
    for expected in manifest["base_file_contract"]:
        path = QWEN_MODEL / str(expected["path"])
        observed = {
            "path": str(expected["path"]),
            "bytes": path.stat().st_size if path.is_file() else -1,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        if observed != expected:
            raise RuntimeError(f"V116 shared base-file mismatch: {expected['path']}")


def feature_text(frame: pd.DataFrame) -> list[str]:
    return [
        f"title {name}\n description {description}"
        for name, description in zip(
            frame.normalized_name.fillna("").astype(str),
            frame.normalized_description.fillna("").astype(str),
            strict=True,
        )
    ]


def predict_cpu_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    import joblib
    from scipy.sparse import hstack

    values = feature_text(frame)
    b1 = joblib.load(ARTIFACTS / "cpu/b1_full.joblib")
    b1_matrix = hstack(
        [b1["char"].transform(values), b1["word"].transform(values)], format="csr"
    )
    b1_scores = np.asarray(b1["model"].decision_function(b1_matrix), dtype=np.float64)
    char = joblib.load(ARTIFACTS / "cpu/char_full.joblib")
    char_scores = np.asarray(
        char["model"].decision_function(char["vectorizer"].transform(values)),
        dtype=np.float64,
    )
    if not np.isfinite(b1_scores).all() or not np.isfinite(char_scores).all():
        raise RuntimeError("V116 CPU experts produced non-finite scores")
    return {"b1": b1_scores, "char": char_scores}


def load_processor() -> Any:
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(str(QWEN_MODEL), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if processor.tokenizer("0", add_special_tokens=False)["input_ids"] != [ZERO_TOKEN_ID]:
        raise RuntimeError("V116 token 0 drift")
    if processor.tokenizer("1", add_special_tokens=False)["input_ids"] != [ONE_TOKEN_ID]:
        raise RuntimeError("V116 token 1 drift")
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("V116 processor has no image processor")
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = MAX_PIXELS
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = MIN_PIXELS
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = MAX_PIXELS
        image_processor.size["shortest_edge"] = MIN_PIXELS
    return processor


def _conversations(frame: pd.DataFrame) -> tuple[list[list[dict[str, Any]]], ExitStack]:
    stack = ExitStack()
    output: list[list[dict[str, Any]]] = []
    for row in frame.to_dict(orient="records"):
        output.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_TEMPLATE.format(
                                title=row["clean_name"], description=row["clean_description"]
                            ),
                        },
                    ],
                },
            ]
        )
    return output, stack

def _features(processor: Any, frame: pd.DataFrame, device: Any) -> Any:
    conversations, stack = _conversations(frame)
    try:
        features = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
            enable_thinking=False,
        )
    finally:
        stack.close()
    if "input_ids" not in features or int(features["input_ids"].shape[0]) != len(frame):
        raise RuntimeError("V116 processor lost rows")
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in features.items()
    }


def _score_adapter(model: Any, processor: Any, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    device = next(model.parameters()).device
    order = sorted(
        range(len(frame)),
        key=lambda index: (
            len(str(frame.at[index, "clean_name"])) + len(str(frame.at[index, "clean_description"])),
            str(frame.at[index, "runtime_uid"]),
        ),
    )
    result = np.zeros(len(frame), dtype=np.float64)
    widths: list[int] = []
    started = time.perf_counter()
    for begin in range(0, len(order), SCORE_BATCH_SIZE):
        indices = order[begin : begin + SCORE_BATCH_SIZE]
        features = _features(processor, frame.loc[indices], device)
        widths.append(int(features["input_ids"].shape[1]))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**features, use_cache=False)
            logits = output.logits[:, -1, [ZERO_TOKEN_ID, ONE_TOKEN_ID]].float()
        scores = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        for index, score in zip(indices, scores, strict=True):
            result[index] = float(score)
    if not np.isfinite(result).all():
        raise RuntimeError("V116 LoRA produced non-finite scores")
    elapsed = time.perf_counter() - started
    return result, {
        "rows": len(frame),
        "seconds": elapsed,
        "rows_per_second": len(frame) / max(elapsed, 1e-9),
        "max_prompt_width": max(widths, default=0),
    }


def mock_lora_scores(frame: pd.DataFrame, *, namespace: str) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for checkpoint, name in CHECKPOINTS:
        output[name] = np.asarray(
            [
                (int(hashlib.sha256(f"{namespace}\0{checkpoint}\0{tie}".encode()).hexdigest()[:16], 16) / (2**64 - 1)) * 2 - 1
                for tie in frame.tie_key.astype(str)
            ],
            dtype=np.float64,
        )
    return output

def predict_lora_scores(frame: pd.DataFrame, *, mock: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    manifest = production_manifest()
    if mock:
        return (
            mock_lora_scores(frame, namespace="control"),
            mock_lora_scores(frame, namespace="candidate"),
            {"mock": True, "rows": len(frame)},
        )
    import torch
    import transformers
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V116 production inference")
    verify_base_snapshot(manifest)
    processor = load_processor()
    model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if model_class is None:
        raise RuntimeError("V116 runtime lacks Qwen3_5ForConditionalGeneration")
    base = model_class.from_pretrained(
        str(QWEN_MODEL), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    first_checkpoint, _ = CHECKPOINTS[0]
    first_adapter = f"control_{first_checkpoint}"
    model = PeftModel.from_pretrained(
        base,
        str(ARTIFACTS / "lora" / "control" / first_checkpoint),
        adapter_name=first_adapter,
        is_trainable=False,
    )
    for namespace in ("control", "candidate"):
        for checkpoint, _ in CHECKPOINTS:
            adapter_name = f"{namespace}_{checkpoint}"
            if adapter_name == first_adapter:
                continue
            model.load_adapter(
                str(ARTIFACTS / "lora" / namespace / checkpoint),
                adapter_name=adapter_name,
                is_trainable=False,
            )
    model = model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    trajectories: dict[str, dict[str, np.ndarray]] = {"control": {}, "candidate": {}}
    performance: dict[str, Any] = {}
    try:
        for namespace in ("control", "candidate"):
            performance[namespace] = {}
            for checkpoint, name in CHECKPOINTS:
                model.set_adapter(f"{namespace}_{checkpoint}")
                values, status = _score_adapter(model, processor, frame)
                trajectories[namespace][name] = values
                performance[namespace][checkpoint] = status
    finally:
        del model, base, processor
        gc.collect()
        torch.cuda.empty_cache()
    performance["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return trajectories["control"], trajectories["candidate"], {
        "mock": False, "rows": len(frame), "checkpoints": performance
    }

def _percentile(values: pd.Series) -> pd.Series:
    return values.astype(float).rank(method="average", pct=True)


def assemble_features(
    frame: pd.DataFrame,
    scores: Mapping[str, Sequence[float]],
    predictions: Mapping[str, Sequence[int]],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for base in BASES:
        score = pd.Series(np.asarray(scores[base], dtype=np.float64), index=frame.index)
        features[f"{base}_score"] = score
        features[f"{base}_rank"] = _percentile(score)
        std = float(score.std())
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        features[f"{base}_z"] = (score - float(score.mean())) / std
        features[f"{base}_prediction"] = np.asarray(predictions[base], dtype=np.int8)
    z_columns = [f"{base}_z" for base in BASES]
    rank_columns = [f"{base}_rank" for base in BASES]
    pred_columns = [f"{base}_prediction" for base in BASES]
    features["vote_count"] = features[pred_columns].sum(axis=1)
    for prefix, columns in (("z", z_columns), ("rank", rank_columns)):
        features[f"{prefix}_mean"] = features[columns].mean(axis=1)
        features[f"{prefix}_std"] = features[columns].std(axis=1)
        features[f"{prefix}_min"] = features[columns].min(axis=1)
        features[f"{prefix}_max"] = features[columns].max(axis=1)
    for left, right, name in (
        ("lora050", "lora025", "lora_delta_050_025"),
        ("lora100", "lora050", "lora_delta_100_050"),
        ("lora100", "lora025", "lora_delta_100_025"),
    ):
        features[f"{name}_score"] = features[f"{left}_z"] - features[f"{right}_z"]
        features[f"{name}_rank"] = features[f"{left}_rank"] - features[f"{right}_rank"]
    features["component_log1p"] = np.log1p(frame.component_size.astype(float))
    features["image_count"] = frame.image_count.astype(float)
    features["name_chars_log1p"] = np.log1p(frame.clean_name.fillna("").astype(str).str.len())
    features["description_chars_log1p"] = np.log1p(
        frame.clean_description.fillna("").astype(str).str.len()
    )
    expected = list(feature_columns)
    if set(features.columns) != set(expected):
        raise RuntimeError(
            f"V116 feature schema drift: missing={sorted(set(expected)-set(features))}, "
            f"extra={sorted(set(features)-set(expected))}"
        )
    features = features[expected]
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise RuntimeError("V116 selector features are non-finite")
    return features


def predict_flammable(frame: pd.DataFrame, *, mock: bool) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = production_manifest()
    frame = attach_runtime_component_size(frame)
    cpu_scores = predict_cpu_scores(frame)
    control_lora, candidate_lora, lora_status = predict_lora_scores(frame, mock=mock)
    control_scores = {**cpu_scores, **control_lora}
    control_predictions = {
        name: rank_predictions(values, frame.tie_key.astype(str))
        for name, values in control_scores.items()
    }
    candidate_predictions = {
        name: rank_predictions(values, frame.tie_key.astype(str))
        for name, values in candidate_lora.items()
    }
    control_columns = [
        column for column in manifest["feature_columns"]
        if not column.startswith("round7_")
    ]
    base_features = assemble_features(
        frame, control_scores, control_predictions, control_columns
    )
    features = augment_runtime_features(
        base_features,
        control_scores=control_lora,
        control_predictions={name: control_predictions[name] for name in control_lora},
        candidate_scores=candidate_lora,
        candidate_predictions=candidate_predictions,
        expected_columns=manifest["feature_columns"],
    )
    from catboost import CatBoostClassifier

    selector = CatBoostClassifier()
    selector.load_model(ARTIFACTS / "cpu/selector_augmented_fold01234.cbm")
    selector_scores = np.asarray(selector.predict_proba(features)[:, 1], dtype=np.float64)
    final = rank_predictions(selector_scores, frame.tie_key.astype(str))
    return final, {
        "rows": len(frame),
        "mock": bool(mock),
        "text_only_lora": True,
        "positive_quota": positive_quota(len(frame)),
        "control_positive_counts": {
            name: int(value.sum()) for name, value in control_predictions.items()
        },
        "candidate_positive_counts": {
            name: int(value.sum()) for name, value in candidate_predictions.items()
        },
        "selector_positive_count": int(final.sum()),
        "finite_selector_scores": bool(np.isfinite(selector_scores).all()),
        "lora": lora_status,
    }
