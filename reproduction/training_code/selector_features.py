"""Construct the fixed target-free 71-feature selector matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAJECTORY = ("lora025", "lora050", "lora100")
BASES = ("b1", "char", *TRAJECTORY)


def _percentile(values: pd.Series, folds: pd.Series) -> pd.Series:
    return values.astype(float).groupby(folds).rank(method="average", pct=True)


def _zscore(values: pd.Series, folds: pd.Series) -> pd.Series:
    values = values.astype(float)
    mean = values.groupby(folds).transform("mean")
    std = values.groupby(folds).transform("std").replace(0, 1.0).fillna(1.0)
    return (values - mean) / std


def build_base_features(
    rows: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = {f"{base}_{suffix}" for base in BASES for suffix in ("score", "prediction")}
    if not required.issubset(signals.columns):
        raise RuntimeError(f"V170 selector signals missing {sorted(required - set(signals.columns))}")
    merged = rows.merge(signals, on="id", validate="one_to_one")
    if len(merged) != len(rows):
        raise RuntimeError("V170 selector signal merge lost rows")
    features = pd.DataFrame(index=merged.index)
    for base in BASES:
        score = merged[f"{base}_score"].astype(float)
        features[f"{base}_score"] = score
        features[f"{base}_rank"] = _percentile(score, merged.fold)
        features[f"{base}_z"] = _zscore(score, merged.fold)
        features[f"{base}_prediction"] = merged[f"{base}_prediction"].astype(np.int8)
    score_columns = [f"{base}_z" for base in BASES]
    rank_columns = [f"{base}_rank" for base in BASES]
    prediction_columns = [f"{base}_prediction" for base in BASES]
    features["vote_count"] = features[prediction_columns].sum(axis=1)
    for prefix, columns in (("z", score_columns), ("rank", rank_columns)):
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
    features["component_log1p"] = np.log1p(merged.component_size.astype(float))
    features["image_count"] = merged.image_count.astype(float)
    features["name_chars_log1p"] = np.log1p(
        merged.clean_name.fillna("").astype(str).str.len()
    )
    features["description_chars_log1p"] = np.log1p(
        merged.clean_description.fillna("").astype(str).str.len()
    )
    metadata = merged[["id", "blind_uid", "component_key", "fold", "label"]].copy()
    output = pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    columns = list(features.columns)
    if len(columns) != 39:
        raise RuntimeError(f"V170 base selector feature-count drift: {len(columns)}")
    return output, columns


def build_augmented_features(
    rows: pd.DataFrame,
    control_signals: pd.DataFrame,
    candidate_signals: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    matrix, columns = build_base_features(rows, control_signals)
    source = rows[["id", "fold"]].merge(
        candidate_signals,
        on="id",
        validate="one_to_one",
    ).merge(
        control_signals,
        on="id",
        validate="one_to_one",
        suffixes=("_candidate", "_control"),
    )
    extras = source[["id"]].copy()
    delta_rank_columns: list[str] = []
    delta_z_columns: list[str] = []
    candidate_prediction_columns: list[str] = []
    for name in TRAJECTORY:
        candidate = source[f"{name}_score_candidate"].astype(float)
        control = source[f"{name}_score_control"].astype(float)
        candidate_rank = _percentile(candidate, source.fold)
        control_rank = _percentile(control, source.fold)
        candidate_z = _zscore(candidate, source.fold)
        control_z = _zscore(control, source.fold)
        candidate_prediction = source[f"{name}_prediction_candidate"].astype(np.int8)
        control_prediction = source[f"{name}_prediction_control"].astype(np.int8)
        prefix = f"round7_{name}"
        extras[f"{prefix}_score"] = candidate
        extras[f"{prefix}_rank"] = candidate_rank
        extras[f"{prefix}_z"] = candidate_z
        extras[f"{prefix}_prediction"] = candidate_prediction
        extras[f"{prefix}_delta_score"] = candidate - control
        extras[f"{prefix}_delta_rank"] = candidate_rank - control_rank
        extras[f"{prefix}_delta_z"] = candidate_z - control_z
        extras[f"{prefix}_disagrees_control"] = (
            candidate_prediction != control_prediction
        ).astype(np.int8)
        delta_rank_columns.append(f"{prefix}_delta_rank")
        delta_z_columns.append(f"{prefix}_delta_z")
        candidate_prediction_columns.append(f"{prefix}_prediction")
    extras["round7_vote_count"] = extras[candidate_prediction_columns].sum(axis=1)
    extras["round7_control_disagreement_count"] = extras[
        [f"round7_{name}_disagrees_control" for name in TRAJECTORY]
    ].sum(axis=1)
    for prefix, selected in (("rank", delta_rank_columns), ("z", delta_z_columns)):
        extras[f"round7_delta_{prefix}_mean"] = extras[selected].mean(axis=1)
        extras[f"round7_delta_{prefix}_min"] = extras[selected].min(axis=1)
        extras[f"round7_delta_{prefix}_max"] = extras[selected].max(axis=1)
    extra_columns = [column for column in extras if column != "id"]
    matrix = matrix.merge(extras, on="id", validate="one_to_one")
    columns.extend(extra_columns)
    if (
        len(columns) != 71
        or len(set(columns)) != len(columns)
        or any(column.startswith("rule_") for column in columns)
        or matrix[columns].isna().any().any()
        or not np.isfinite(matrix[columns].to_numpy(dtype=float)).all()
    ):
        raise RuntimeError("V170 augmented selector 71-feature contract failed")
    return matrix, columns


def component_class_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)
    for label in (0, 1):
        subset = frame.loc[frame.label.astype(int).eq(label)]
        components = subset.groupby("component_key").size()
        for index in subset.index:
            key = str(frame.at[index, "component_key"])
            weights[int(index)] = 0.5 / (len(components) * int(components.loc[key]))
    weights *= len(frame)
    if abs(float(weights.mean()) - 1.0) > 1e-12:
        raise RuntimeError("V170 selector component/class weights drift")
    return weights
