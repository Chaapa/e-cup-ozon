#!/usr/bin/env python3
"""Target-free feature augmentation shared by fit and inference runtime."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRAJECTORY = ("lora025", "lora050", "lora100")
COMPONENT_RADIUS_THRESHOLD = 0.90


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _union_equal_keys(union: _UnionFind, keys: Iterable[str]) -> None:
    first: dict[str, int] = {}
    for position, raw_key in enumerate(keys):
        key = str(raw_key).strip()
        if not key:
            continue
        if key in first:
            union.union(position, first[key])
        else:
            first[key] = position


def attach_runtime_component_size(
    frame: pd.DataFrame,
    *,
    threshold: float = COMPONENT_RADIUS_THRESHOLD,
    block_size: int = 256,
) -> pd.DataFrame:
    """Rebuild the target-free V108 duplicate component size on a query roster."""

    required = {"normalized_name", "exact_text_key", "gallery_signature"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"V115 component inputs missing: {sorted(missing)}")
    output = frame.copy()
    if output.empty:
        output["component_size"] = pd.Series(index=output.index, dtype=np.int64)
        return output
    if not 0.0 < float(threshold) <= 1.0 or int(block_size) <= 0:
        raise ValueError("Invalid V115 component radius configuration")

    union = _UnionFind(len(output))
    _union_equal_keys(union, output.exact_text_key.fillna("").astype(str))
    _union_equal_keys(union, output.gallery_signature.fillna("").astype(str))

    # Lazy import keeps the module importable before runtime dependencies load.
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60_000,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )
    names = output.normalized_name.fillna("").astype(str)
    try:
        matrix = vectorizer.fit_transform(names)
    except ValueError as exc:
        # A small or all-empty roster can have no vocabulary after min_df=2;
        # then there are no radius edges and the exact-key graph is complete.
        message = str(exc).lower()
        if not any(
            marker in message
            for marker in ("empty vocabulary", "max_df corresponds", "no terms remain")
        ):
            raise
        matrix = None
    if matrix is not None:
        for start in range(0, matrix.shape[0], int(block_size)):
            stop = min(matrix.shape[0], start + int(block_size))
            similarities = (matrix[start:stop] @ matrix.T).tocsr()
            for local in range(stop - start):
                left = start + local
                begin, end = similarities.indptr[local], similarities.indptr[local + 1]
                for right, similarity in zip(
                    similarities.indices[begin:end],
                    similarities.data[begin:end],
                    strict=True,
                ):
                    right = int(right)
                    if right <= left or float(similarity) + 1e-7 < float(threshold):
                        continue
                    union.union(left, right)

    roots = [union.find(position) for position in range(len(output))]
    counts = Counter(roots)
    output["component_size"] = np.asarray([counts[root] for root in roots], dtype=np.int64)
    if (output.component_size < 1).any():
        raise RuntimeError("V115 component reconstruction produced an invalid size")
    return output


def percentile(values: pd.Series) -> pd.Series:
    return values.astype(float).rank(method="average", pct=True)


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = float(values.std())
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return (values - float(values.mean())) / std


def augment_runtime_features(
    base_features: pd.DataFrame,
    *,
    control_scores: Mapping[str, Sequence[float]],
    control_predictions: Mapping[str, Sequence[int]],
    candidate_scores: Mapping[str, Sequence[float]],
    candidate_predictions: Mapping[str, Sequence[int]],
    expected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Append the exact 32 candidate/control trajectory features used in OOF."""

    output = base_features.copy()
    row_count = len(output)
    delta_rank_columns: list[str] = []
    delta_z_columns: list[str] = []
    candidate_prediction_columns: list[str] = []
    for name in TRAJECTORY:
        candidate = pd.Series(
            np.asarray(candidate_scores[name], dtype=np.float64), index=output.index
        )
        control = pd.Series(
            np.asarray(control_scores[name], dtype=np.float64), index=output.index
        )
        candidate_prediction = np.asarray(candidate_predictions[name], dtype=np.int8)
        control_prediction = np.asarray(control_predictions[name], dtype=np.int8)
        if any(len(value) != row_count for value in (
            candidate, control, candidate_prediction, control_prediction
        )):
            raise RuntimeError("V115 runtime trajectory row-count drift")
        candidate_rank = percentile(candidate)
        control_rank = percentile(control)
        candidate_z = zscore(candidate)
        control_z = zscore(control)
        prefix = f"round7_{name}"
        output[f"{prefix}_score"] = candidate
        output[f"{prefix}_rank"] = candidate_rank
        output[f"{prefix}_z"] = candidate_z
        output[f"{prefix}_prediction"] = candidate_prediction
        output[f"{prefix}_delta_score"] = candidate - control
        output[f"{prefix}_delta_rank"] = candidate_rank - control_rank
        output[f"{prefix}_delta_z"] = candidate_z - control_z
        output[f"{prefix}_disagrees_control"] = (
            candidate_prediction != control_prediction
        ).astype(np.int8)
        delta_rank_columns.append(f"{prefix}_delta_rank")
        delta_z_columns.append(f"{prefix}_delta_z")
        candidate_prediction_columns.append(f"{prefix}_prediction")
    output["round7_vote_count"] = output[candidate_prediction_columns].sum(axis=1)
    output["round7_control_disagreement_count"] = output[
        [f"round7_{name}_disagrees_control" for name in TRAJECTORY]
    ].sum(axis=1)
    for prefix, columns in (("rank", delta_rank_columns), ("z", delta_z_columns)):
        output[f"round7_delta_{prefix}_mean"] = output[columns].mean(axis=1)
        output[f"round7_delta_{prefix}_min"] = output[columns].min(axis=1)
        output[f"round7_delta_{prefix}_max"] = output[columns].max(axis=1)
    if (
        len(output) != row_count
        or output.isna().any().any()
        or not np.isfinite(output.to_numpy(dtype=float)).all()
    ):
        raise RuntimeError("V115 augmented runtime features are invalid")
    if expected_columns is not None:
        expected = list(expected_columns)
        if set(output.columns) != set(expected) or len(expected) != len(set(expected)):
            raise RuntimeError("V115 augmented runtime feature schema drift")
        output = output[expected]
    return output
