"""Deterministic matched CE and pair-channel schedules for V170."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import canonical_sha256, stable_rank


def component_class_weights(
    train: pd.DataFrame,
    *,
    target_column: str = "label",
) -> np.ndarray:
    labels = train[target_column].astype(int).to_numpy()
    if set(labels) != {0, 1}:
        raise RuntimeError("V170 CE training roster requires both classes")
    weights = np.zeros(len(train), dtype=np.float64)
    for label in (0, 1):
        class_rows = train.loc[train[target_column].eq(label)]
        counts = class_rows.groupby("component_key").size()
        for index in class_rows.index:
            component = str(train.at[index, "component_key"])
            weights[int(index)] = 0.5 / (len(counts) * int(counts.loc[component]))
    weights *= len(train)
    if (
        abs(float(weights.mean()) - 1.0) > 1e-12
        or abs(float(weights[labels == 0].sum()) - len(train) * 0.5) > 1e-8
        or abs(float(weights[labels == 1].sum()) - len(train) * 0.5) > 1e-8
    ):
        raise RuntimeError("V170 component/class loss mass drift")
    return weights.astype(np.float32)


def stratified_epoch_order(
    train: pd.DataFrame,
    *,
    effective_batch_size: int,
    seed: str,
) -> list[int]:
    if not train.index.equals(pd.RangeIndex(len(train))):
        raise RuntimeError("V170 CE schedule requires a reset RangeIndex")
    blocks = int(math.ceil(len(train) / effective_batch_size))
    positives = train.index[train.label.eq(1)].astype(int).tolist()
    negatives = train.index[train.label.eq(0)].astype(int).tolist()
    positives.sort(
        key=lambda index: stable_rank(seed, "positive-order", str(train.at[index, "blind_uid"]))
    )
    negatives.sort(
        key=lambda index: stable_rank(seed, "negative-order", str(train.at[index, "blind_uid"]))
    )
    block_sizes = [
        min(effective_batch_size, len(train) - block * effective_batch_size)
        for block in range(blocks)
    ]
    scheduled: list[list[int]] = [[] for _ in range(blocks)]
    for index in positives:
        eligible = [
            block for block in range(blocks) if len(scheduled[block]) < block_sizes[block]
        ]
        target = min(
            eligible,
            key=lambda block: (len(scheduled[block]) / block_sizes[block], block),
        )
        scheduled[target].append(index)
    cursor = 0
    for block, values in enumerate(scheduled):
        needed = block_sizes[block] - len(values)
        values.extend(negatives[cursor : cursor + needed])
        cursor += needed
        values.sort(
            key=lambda index: stable_rank(seed, f"block-{block}", str(train.at[index, "blind_uid"]))
        )
    order = [index for values in scheduled for index in values]
    if cursor != len(negatives) or len(order) != len(train) or set(order) != set(train.index):
        raise RuntimeError("V170 CE schedule is not a full-roster permutation")
    return order


def deterministic_group_order(
    frame: pd.DataFrame,
    *,
    group_column: str,
    needed: int,
    seed: str,
) -> list[int]:
    if needed <= 0:
        return []
    if frame.empty or group_column not in frame:
        raise RuntimeError("V170 pair schedule requires a nonempty grouped inventory")
    groups = {
        str(name): [int(value) for value in indices]
        for name, indices in frame.groupby(group_column, sort=True).indices.items()
    }
    rng = random.Random(seed)
    names = sorted(groups)
    queues: dict[str, list[int]] = {name: [] for name in names}
    order: list[int] = []
    while len(order) < needed:
        cycle = names.copy()
        rng.shuffle(cycle)
        for name in cycle:
            if not queues[name]:
                queues[name] = groups[name].copy()
                rng.shuffle(queues[name])
            order.append(queues[name].pop())
            if len(order) == needed:
                break
    return order


def deterministic_weighted_order(
    frame: pd.DataFrame,
    *,
    needed: int,
    seed: str,
    weight_column: str,
) -> list[int]:
    if needed <= 0:
        return []
    weights = frame[weight_column].astype(float).to_numpy()
    if len(weights) == 0 or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise RuntimeError("V170 weighted pair schedule has invalid weights")
    weights /= weights.sum()
    expected = weights * needed
    counts = np.floor(expected).astype(int)
    remaining = needed - int(counts.sum())
    rng = random.Random(seed)
    tie = list(range(len(weights)))
    rng.shuffle(tie)
    tie_rank = {index: rank for rank, index in enumerate(tie)}
    remainders = sorted(
        range(len(weights)),
        key=lambda index: (-(expected[index] - counts[index]), tie_rank[index]),
    )
    for index in remainders[:remaining]:
        counts[index] += 1
    order = [index for index, count in enumerate(counts) for _ in range(int(count))]
    rng.shuffle(order)
    if len(order) != needed:
        raise RuntimeError("V170 weighted pair schedule length drift")
    return order


def checkpoint_steps(total_steps: int, fractions: list[float]) -> dict[int, float]:
    if total_steps <= 0 or fractions != [0.25, 0.5, 1.0]:
        raise RuntimeError("V170 checkpoint-fraction contract drift")
    steps = {
        max(1, int(round(total_steps * fraction))): float(fraction)
        for fraction in fractions
    }
    if len(steps) != len(fractions) or max(steps) != total_steps:
        raise RuntimeError("V170 checkpoint steps collide or miss the final step")
    return steps


def build_schedule(
    train: pd.DataFrame,
    r4: pd.DataFrame,
    r5: pd.DataFrame,
    r7: pd.DataFrame,
    *,
    arm: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    seed = str(training["seed"])
    batch_size = int(training["effective_batch_size"])
    ce_order = stratified_epoch_order(train, effective_batch_size=batch_size, seed=seed)
    steps = int(math.ceil(len(train) / batch_size))
    r4_needed = steps * int(training["r4_batch_size"])
    r5_needed = steps * int(training["r5_batch_size"])
    r7_needed = steps * int(training["r7_batch_size"]) if arm == "candidate" else 0
    r4_order = deterministic_group_order(
        r4,
        group_column="positive_component_key",
        needed=r4_needed,
        seed=f"{seed}:r4-real-pair:positive-component",
    )
    r5_order = deterministic_group_order(
        r5,
        group_column="sampling_frame",
        needed=r5_needed,
        seed=f"{seed}:r5-synthetic-pair:style-frame",
    )
    r7_order = deterministic_weighted_order(
        r7,
        needed=r7_needed,
        seed=f"{seed}:r7-real-pair:mass",
        weight_column="normalized_pair_weight",
    )
    checkpoints = checkpoint_steps(steps, list(training["checkpoint_fractions"]))
    result: dict[str, Any] = {
        "ce_order": ce_order,
        "r4_order": r4_order,
        "r5_order": r5_order,
        "r7_order": r7_order,
        "optimizer_steps": steps,
        "warmup_steps": int(math.ceil(steps * float(training["warmup_ratio"]))),
        "checkpoint_steps": {str(step): fraction for step, fraction in checkpoints.items()},
        "ce_order_sha256": canonical_sha256(ce_order),
        "r4_order_sha256": canonical_sha256(r4_order),
        "r5_order_sha256": canonical_sha256(r5_order),
        "r7_order_sha256": canonical_sha256(r7_order),
        "r4_draws": len(r4_order),
        "r5_draws": len(r5_order),
        "r7_draws": len(r7_order),
        "r4_group_draw_min": min(Counter(r4.iloc[r4_order].positive_component_key).values()),
        "r4_group_draw_max": max(Counter(r4.iloc[r4_order].positive_component_key).values()),
        "r5_group_draw_min": min(Counter(r5.iloc[r5_order].sampling_frame).values()),
        "r5_group_draw_max": max(Counter(r5.iloc[r5_order].sampling_frame).values()),
    }
    if r7_order:
        r7_counts = Counter(r7_order)
        result["r7_pair_draw_min"] = min(r7_counts.values())
        result["r7_pair_draw_max"] = max(r7_counts.values())
    result["schedule_sha256"] = canonical_sha256(result)
    return result


def schedule_serializable_view(schedule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in schedule.items()
        if not key.endswith("_order")
    }

