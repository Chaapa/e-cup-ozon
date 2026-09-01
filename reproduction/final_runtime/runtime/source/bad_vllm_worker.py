#!/usr/bin/env python3
"""Isolated ordered-batch vLLM worker for the BAD branch."""

from __future__ import annotations

import argparse
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import shutil
import time

from src.reasoning_comments import (
    ALLOWED_A,
    ALLOWED_B,
    CRITIC_PROMPT,
    PROMPT_A,
    PROMPT_B,
    SELECTOR_PROMPT,
    clean,
    fallback_comment,
    policy_sha256,
    render,
    select_candidate,
    user_prompt,
    valid_comment,
    weighted_score,
)


VLLM_VERSION = "0.26.0+cu129"
ZERO_TOKEN_ID = 15
ONE_TOKEN_ID = 16


def configure_runtime_caches(runtime_root: Path) -> None:
    """Route JIT/runtime caches to writable storage."""
    cache_dirs = {
        "VLLM_CACHE_ROOT": runtime_root / "vllm",
        "FLASHINFER_WORKSPACE_BASE": runtime_root / "flashinfer-base",
        "TRITON_CACHE_DIR": runtime_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": runtime_root / "torchinductor",
        "TORCH_EXTENSIONS_DIR": runtime_root / "torch-extensions",
        "CUDA_CACHE_PATH": runtime_root / "cuda",
        "NUMBA_CACHE_DIR": runtime_root / "numba",
        "XDG_CACHE_HOME": runtime_root / "xdg",
    }
    for variable, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)

    prewarmed = Path(
        os.environ.get("VLLM_PREWARMED_CACHE_ROOT", "/opt/vllm-prewarmed-cache")
    ) / "modelinfos"
    runtime_modelinfos = cache_dirs["VLLM_CACHE_ROOT"] / "modelinfos"
    if prewarmed.is_dir():
        shutil.copytree(prewarmed, runtime_modelinfos, dirs_exist_ok=True)


def parse_json(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value.strip())
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def basis_schema(allowed: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "evidence_id": {"type": "integer", "minimum": 0, "maximum": 63},
            "basis": {"type": "string", "enum": list(allowed)},
        },
        "required": ["evidence_id", "basis"],
        "additionalProperties": False,
    }


def generate_reason_candidates(engine, rows: list[dict[str, object]], *, candidate: str):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompt = PROMPT_A if candidate == "A" else PROMPT_B
    allowed_map = ALLOWED_A if candidate == "A" else ALLOWED_B
    prepared = [user_prompt(row, candidate=candidate) for row in rows]
    raw_outputs: list[str] = [""] * len(rows)
    for group_key, allowed in allowed_map.items():
        indices = [
            index
            for index, row in enumerate(rows)
            if (str(row["category"]), int(row["label"])) == group_key
        ]
        if not indices:
            continue
        parameters = SamplingParams(
            temperature=0,
            max_tokens=80,
            seed=0,
            structured_outputs=StructuredOutputsParams(json=basis_schema(allowed)),
        )
        outputs = engine.chat(
            [
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": prepared[index][0]},
                ]
                for index in indices
            ],
            sampling_params=parameters,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        for index, output in zip(indices, outputs, strict=True):
            raw_outputs[index] = output.outputs[0].text
    records = []
    for row, prepared_row, raw in zip(rows, prepared, raw_outputs, strict=True):
        candidates = prepared_row[1]
        evidence_id, basis, repaired = select_candidate(
            row, candidates, parse_json(raw), candidate=candidate
        )
        records.append(
            {
                "evidence_id": evidence_id,
                "basis": basis,
                "evidence": candidates[evidence_id],
                "comment": render(candidates[evidence_id], basis, candidate=candidate),
                "repaired": repaired,
                "raw": raw,
                "prompt": prepared_row[0],
                "candidates": candidates,
            }
        )
    return records


def critic_candidate_b(engine, rows, candidate_b):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    for group_key, allowed in ALLOWED_B.items():
        indices = [
            index
            for index, (row, candidate) in enumerate(zip(rows, candidate_b, strict=True))
            if (str(row["category"]), int(row["label"])) == group_key
            and bool(candidate["repaired"])
        ]
        if not indices:
            continue
        schema = basis_schema(allowed)
        schema["properties"]["action"] = {"type": "string", "enum": ["keep", "replace"]}
        schema["required"] = ["action", "evidence_id", "basis"]
        parameters = SamplingParams(
            temperature=0,
            max_tokens=90,
            seed=1,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        conversations = []
        for index in indices:
            item = candidate_b[index]
            prompt = (
                f"{item['prompt']}\nПредложенная пара:\n"
                f"evidence_id={item['evidence_id']}\nevidence={item['evidence']}\n"
                f"basis={item['basis']}"
            )
            conversations.append(
                [
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        outputs = engine.chat(
            conversations,
            sampling_params=parameters,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        for index, output in zip(indices, outputs, strict=True):
            parsed = parse_json(output.outputs[0].text)
            evidence_id = int(parsed.get("evidence_id", -1))
            basis = str(parsed.get("basis") or "")
            candidates = candidate_b[index]["candidates"]
            if (
                parsed.get("action") == "replace"
                and 0 <= evidence_id < len(candidates)
                and basis in allowed
                and weighted_score(basis, candidates[evidence_id]) > 0
            ):
                evidence = candidates[evidence_id]
                candidate_b[index].update(
                    evidence_id=evidence_id,
                    basis=basis,
                    evidence=evidence,
                    comment=render(evidence, basis, candidate="B"),
                    repaired=True,
                )


def select_reason_comments(engine, rows, candidate_a, candidate_b):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    parameters = SamplingParams(
        temperature=0,
        max_tokens=30,
        seed=2,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    conversations = []
    for row, first, second in zip(rows, candidate_a, candidate_b, strict=True):
        verdict = "соответствует категории" if int(row["label"]) else "не соответствует категории"
        payload = (
            f"Категория: {row['category']}\nЗафиксированный итог: товар {verdict}.\n"
            f"Название: {clean(row.get('name'), 900) or '[нет]'}\n"
            f"Описание: {clean(row.get('description'), 7000) or '[нет]'}\n"
            f"A: {json.dumps({'evidence': first['evidence'], 'comment': first['comment']}, ensure_ascii=False)}\n"
            f"B: {json.dumps({'evidence': second['evidence'], 'comment': second['comment']}, ensure_ascii=False)}"
        )
        conversations.append(
            [
                {"role": "system", "content": SELECTOR_PROMPT},
                {"role": "user", "content": payload},
            ]
        )
    outputs = engine.chat(
        conversations,
        sampling_params=parameters,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    comments = []
    for row, first, second, output in zip(rows, candidate_a, candidate_b, outputs, strict=True):
        choice = str(parse_json(output.outputs[0].text).get("choice") or "A")
        selected = second if choice == "B" else first
        comment = str(selected["comment"])
        evidence = str(selected["evidence"])
        if not valid_comment(row, evidence, comment):
            comment = fallback_comment(str(row["category"]), int(row["label"]))
            choice = "fallback"
        comments.append({"comment": comment, "source": choice})
    return comments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_runtime_caches(args.output.parent / ".runtime_cache")
    observed_version = package_version("vllm")
    if observed_version != VLLM_VERSION:
        raise RuntimeError(
            f"vLLM version drift: expected {VLLM_VERSION}, found {observed_version}"
        )
    request_payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(request_payload, dict):
        raise RuntimeError("BAD worker request payload must be an object")
    requests = request_payload.get("bad_requests")
    reason_rows = request_payload.get("reason_rows")
    threshold = float(request_payload.get("sum_margin_threshold"))
    if not isinstance(requests, list) or not requests:
        raise RuntimeError("Ordered BAD request batch is empty")
    if not isinstance(reason_rows, list) or not reason_rows:
        raise RuntimeError("Ordered reason row batch is empty")
    conversations = [item["messages"] for item in requests]
    if [int(item["position"]) for item in requests] != list(range(len(requests))):
        raise RuntimeError("Ordered BAD request positions drift")

    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    engine = LLM(
        model=str(args.model),
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        max_model_len=12_288,
        gpu_memory_utilization=0.92,
        disable_log_stats=True,
    )
    startup_seconds = time.perf_counter() - started
    parameters = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=[ZERO_TOKEN_ID, ONE_TOKEN_ID],
        seed=0,
    )
    inference_started = time.perf_counter()
    outputs = engine.chat(
        conversations,
        sampling_params=parameters,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    inference_seconds = time.perf_counter() - inference_started
    if len(outputs) != len(requests):
        raise RuntimeError("Ordered BAD batch lost outputs")
    records = []
    for request, output in zip(requests, outputs, strict=True):
        completion = output.outputs[0]
        first = completion.logprobs[0]
        if ZERO_TOKEN_ID not in first or ONE_TOKEN_ID not in first:
            raise RuntimeError("Binary token missing from vLLM logprobs")
        logp0 = float(first[ZERO_TOKEN_ID].logprob)
        logp1 = float(first[ONE_TOKEN_ID].logprob)
        records.append(
            {
                "position": int(request["position"]),
                "branch": str(request["branch"]),
                "row_position": int(request["row_position"]),
                "margin": logp1 - logp0,
            }
        )
    strict = [None] * (len(requests) // 2)
    broad = [None] * (len(requests) // 2)
    for record in records:
        destination = strict if record["branch"] == "strict" else broad
        destination[int(record["row_position"])] = float(record["margin"])
    bad_labels = []
    for index, row in enumerate(request_payload.get("bad_rows") or []):
        if strict[index] is None or broad[index] is None:
            raise RuntimeError("BAD margin reconstruction failed")
        label = int(float(strict[index]) + float(broad[index]) >= threshold)
        if bool(row.get("strict_negation")):
            label = 0
        bad_labels.append(label)
    if len(bad_labels) != len(strict):
        raise RuntimeError("BAD label derivation lost rows")
    for row in reason_rows:
        if str(row["category"]) == "БАД":
            row["label"] = bad_labels[int(row["row_position"])]
        else:
            row["label"] = int(row["label"])

    reason_started = time.perf_counter()
    candidate_a = generate_reason_candidates(engine, reason_rows, candidate="A")
    candidate_b = generate_reason_candidates(engine, reason_rows, candidate="B")
    critic_candidate_b(engine, reason_rows, candidate_b)
    selected_comments = select_reason_comments(engine, reason_rows, candidate_a, candidate_b)
    reason_seconds = time.perf_counter() - reason_started
    comment_records = []
    for row, selected in zip(reason_rows, selected_comments, strict=True):
        comment_records.append(
            {
                "category": str(row["category"]),
                "row_position": int(row["row_position"]),
                "label": int(row["label"]),
                "comment": str(selected["comment"]),
                "source": str(selected["source"]),
            }
        )

    payload = {
        "status": "ORDERED_VLLM_BAD_AND_COMMENTS_COMPLETE",
        "vllm_version": observed_version,
        "request_count": len(records),
        "startup_seconds": startup_seconds,
        "inference_seconds": inference_seconds,
        "reason_seconds": reason_seconds,
        "comment_count": len(comment_records),
        "comment_source_counts": {
            source: sum(item["source"] == source for item in comment_records)
            for source in ("A", "B", "fallback")
        },
        "reason_policy_sha256": policy_sha256(),
        "records": records,
        "comments": comment_records,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"records", "comments"}},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
