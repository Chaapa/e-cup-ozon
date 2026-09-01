"""OpenRouter client and deterministic artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests


KEY_ENV = "OPENROUTER_API_KEY"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(dict(row)) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["model"]["parameter_count"] > 400_000_000_000:
        raise RuntimeError("OpenRouter model exceeds the 400B parameter ceiling")
    if config["budget"]["hard_cap_usd"] != 10.0:
        raise RuntimeError("OpenRouter hard budget cap drift")
    return config


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def request_body(
    config: Mapping[str, Any],
    system: str,
    user: str,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    model = config["model"]
    provider: dict[str, Any] = {
        "order": [model["provider"]],
        "only": [model["provider"]],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    if model.get("quantization"):
        provider["quantizations"] = [model["quantization"]]
    return {
        "model": model["openrouter_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "provider": provider,
    }


def estimated_cost(body: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    chars = sum(len(str(message["content"])) for message in body["messages"])
    input_tokens = math.ceil(chars / 3.2)
    output_tokens = int(body["max_tokens"])
    model = config["model"]
    return (
        input_tokens * model["prompt_usd_per_million_tokens"] / 1e6
        + output_tokens * model["completion_usd_per_million_tokens"] / 1e6
    )


def extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith(chr(96) * 3):
        text = re.sub(r"^`{3}(?:json)?\s*|\s*`{3}$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        begin, end = text.find("{"), text.rfind("}")
        if begin < 0 or end <= begin:
            raise
        value = json.loads(text[begin : end + 1])
    if not isinstance(value, dict):
        raise ValueError("OpenRouter response is not a JSON object")
    return value


@dataclass
class OpenRouterRunner:
    config: Mapping[str, Any]
    output: Path
    stage: str
    cap: float
    spent: float = 0.0

    def complete(
        self, body: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_hash = sha256_text(canonical_json(body))
        cache = self.output / "cache" / f"{request_hash}.json"
        if cache.exists():
            raw = json.loads(cache.read_text(encoding="utf-8"))
            return extract_json(raw["choices"][0]["message"]["content"]), raw
        projection = estimated_cost(body, self.config)
        hard_cap = float(self.config["budget"]["hard_cap_usd"])
        if self.spent + projection > self.cap or self.spent + projection > hard_cap:
            raise RuntimeError(f"{self.stage} projected cost exceeds hard cap")
        key = os.environ.get(KEY_ENV)
        if not key:
            raise RuntimeError(f"{KEY_ENV} is not set")
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=240,
        )
        response.raise_for_status()
        raw = response.json()
        if not raw.get("choices"):
            raise RuntimeError(
                f"OpenRouter returned no choices: {raw.get('error', 'unknown error')}"
            )
        usage = raw.get("usage") or {}
        actual = usage.get("cost")
        if actual is None:
            model = self.config["model"]
            actual = (
                float(usage.get("prompt_tokens", 0))
                * model["prompt_usd_per_million_tokens"]
                + float(usage.get("completion_tokens", 0))
                * model["completion_usd_per_million_tokens"]
            ) / 1e6
        self.spent += float(actual)
        write_json(cache, raw)
        return extract_json(raw["choices"][0]["message"]["content"]), raw


def exact_overlap_count(
    synthetic: list[Mapping[str, Any]], rows: pd.DataFrame
) -> int:
    def norm(value: object) -> str:
        return re.sub(r"[^0-9a-zа-яё]+", " ", str(value).lower()).strip()

    originals = {
        norm(name) + "\n" + norm(description)
        for name, description in zip(rows.clean_name, rows.clean_description)
    }
    count = 0
    for item in synthetic:
        for side in ("positive", "negative"):
            count += int(
                norm(item[f"{side}_name"])
                + "\n"
                + norm(item[f"{side}_description"])
                in originals
            )
    return count

