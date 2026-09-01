#!/usr/bin/env python3
"""Standalone BAD and FLV inference runtime."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "vendor_runtime_linux"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")

from src.constants import (  # noqa: E402
    BAD,
    BASE_MODEL,
    BASE_REVISION,
    FLAMMABLE,
    QWEN_MODEL,
)
from src.postprocess import (  # noqa: E402
    format_flammable_result,
    validate_output,
    verdict_label,
)
from src.reasoning_comments import policy_sha256 as reason_policy_sha256  # noqa: E402
from src.v108_sheets import load_source, prepare_flammable  # noqa: E402
from src.v116_model import predict_flammable, production_manifest  # noqa: E402


PACKAGE_ID = "ecup_quality_inference_runtime"
STRICT_PROMPT_SHA256 = "55d5c93e3e4b1fad6a437f2c099648bd5bec710145c717fbc085df38b7199088"
BROAD_PROMPT_SHA256 = "3efd82f802cc72fc8c9628018d34e318705a4e3c297432a01ba785198dd32904"
SUM_MARGIN_THRESHOLD = 0.6875
VLLM_VERSION = "0.26.0+cu129"
REASON_POLICY_SHA256 = "665f275706c4dca702e1d58760cf6c6ae416503b910a190b85e675711ff19c12"


STRICT_PROMPT = """Определи метку текущего продаваемого товара для категории «БАД» по конвенции организаторов.

Сначала мысленно установи, что именно физически продаётся и к какому объекту относятся слова в тексте. Затем примени правила по порядку.

Метка 1: сам текущий товар прямо обозначен как БАД, биологически активная добавка, dietary supplement или имеет Supplement Facts. Полная формула статуса важнее общих слов о форме и пользе.

Метка 0: у самого текущего товара такого статуса нет; формулировка относится к другому товару, производителю, исследованию, рекомендации, будущему содержимому или SEO-перечню; либо текущий товар прежде всего является спортивным питанием, обычной едой/напитком, сырьём/ингредиентом, наружным средством, устройством, упаковкой или пустым носителем без собственного прямого supplement-статуса.

Не приравнивай к БАД слова «витамины», «капсулы», «порошок», пользу для здоровья, дозировку, GMP, Non-GMO, vegan, kosher, «пищевая добавка» или «не лекарство». Но и спортивный контекст сам по себе не отменяет явный прямой статус БАД текущего товара.

При конфликте сначала учитывай явное отрицание статуса, затем первичную сущность продаваемого объекта, затем прямую маркировку именно этого объекта. Не показывай рассуждение. Ответь ровно одним символом: 0 или 1."""


BROAD_PROMPT = """Классифицируй текущий продаваемый товар для категории «БАД» по конвенции организаторов. Ответ должен отражать статус и первичную сущность именно продаваемого объекта, а не случайные слова в карточке.

Применяй решение в таком порядке.

1. Если про текущий товар прямо сказано «не БАД», «не является БАД» или «не является лекарством и БАД», ставь 0. Это отрицание сильнее рекламных слов, капсул, дозировки и пользы.

2. Ставь 1, если текущий товар прямо назван БАД, биологически активной добавкой, dietary supplement, имеет Supplement Facts либо представляет собой самостоятельный предназначенный для приёма внутрь supplement-продукт: дозированные витамины, минералы, аминокислоты, жирные кислоты, растительный экстракт или иной активный комплекс в капсулах, таблетках, порциях, каплях или с указанной схемой приёма. Для такого продукта официальная русская формула может отсутствовать.

3. Ставь 0, если спорные слова относятся к другому товару, производителю, исследованию, рекомендации, будущему содержимому или SEO-перечню; либо первичный продаваемый объект — спортивное питание/предтренировочный или белковый продукт, обычная еда или напиток, сырьё, наружное средство, устройство, упаковка или пустой носитель без собственного прямого статуса БАД.

«Пищевая добавка», польза для здоровья, дозировка, GMP, Non-GMO, vegan, kosher и «не лекарство» по отдельности не доказывают БАД. Спортивный контекст по отдельности не отменяет прямой статус БАД, но явная первичная сущность спортивного питания без такого статуса означает 0.

Мысленно проверь продаваемый объект, область действия маркировки и наличие явного отрицания. Не показывай рассуждение. Ответь ровно одним символом: 0 или 1."""


STRICT_NEGATION_PATTERN = re.compile(
    r"не\s+(?:является|считается|относится)(?:\s+[а-яa-z-]+){0,6}\s+(?:и|или)\s+бад\b"
    r"|не\s+(?:является|считается|относится)\s+бад\b|\bне\s+бад\b"
)


BAD_COMMENTS = {
    1: (
        "Карточка подтверждает соответствие правилам категории БАД; приоритетные "
        "исключения для спортивного питания или прямого отрицания не установлены."
    ),
    0: (
        "Карточка не подтверждает обязательную маркировку БАД либо содержит "
        "приоритетное исключение для спортивного питания или прямого отрицания."
    ),
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_frozen_policy() -> None:
    if BASE_MODEL != "Qwen/Qwen3.5-4B":
        raise RuntimeError("Submission base model drift")
    if BASE_REVISION != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise RuntimeError("Submission base revision drift")
    if sha256_text(STRICT_PROMPT) != STRICT_PROMPT_SHA256:
        raise RuntimeError("Submission strict prompt drift")
    if sha256_text(BROAD_PROMPT) != BROAD_PROMPT_SHA256:
        raise RuntimeError("Submission broad prompt drift")
    if SUM_MARGIN_THRESHOLD != 0.6875:
        raise RuntimeError("Submission BAD threshold drift")
    if reason_policy_sha256() != REASON_POLICY_SHA256:
        raise RuntimeError("Submission reason policy drift")


def normalize_text(value: object) -> str:
    text = html.unescape(str(value or "")).lower().replace("ё", "е")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_strict_bad_negation(name: object, description: object) -> bool:
    text = f"{normalize_text(name)} {normalize_text(description)}".strip()
    return bool(STRICT_NEGATION_PATTERN.search(text))


def bad_user_prompt(name: object, description: object) -> str:
    title = str(name or "").strip() or "[нет]"
    body = str(description or "").strip() or "[нет]"
    if len(body) > 9000:
        body = body[:6500] + "\n[середина длинного текста опущена]\n" + body[-2500:]
    return f"Название: {title[:1000]}\nОписание: {body}\nИтоговая метка:"


def _mock_margins(rows: pd.DataFrame, namespace: str) -> np.ndarray:
    return np.asarray(
        [
            (
                int(
                    hashlib.sha256(
                        f"{namespace}\0{name}\0{description}".encode("utf-8")
                    ).hexdigest()[:16],
                    16,
                )
                / (2**64 - 1)
            )
            * 2
            - 1
            for name, description in zip(rows.name, rows.description, strict=True)
        ],
        dtype=np.float64,
    )


def _predict_bad_vllm(
    rows: pd.DataFrame,
    flv_rows: pd.DataFrame,
    flv_labels: np.ndarray,
    workspace: Path,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    request_path = workspace / "vllm_bad_requests.json"
    result_path = workspace / "vllm_bad_results.json"
    log_path = workspace / "vllm_bad_worker.log"
    requests: list[dict[str, Any]] = []
    for branch, prompt in (("strict", STRICT_PROMPT), ("broad", BROAD_PROMPT)):
        for row_position, row in rows.iterrows():
            requests.append(
                {
                    "position": len(requests),
                    "branch": branch,
                    "row_position": int(row_position),
                    "messages": [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": bad_user_prompt(row["name"], row["description"]),
                        },
                    ],
                }
            )
    bad_rows = [
        {
            "row_position": int(position),
            "name": str(row["name"] or ""),
            "description": str(row["description"] or ""),
            "strict_negation": contains_strict_bad_negation(row["name"], row["description"]),
        }
        for position, row in rows.iterrows()
    ]
    reason_rows = [
        {
            "category": BAD,
            "row_position": int(position),
            "name": str(row["name"] or ""),
            "description": str(row["description"] or ""),
            "label": None,
        }
        for position, row in rows.iterrows()
    ]
    reason_rows.extend(
        {
            "category": FLAMMABLE,
            "row_position": int(position),
            "name": str(row["name"] or ""),
            "description": str(row["description"] or ""),
            "label": int(label),
        }
        for (position, row), label in zip(flv_rows.iterrows(), flv_labels, strict=True)
    )
    worker_request = {
        "bad_requests": requests,
        "bad_rows": bad_rows,
        "reason_rows": reason_rows,
        "sum_margin_threshold": SUM_MARGIN_THRESHOLD,
    }
    request_path.write_text(
        json.dumps(worker_request, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / "bad_vllm_worker.py"),
        "--model",
        str(QWEN_MODEL),
        "--input",
        str(request_path),
        "--output",
        str(result_path),
    ]
    started = time.perf_counter()
    with log_path.open("wb") as log_handle:
        process = subprocess.run(
            command, check=False,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    if process.returncode != 0 or not result_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"Ordered vLLM BAD worker failed ({process.returncode}): {tail}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    comments = payload.get("comments")
    if (
        payload.get("status") != "ORDERED_VLLM_BAD_AND_COMMENTS_COMPLETE"
        or payload.get("vllm_version") != VLLM_VERSION
        or payload.get("reason_policy_sha256") != REASON_POLICY_SHA256
        or not isinstance(records, list)
        or len(records) != 2 * len(rows)
        or not isinstance(comments, list)
        or len(comments) != len(rows) + len(flv_rows)
    ):
        raise RuntimeError("Ordered vLLM BAD result contract drift")
    strict = np.full(len(rows), np.nan, dtype=np.float64)
    broad = np.full(len(rows), np.nan, dtype=np.float64)
    for expected_position, record in enumerate(records):
        if int(record["position"]) != expected_position:
            raise RuntimeError("Ordered vLLM BAD output order drift")
        destination = strict if record["branch"] == "strict" else broad
        destination[int(record["row_position"])] = float(record["margin"])
    if not np.isfinite(strict).all() or not np.isfinite(broad).all():
        raise RuntimeError("Ordered vLLM BAD margins are incomplete")
    comment_by_position: dict[tuple[str, int], dict[str, Any]] = {}
    for item in comments:
        key = (str(item["category"]), int(item["row_position"]))
        if key in comment_by_position:
            raise RuntimeError("Ordered vLLM comments contain duplicate positions")
        comment_by_position[key] = {
            "label": int(item["label"]),
            "comment": str(item["comment"]),
            "source": str(item["source"]),
        }
    return strict, broad, comment_by_position, {
        "engine": "vllm_offline_ordered_batch",
        "vllm_version": payload["vllm_version"],
        "startup_seconds": float(payload["startup_seconds"]),
        "inference_seconds": float(payload["inference_seconds"]),
        "reason_seconds": float(payload["reason_seconds"]),
        "comment_count": int(payload["comment_count"]),
        "comment_source_counts": dict(payload["comment_source_counts"]),
        "reason_policy_sha256": str(payload["reason_policy_sha256"]),
        "request_count": int(payload["request_count"]),
        "subprocess_seconds": time.perf_counter() - started,
    }


def predict_bad(
    rows: pd.DataFrame,
    flv_rows: pd.DataFrame,
    flv_labels: np.ndarray,
    *,
    mock: bool,
    workspace: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    verify_frozen_policy()
    rows = rows.reset_index(drop=True)
    if rows.empty:
        return np.zeros(0, dtype=np.int8), {"rows": 0, "mock": bool(mock)}, {}
    if mock:
        strict = _mock_margins(rows, "strict")
        broad = _mock_margins(rows, "broad")
        performance: dict[str, Any] = {"mock": True}
        comments: dict[tuple[str, int], dict[str, Any]] = {}
    else:
        strict, broad, comments, performance = _predict_bad_vllm(
            rows, flv_rows, flv_labels, workspace
        )
        performance["mock"] = False
    negation = np.asarray(
        [
            contains_strict_bad_negation(name, description)
            for name, description in zip(rows.name, rows.description, strict=True)
        ],
        dtype=bool,
    )
    prediction = ((strict + broad) >= SUM_MARGIN_THRESHOLD).astype(np.int8)
    prediction[negation] = 0
    if not mock:
        for position, label in enumerate(prediction):
            item = comments.get((BAD, position))
            if item is None or int(item["label"]) != int(label):
                raise RuntimeError("Reason worker changed a frozen BAD label")
        for position, label in enumerate(flv_labels):
            item = comments.get((FLAMMABLE, position))
            if item is None or int(item["label"]) != int(label):
                raise RuntimeError("Reason worker changed a frozen FLV label")
    performance.update(
        {
            "rows": len(rows),
            "positive_count": int(prediction.sum()),
            "literal_negation_count": int(negation.sum()),
            "finite_margins": bool(np.isfinite(strict).all() and np.isfinite(broad).all()),
        }
    )
    return prediction, performance, comments


def flv_rule_label(name: object, description: object, fallback: int) -> int:
    """The four fixed FLV text gates."""
    n = str(name or "").lower()
    x = str(description or "").lower()
    kit = bool(
        re.search(
            r"(?:плита|горелк|печь).{0,80}\bи\b.{0,80}(?:баллон|картридж).{0,20}\d+\s*(?:мл|л\b)",
            n,
        )
    )
    paraffin = bool(
        re.search(r"бруск", n)
        and re.search(r"(?:парафин.{0,120}пропитан|пропитан.{0,120}парафин)", x)
    )
    stove = bool(
        re.search(r"печь", n)
        and re.search(r"в качестве топлива использ|топливо.*использ|работает на|требует.*топлив", x)
        and not re.search(r"баллон|картридж|топливо|газ", n)
    )
    wood = bool(
        re.search(r"ролл|растоп|стружк", n)
        and re.search(r"без химическ|химическ.{0,80}отсутств|отсутств.{0,80}химическ", x)
        and not re.search(r"парафин|жидкост|пропитан", x)
    )
    if int(fallback) == 0 and (kit or paraffin):
        return 1
    if int(fallback) == 1 and (stove or wood):
        return 0
    return int(fallback)


def format_bad_result(label: int, comment: str | None = None) -> str:
    value = int(label)
    if value not in {0, 1}:
        raise ValueError(f"Unexpected BAD label: {value}")
    verdict = "не бан" if value == 1 else "бан"
    selected_comment = str(comment or BAD_COMMENTS[value]).strip()
    if not 50 <= len(selected_comment) <= 300:
        selected_comment = BAD_COMMENTS[value]
    result = f"<комментарий>{selected_comment}<вердикт>{verdict}"
    if verdict_label(result) != value:
        raise AssertionError("Invalid organizer-facing BAD result")
    return result


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        prefix=f".{destination.name}.",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, destination)


def run_pipeline(
    *, test_data_path: Path, output_path: Path, mock_inference: bool
) -> dict[str, Any]:
    started = time.perf_counter()
    test_data_path = test_data_path.resolve()
    output_path = output_path.resolve()
    source = load_source(test_data_path)
    verify_frozen_policy()
    manifest = production_manifest()
    bad_source = source.loc[source.category.eq(BAD)].reset_index(drop=True)
    flv_source = source.loc[source.category.eq(FLAMMABLE)].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ecup_runtime_", dir=output_path.parent) as temporary:
        workspace = Path(temporary)
        if len(flv_source):
            flammable = prepare_flammable(
                source, test_data_path, workspace / "flv_sheets"
            )
            flv_fallback, flv_status = predict_flammable(
                flammable, mock=bool(mock_inference)
            )
            flv_final = np.asarray(
                [
                    flv_rule_label(row.name, row.description, fallback)
                    for row, fallback in zip(
                        flv_source.itertuples(index=False), flv_fallback, strict=True
                    )
                ],
                dtype=np.int8,
            )
        else:
            flv_fallback = np.zeros(0, dtype=np.int8)
            flv_final = np.zeros(0, dtype=np.int8)
            flv_status = {"rows": 0, "mock": bool(mock_inference)}
        bad_final, bad_status, generated_comments = predict_bad(
            bad_source,
            flv_source,
            flv_final,
            mock=bool(mock_inference),
            workspace=workspace,
        )

    label_by_id: dict[int, int] = {}
    for identifier, label in zip(bad_source.id.astype(int), bad_final, strict=True):
        label_by_id[int(identifier)] = int(label)
    for identifier, label in zip(flv_source.id.astype(int), flv_final, strict=True):
        label_by_id[int(identifier)] = int(label)
    if len(label_by_id) != len(source):
        raise RuntimeError("Category branches lost source rows")
    category_positions = {BAD: 0, FLAMMABLE: 0}
    results = []
    for row in source.itertuples(index=False):
        category = str(row.category)
        position = category_positions[category]
        category_positions[category] += 1
        generated = generated_comments.get((category, position), {})
        comment = generated.get("comment")
        if category == BAD:
            results.append(format_bad_result(label_by_id[int(row.id)], comment))
        else:
            results.append(format_flammable_result(label_by_id[int(row.id)], comment))
    output = pd.DataFrame({"id": source.id.astype(np.int64), "result": results})
    validate_output(source, output)
    atomic_csv(output, output_path)
    status = {
        "status": "INFERENCE_RUNTIME_COMPLETE",
        "package_id": PACKAGE_ID,
        "model": BASE_MODEL,
        "model_revision": BASE_REVISION,
        "rows": len(source),
        "bad_rows": len(bad_source),
        "bad": bad_status,
        "flv_rows": len(flv_source),
        "flv_rule_changes": int(np.sum(flv_final != flv_fallback)),
        "flv": flv_status,
        "production_manifest_self_sha256": manifest["self_sha256"],
        "mock": bool(mock_inference),
        "elapsed_seconds": time.perf_counter() - started,
        "output_path": str(output_path),
    }
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        "--test-data-path",
        "-i",
        type=Path,
        required=True,
        dest="test_data_path",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        "-o",
        type=Path,
        required=True,
        dest="output_path",
    )
    parser.add_argument(
        "--mock-inference",
        action="store_true",
        default=os.environ.get("ECUP_MOCK_INFERENCE") == "1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = run_pipeline(
        test_data_path=args.test_data_path,
        output_path=args.output_path,
        mock_inference=bool(args.mock_inference),
    )
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
