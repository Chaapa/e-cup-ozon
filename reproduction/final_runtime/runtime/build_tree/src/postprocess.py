from __future__ import annotations

import re

import pandas as pd

from .constants import BAD, FLAMMABLE


# Exact organizer-facing output grammar. The runtime validates it before and
# after the subprocess boundary.
COMMENTS = {
    (FLAMMABLE, 1): (
        "Карточка подтверждает источник воспламенения, горючее вещество или "
        "газ либо отдельный легковоспламеняющийся предмет в комплекте."
    ),
    (FLAMMABLE, 0): (
        "Не подтверждены источник воспламенения, горючее содержимое или топливо "
        "в комплекте; пустая конструкция сама по себе не относится к категории."
    ),
}
RESULT_PATTERN = re.compile(
    r"^<комментарий>(?P<comment>.{50,300})<вердикт>(?P<verdict>бан|не бан)$",
    re.DOTALL,
)


def format_flammable_result(label: int, comment: str | None = None) -> str:
    label = int(label)
    if label not in {0, 1}:
        raise ValueError(f"Unexpected FLV label: {label}")
    verdict = "не бан" if label == 1 else "бан"
    selected_comment = str(comment or COMMENTS[(FLAMMABLE, label)]).strip()
    if not 50 <= len(selected_comment) <= 300:
        selected_comment = COMMENTS[(FLAMMABLE, label)]
    result = f"<комментарий>{selected_comment}<вердикт>{verdict}"
    if not RESULT_PATTERN.fullmatch(result):
        raise AssertionError("Invalid organizer-facing FLV result")
    return result


def verdict_label(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("FLV output result is not text")
    match = RESULT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("FLV output result does not match the official grammar")
    return int(match.group("verdict") == "не бан")


def validate_output(source: pd.DataFrame, output: pd.DataFrame) -> None:
    if list(output.columns) != ["id", "result"]:
        raise RuntimeError("Hybrid output schema drift")
    identifiers = pd.to_numeric(output["id"], errors="raise").astype("int64")
    expected_identifiers = source["id"].astype("int64")
    if identifiers.tolist() != expected_identifiers.tolist():
        raise RuntimeError("Hybrid output lost source ID order or coverage")
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise RuntimeError("Hybrid output IDs must be non-null and unique")
    valid = output["result"].map(
        lambda value: isinstance(value, str) and RESULT_PATTERN.fullmatch(value) is not None
    )
    if not bool(valid.all()):
        raise RuntimeError("Hybrid output string grammar drift")
    if not set(source["category"].astype(str)).issubset({BAD, FLAMMABLE}):
        raise RuntimeError("Hybrid source category domain drift")
