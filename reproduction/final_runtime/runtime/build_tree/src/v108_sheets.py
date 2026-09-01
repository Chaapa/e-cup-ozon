from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import CATEGORIES, CONTACT_SHEET_SIZE, FLAMMABLE


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"(?u)\b\w+\b")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def clean_text(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalized_text(value: object) -> str:
    return " ".join(WORD_RE.findall(clean_text(value).lower().replace("ё", "е")))


def image_paths(images_root: Path, product_id: int) -> list[Path]:
    folder = images_root / str(product_id)
    if not folder.is_dir():
        return []
    paths = [
        path
        for path in sorted(folder.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(paths) > 6:
        raise RuntimeError(
            f"V108 renderer supports at most six gallery panels; id={product_id}, found={len(paths)}"
        )
    return paths


def layout(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    if count == 2:
        return 2, 1
    if count <= 4:
        return 2, 2
    return 3, 2


def render_contact_sheet(
    paths: list[Path], destination: Path, *, size: int = CONTACT_SHEET_SIZE
) -> None:
    columns, rows = layout(len(paths))
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    cell_width, cell_height = size // columns, size // rows
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((cell_width - 12, cell_height - 42), Image.Resampling.LANCZOS)
            column, row = index % columns, index // columns
            cell_left, cell_top = column * cell_width, row * cell_height
            draw.rectangle(
                (cell_left, cell_top, cell_left + cell_width - 1, cell_top + cell_height - 1),
                outline=(180, 180, 180),
                width=2,
            )
            left = cell_left + (cell_width - image.width) // 2
            top = cell_top + 34 + (cell_height - 38 - image.height) // 2
            canvas.paste(image, (left, top))
            draw.rectangle(
                (cell_left + 4, cell_top + 4, cell_left + 116, cell_top + 30),
                fill="white",
            )
            draw.text(
                (cell_left + 9, cell_top + 7),
                f"IMAGE {index + 1}",
                fill="black",
                font=font,
            )
    canvas.save(
        destination,
        format="JPEG",
        quality=84,
        optimize=True,
        progressive=True,
    )


def _integer_ids(raw: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Input IDs must be finite integers")
    output = numeric.astype(np.int64)
    if output.duplicated().any():
        raise ValueError("Input IDs must be unique")
    return output


def _prepare_flv_row(row: dict[str, Any], images_root: Path, cache: Path) -> dict[str, Any]:
    product_id = int(row["id"])
    name = clean_text(row.get("name"))
    description = clean_text(row.get("description"))
    normalized_name = normalized_text(name)
    normalized_description = normalized_text(description)
    paths = image_paths(images_root, product_id)
    gallery_hashes = [sha256_file(path) for path in paths]
    exact_text_key = sha256_bytes(
        f"{normalized_name} {normalized_description}".strip().encode("utf-8")
    )
    gallery_signature = (
        sha256_bytes(canonical(sorted(gallery_hashes)).encode("utf-8"))
        if gallery_hashes else ""
    )
    destination = cache / f"v108_{product_id}.jpg"
    render_contact_sheet(paths, destination)
    content_hash = sha256_bytes(
        canonical(
            {
                "normalized_name": normalized_name,
                "normalized_description": normalized_description,
                "gallery_sha256": gallery_hashes,
            }
        ).encode("utf-8")
    )
    runtime_uid = sha256_bytes(
        canonical(
            {"namespace": "v108-production-row-v1", "id": str(product_id)}
        ).encode("utf-8")
    )
    tie_key = sha256_bytes(
        canonical({"content": content_hash, "row": runtime_uid}).encode("utf-8")
    )
    return {
        "id": product_id,
        "clean_name": name,
        "clean_description": description,
        "normalized_name": normalized_name,
        "normalized_description": normalized_description,
        "exact_text_key": exact_text_key,
        "gallery_signature": gallery_signature,
        "image_count": len(paths),
        "contact_sheet": str(destination),
        "content_hash": content_hash,
        "runtime_uid": runtime_uid,
        "tie_key": tie_key,
    }


def load_source(data_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_path).drop(columns=["Unnamed: 0"], errors="ignore")
    required = {"id", "name", "description", "category"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    raw = raw.copy()
    raw["id"] = _integer_ids(raw["id"])
    raw["category"] = raw["category"].map(clean_text)
    unknown = sorted(set(raw["category"]) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}")
    return raw


def prepare_flammable(source: pd.DataFrame, data_path: Path, cache: Path) -> pd.DataFrame:
    selected = source.loc[source["category"].eq(FLAMMABLE)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "clean_name",
                "clean_description",
                "normalized_name",
                "normalized_description",
                "exact_text_key",
                "gallery_signature",
                "image_count",
                "contact_sheet",
                "content_hash",
                "runtime_uid",
                "tie_key",
            ]
        )
    cache.mkdir(parents=True, exist_ok=False)
    records = selected.to_dict(orient="records")
    images_root = data_path.parent / "images"
    with ThreadPoolExecutor(max_workers=min(20, len(records))) as pool:
        prepared = list(
            pool.map(lambda row: _prepare_flv_row(row, images_root, cache), records)
        )
    frame = pd.DataFrame(prepared)
    if frame["tie_key"].duplicated().any() or frame["id"].tolist() != selected["id"].tolist():
        raise RuntimeError("V108 production preprocessing lost order or unique ties")
    return frame
