from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SHARED_MODELS = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
QWEN_MODEL = SHARED_MODELS / "Qwen" / "Qwen3.5-4B"

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
CATEGORIES = (BAD, FLAMMABLE)

SYSTEM_PROMPT = """Ты классифицируешь товар строго по конвенции организаторов.

1: самостоятельный источник открытого огня; или горючее/ЛВЖ/горючий газ в продаваемом товаре; или ЛВЖ-товар явно входит в комплект.

0: устройство лишь используется с огнём/топливом; содержимое отсутствует; источник воспламенения только встроен; горючий материал лишь компонент другого изделия; либо ЛВЖ-предмет явно не входит в комплект.

Изучи название, описание и все панели текущей галереи. Ответь ровно одним символом: 0 или 1."""

USER_TEMPLATE = """Название: {title}

Описание: {description}

Класс товара:"""

ZERO_TOKEN_ID = 15
ONE_TOKEN_ID = 16
CONTACT_SHEET_SIZE = 896
MIN_PIXELS = 12_544
MAX_PIXELS = 451_584
SCORE_BATCH_SIZE = 8
QUOTA_NUMERATOR = 198
QUOTA_DENOMINATOR = 5_502

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
