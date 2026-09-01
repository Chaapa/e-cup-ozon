"""Train-validated evidence selection and rendering for organizer comments."""

from __future__ import annotations

import html
import hashlib
import json
import re
from typing import Any

from .constants import BAD, FLAMMABLE


PROMPT_A = """Итог уже зафиксирован и не подлежит изменению.
Верни JSON {"evidence_id":0,"basis":"..."}, выбрав только существующий
evidence_id и basis из списка пользователя. Цитата должна прямо называть
продаваемый объект или его статус; обещанный эффект сам по себе недостаточен.
Приоритет: прямой статус или отрицание; затем точное название типа товара.
Для FLV: спички/розжиг/зажигалка — ignition_source; хлопушка/дым/пиротехника —
pyrotechnic_article; перезаправляемая горелка — refillable_flame_device;
топливо/горючее — sold_fuel; устройство с внешним баллоном — device_only.
Для BAD: выбирай прямые БАД/supplement, «не является БАД», спортивное питание,
BCAA/протеин/предтренировочный комплекс, обычную еду или наружное средство.
Не достраивай отсутствующие свойства и не используй соседний объект."""

PROMPT_B = """Итог уже зафиксирован и не подлежит изменению.
Выбери существующий evidence_id и basis только из списка пользователя. Фрагмент
должен прямо описывать продаваемый объект или его решающий статус. Прямое
утверждение сильнее рекламного перечисления, название товара сильнее эффекта.
Не выводи отсутствующие свойства и не используй соседний объект. Верни только
JSON {"evidence_id":0,"basis":"..."}."""

CRITIC_PROMPT = """Проверь предложенную пару evidence/basis для уже заданного
итога. Сохрани её, если цитата буквально описывает продаваемый объект и прямо
поддерживает basis. Замени только если цитата относится к рекламе, соседнему
объекту, лишь эффекту применения или другому смыслу. Для замены выбери только
существующий evidence_id и допустимый basis. Не придумывай факты. Верни JSON
{"action":"keep","evidence_id":0,"basis":"..."}."""

SELECTOR_PROMPT = """A — основной уже проверенный кандидат. Сохрани A по
умолчанию. Выбери B только если A буквально не подтверждает собственное
основание, относится к рекламе/соседнему объекту/эффекту, а B при этом содержит
однозначно более прямой статус или название именно продаваемого товара.
Стилистическое улучшение, более короткая фраза или другое допустимое основание
не являются причиной менять A. При любом сомнении и при двух допустимых
объяснениях выбери A. Верни только JSON {"choice":"A"}."""

PHRASES_B = {
    "explicit_bad_status": "прямой статус БАД у продаваемого товара",
    "oral_dosed_supplement": "дозированную добавку для приёма внутрь",
    "explicit_not_bad": "прямое отрицание статуса БАД",
    "sport_nutrition": "прямо названный BCAA-, протеиновый или предтренировочный продукт",
    "ordinary_food": "то, что продаётся обычный пищевой продукт",
    "external_product": "наружное средство, а не добавку для приёма внутрь",
    "raw_or_carrier": "сырьё или носитель без самостоятельного статуса БАД",
    "mushroom_product": "прямо названный набор грибов или грибных экстрактов",
    "other_non_bad_product": "иной прямо названный тип продаваемого товара",
    "sold_fuel": "продаваемое топливо или горючий материал",
    "ignition_source": "самостоятельное средство розжига или источник огня",
    "included_flammable_content": "включённое в продаваемый комплект горючее содержимое",
    "pyrotechnic_article": "пиротехническое или дымовое изделие",
    "refillable_flame_device": "перезаправляемое устройство, создающее пламя",
    "device_only": "устройство, использующее внешний источник топлива",
    "external_fuel_only": "упоминание топлива лишь как внешнего условия применения",
    "accessory_only": "аксессуар или комплектующую без продаваемого горючего содержимого",
    "ordinary_object": "обычный предмет, а не продаваемое топливо или источник огня",
}
PHRASES_A = {**PHRASES_B, "sport_nutrition": "то, что продаётся спортивное питание"}

ALLOWED_B = {
    (BAD, 1): ("explicit_bad_status", "oral_dosed_supplement"),
    (BAD, 0): (
        "explicit_not_bad", "sport_nutrition", "ordinary_food", "external_product",
        "raw_or_carrier", "mushroom_product", "other_non_bad_product",
    ),
    (FLAMMABLE, 1): (
        "sold_fuel", "ignition_source", "included_flammable_content",
        "pyrotechnic_article", "refillable_flame_device",
    ),
    (FLAMMABLE, 0): (
        "device_only", "external_fuel_only", "accessory_only", "ordinary_object",
    ),
}
ALLOWED_A = {
    key: tuple(value for value in values if value != "mushroom_product")
    for key, values in ALLOWED_B.items()
}

MARKERS = {
    "explicit_bad_status": ("бад", "supplement", "добавк", "биологически активн"),
    "oral_dosed_supplement": ("капсул", "таблет", "принимать", "приём", "дозиров"),
    "explicit_not_bad": ("не является бад", "не является биологически активной"),
    "sport_nutrition": ("спортивное питание", "bcaa", "бцаа", "протеин", "предтренировоч"),
    "ordinary_food": ("напиток", "сок", "чай", "кофе", "батончик", "конфет", "еда"),
    "external_product": ("наружн", "космет", "крем", "сыворотк", "мазь", "шампун"),
    "raw_or_carrier": ("сырь", "пуст", "капсул для", "оболочк", "носител"),
    "mushroom_product": ("гриб", "ежовик", "кордицепс", "траметес"),
    "other_non_bad_product": ("питьевой", "комплекс", "порошок", "масло", "экстракт"),
    "sold_fuel": ("сухое горючее", "топливо", "бензин", "горючий материал"),
    "ignition_source": ("спич", "розжиг", "зажигал"),
    "included_flammable_content": ("в комплект", "состав набора", "баллон в комплект"),
    "pyrotechnic_article": ("хлопуш", "дым", "пиротех", "фейервер"),
    "refillable_flame_device": ("перезаправ", "заправляем", "газовая зажигал"),
    "device_only": ("плита", "горелка", "устройство", "гриль"),
    "external_fuel_only": ("подключ", "баллон", "газовые смеси", "внешн"),
    "accessory_only": ("насадк", "переходник", "чехол", "крепеж", "комплектующ"),
    "ordinary_object": ("пистон", "держатель", "контейнер", "посуда", "решетка"),
}

KEYWORDS = {
    BAD: ("бад", "supplement", "добавк", "витамин", "капсул", "таблет", "приём", "пищ", "спорт", "наруж", "лекар"),
    FLAMMABLE: ("горюч", "огн", "плам", "топлив", "газ", "баллон", "спич", "розжиг", "зажиг", "свеч", "дым", "пиротех"),
}

FALLBACK = {
    (BAD, 1): "Карточка подтверждает соответствие правилам категории БАД; приоритетные исключения для спортивного питания или прямого отрицания не установлены.",
    (BAD, 0): "Карточка не подтверждает обязательную маркировку БАД либо содержит приоритетное исключение для спортивного питания или прямого отрицания.",
    (FLAMMABLE, 1): "Карточка подтверждает источник воспламенения, горючее вещество или газ либо отдельный легковоспламеняющийся предмет в комплекте.",
    (FLAMMABLE, 0): "Не подтверждены источник воспламенения, горючее содержимое или топливо в комплекте; пустая конструкция сама по себе не относится к категории.",
}


def clean(value: object, limit: int = 9000) -> str:
    text = html.unescape(str(value or "")).replace("\x00", " ")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit * 2 // 3] + " … " + text[-limit // 3 :]


def normalized(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or "")).lower()).strip()


def _chunks(text: str, maximum: int = 120) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and len(" ".join(current + [word])) > maximum:
            result.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        result.append(" ".join(current))
    return result


def evidence_candidates(row: dict[str, Any], limit: int = 32) -> list[str]:
    parts: list[str] = []
    for raw in (row.get("name"), row.get("description")):
        for sentence in re.split(r"(?<=[.!?])\s+|[;\n]+", clean(raw)):
            parts.extend(_chunks(sentence.strip()))
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = normalized(part)
        if key and key not in seen:
            seen.add(key)
            unique.append(part)
    keywords = KEYWORDS[str(row["category"])]
    ranked = sorted(
        enumerate(unique),
        key=lambda item: (-sum(normalized(item[1]).count(k) for k in keywords), item[0]),
    )
    indices = sorted(index for index, _ in ranked[:limit])
    return [unique[index] for index in indices] or [clean(row.get("name")) or "Карточка товара"]


def user_prompt(row: dict[str, Any], *, candidate: str) -> tuple[str, list[str], tuple[str, ...]]:
    candidates = evidence_candidates(row)
    label = int(row["label"])
    verdict = "соответствует категории" if label else "не соответствует категории"
    allowed_map = ALLOWED_A if candidate == "A" else ALLOWED_B
    phrases = PHRASES_A if candidate == "A" else PHRASES_B
    allowed = allowed_map[(str(row["category"]), label)]
    bases = "\n".join(f"- {key}: {phrases[key]}" for key in allowed)
    evidence = "\n".join(f"E{index}: {value}" for index, value in enumerate(candidates))
    return (
        f"Категория: {row['category']}\nЗафиксированный итог: товар {verdict}.\n"
        f"Допустимые основания:\n{bases}\nФрагменты карточки:\n{evidence}",
        candidates,
        allowed,
    )


def legacy_score(basis: str, evidence: str) -> int:
    text = normalized(evidence)
    return sum(marker in text for marker in MARKERS[basis])


def weighted_score(basis: str, evidence: str) -> int:
    text = normalized(evidence)
    if basis == "explicit_bad_status":
        if re.search(r"биологически активн\w* добавк|пищев\w* добавк|бад к пище", text): return 6
        return 4 if "supplement" in text or re.search(r"(?<!\w)бад(?!\w)", text) else 0
    if basis == "explicit_not_bad": return 8 if re.search(r"не является.{0,35}(?:бад|биологически активн)", text) else 0
    if basis == "sport_nutrition": return 5 if re.search(r"спортивн\w* питан|(?<!\w)(?:bcaa|бцаа)(?!\w)|протеин|предтренировоч", text) else 0
    if basis == "ordinary_food": return 4 if re.search(r"(?<!\w)(?:напиток|сок|чай|кофе|батончик|конфета|еда)(?!\w)", text) else 0
    if basis == "sold_fuel": return 6 if re.search(r"сухое горючее|топлив|бензин|горючий материал|угол[ья]|брикет", text) else 0
    if basis == "ignition_source": return 5 if re.search(r"спич|набор.{0,20}розжиг|средств\w* розжиг|зажигал", text) else 0
    if basis == "included_flammable_content":
        has_bundle = re.search(r"в комплект|в состав.{0,20}набор", text)
        has_content = re.search(r"горюч|топлив|спич|газов\w* баллон|бензин", text)
        return 7 if has_bundle and has_content else 0
    if basis == "pyrotechnic_article": return 6 if re.search(r"хлопуш|дымовая шашка|цветн\w* дым|пиротех|фейервер|бенгальск|свеч\w* фонтан", text) else 0
    if basis == "refillable_flame_device": return 6 if re.search(r"перезаправ|заправляем|газовая зажигал", text) else 0
    if basis == "device_only": return 5 if re.search(r"плита|горелка|устройство|гриль", text) else 0
    return 3 if any(marker in text for marker in MARKERS[basis]) else 0


def _repair(row: dict[str, Any], candidates: list[str], evidence_id: int, basis: str, *, candidate: str) -> tuple[int, str, bool]:
    allowed = (ALLOWED_A if candidate == "A" else ALLOWED_B)[(str(row["category"]), int(row["label"]))]
    score = legacy_score if candidate == "A" else weighted_score
    if 0 <= evidence_id < len(candidates) and basis in allowed and score(basis, candidates[evidence_id]):
        return evidence_id, basis, False
    ranked = [
        (score(option, evidence), -index, -basis_index, option)
        for index, evidence in enumerate(candidates)
        for basis_index, option in enumerate(allowed)
        if score(option, evidence)
    ]
    if ranked:
        _, negative_index, _, repaired_basis = max(ranked)
        return -negative_index, repaired_basis, True
    return (evidence_id if 0 <= evidence_id < len(candidates) else 0), (basis if basis in allowed else allowed[-1]), True


def select_candidate(row: dict[str, Any], candidates: list[str], parsed: dict[str, Any], *, candidate: str) -> tuple[int, str, bool]:
    return _repair(row, candidates, int(parsed.get("evidence_id", -1)), str(parsed.get("basis") or ""), candidate=candidate)


def render(evidence: str, basis: str, *, candidate: str) -> str:
    phrase = (PHRASES_A if candidate == "A" else PHRASES_B)[basis]
    return f"Решающий фрагмент карточки — «{evidence}»; он указывает на {phrase}."


def valid_comment(row: dict[str, Any], evidence: str, comment: str) -> bool:
    source = normalized(f"{clean(row.get('name'))} {clean(row.get('description'))}")
    anchor = normalized(evidence).strip(" «»\"'.,;:!?…")
    return bool(
        evidence and anchor and normalized(evidence) in source and anchor in normalized(comment)
        and 50 <= len(comment) <= 300 and "<comment>" not in comment and "<verdict>" not in comment
    )


def fallback_comment(category: str, label: int) -> str:
    return FALLBACK[(str(category), int(label))]


def policy_sha256() -> str:
    payload = {
        "prompt_a": PROMPT_A,
        "prompt_b": PROMPT_B,
        "critic_prompt": CRITIC_PROMPT,
        "selector_prompt": SELECTOR_PROMPT,
        "phrases_a": PHRASES_A,
        "phrases_b": PHRASES_B,
        "allowed_a": {f"{key[0]}|{key[1]}": value for key, value in ALLOWED_A.items()},
        "allowed_b": {f"{key[0]}|{key[1]}": value for key, value in ALLOWED_B.items()},
        "markers": MARKERS,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
