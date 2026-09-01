#!/usr/bin/env python3
"""Generate and validate the OpenRouter R5 synthetic training corpus."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from quality_bench.openrouter_runtime import (
    OpenRouterRunner,
    batches,
    canonical_json,
    estimated_cost,
    exact_overlap_count,
    load_config,
    request_body,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "quality_bench/config/openrouter_model.json"
CONTROL_FREEZE = ROOT / "quality_bench/config/r5_control.json"
PRODUCTION_FREEZE = ROOT / "quality_bench/config/r5_production.json"
BASE_OUTPUT = ROOT / "quality_bench/outputs/r5_generation/local_only"
PLACEHOLDER = "{{SLOT}}"
GLOBAL_API_CAP_USD = 10.0
PARALLEL_API_WORKERS = 4
TRANSPORT_LOG_LOCK = threading.Lock()


def copied_train_spans(
    rows: list[Mapping[str, Any]], source: pd.DataFrame, width: int = 13
) -> int:
    """Count copied token spans shared with organizer-train surfaces."""

    def tokens(value: object) -> list[str]:
        return re.findall(r"(?iu)[0-9a-zа-яё]+", str(value).lower())

    real_spans: set[tuple[str, ...]] = set()
    for name, description in zip(source.clean_name, source.clean_description):
        sequence = tokens(str(name) + " " + str(description))
        real_spans.update(
            tuple(sequence[index : index + width])
            for index in range(max(0, len(sequence) - width + 1))
        )
    hits = 0
    for row in rows:
        for side in ("positive", "negative"):
            sequence = tokens(
                str(row[side]["name"]) + " " + str(row[side]["description"])
            )
            hits += sum(
                tuple(sequence[index : index + width]) in real_spans
                for index in range(max(0, len(sequence) - width + 1))
            )
    return hits


BOUNDARIES: dict[str, dict[str, str]] = {
    "self_contained_torch": {
        "slot": "fuel_storage_mode",
        "positive": "у той же мини-горелки есть встроенный заправляемый резервуар",
        "negative": "та же мини-горелка работает только от отдельного внешнего баллона",
        "forbidden_cochanges": "тип инструмента, сопло, поджиг, назначение, мощность, аксессуары",
    },
    "sold_fuel_gas": {
        "slot": "container_fill_state",
        "positive": "та же продаваемая ёмкость заполнена горючим газом-топливом",
        "negative": "та же продаваемая ёмкость пуста и предназначена для последующей заправки",
        "forbidden_cochanges": "тип ёмкости, клапан, объём, назначение, упаковка",
    },
    "included_flammable": {
        "slot": "flammable_item_included",
        "positive": "спички, рабочая зажигалка или топливо явно входят в тот же комплект",
        "negative": "тот же комплект явно продаётся без них; источник приобретается отдельно",
        "forbidden_cochanges": "основной набор, назначение, количество, прочие аксессуары, стиль",
    },
    "matches_present": {
        "slot": "matches_present",
        "positive": "настоящие спички явно находятся в продаваемом футляре или держателе",
        "negative": "тот же футляр или держатель явно продаётся пустым, без спичек",
        "forbidden_cochanges": "тип держателя, материал, назначение, количество, оформление",
    },
    "standalone_solid_fuel": {
        "slot": "sold_solid_fuel_role",
        "positive": "продаётся карбонизированное основное твёрдое топливо — уголь или антрацит",
        "negative": "продаются некарбонизированные древесные роллы только для розжига отдельного топлива",
        "forbidden_cochanges": "форма фасовки, масса, количество, назначение карточки, стиль",
    },
    "pyro_popper": {
        "slot": "activation_mechanism",
        "positive": "эффект той же хлопушки создаёт пиротехнический горючий заряд",
        "negative": "эффект той же хлопушки создаёт пневматика или пружина без пиросостава",
        "forbidden_cochanges": "тип хлопушки, размер, конфетти, назначение, количество",
    },
    "pyro_cake_or_smoke": {
        "slot": "pyrotechnic_charge_present",
        "positive": "у того же праздничного изделия есть пиротехнический искровой, фонтанный или дымовой заряд",
        "negative": "то же оформление является обычной восковой свечой или декором без пиротехнического состава",
        "forbidden_cochanges": "назначение, размер, цвет, количество, оформление упаковки",
    },
    "integrated_match_firestarter": {
        "slot": "integrated_ignition_source",
        "positive": "в каждый продаваемый элемент розжига встроена спичка или воспламенитель",
        "negative": "тот же элемент требует отдельной спички или зажигалки, которых в комплекте нет",
        "forbidden_cochanges": "горючий материал, форма, время горения, количество, назначение",
    },
    "grill_or_heater_with_fuel": {
        "slot": "filled_fuel_in_device_bundle",
        "positive": "в комплект того же гриля или обогревателя входит заполненный топливный картридж",
        "negative": "тот же комплект продаётся без топлива; картридж приобретается отдельно",
        "forbidden_cochanges": "устройство, характеристики, прочие аксессуары, назначение, количество",
    },
    "standalone_ignition_source": {
        "slot": "working_ignition_source_present",
        "positive": "в продаваемой единице есть рабочие спички, зажигалка или огниво",
        "negative": "продаётся только тот же пустой футляр или корпус без рабочего источника",
        "forbidden_cochanges": "форм-фактор, материал, сценарий, оформление, упаковка",
    },
}


# Row-free methodology only: frame role, controlled field and fixed mass.  No
# No organizer pair/card text or rationale is present here.
FRAME_REGISTRY: list[dict[str, Any]] = [
    # Bucket A: 20 frames x 5 = 100, exact 50 name / 50 description.
    {"frame_id": "torch_spec_reservoir_01", "boundary": "self_contained_torch", "changed_field": "description", "count": 5, "style": "technical reservoir specification for a precision mini-torch"},
    {"frame_id": "torch_title_morphology_02", "boundary": "self_contained_torch", "changed_field": "name", "count": 5, "style": "compact catalogue title for a culinary mini-torch"},
    {"frame_id": "torch_seo_refill_03", "boundary": "self_contained_torch", "changed_field": "description", "count": 5, "style": "natural marketplace description for a refillable craft torch"},
    {"frame_id": "torch_title_storage_04", "boundary": "self_contained_torch", "changed_field": "name", "count": 5, "style": "short title for a heat-shrink micro-torch"},
    {"frame_id": "torch_noisy_card_05", "boundary": "self_contained_torch", "changed_field": "description", "count": 5, "style": "detailed but neutral hobby-tool card with invariant specifications"},
    {"frame_id": "gas_title_filled_06", "boundary": "sold_fuel_gas", "changed_field": "name", "count": 5, "style": "short title for a threaded camping gas container"},
    {"frame_id": "gas_spec_state_07", "boundary": "sold_fuel_gas", "changed_field": "description", "count": 5, "style": "neutral specification of a portable-lamp gas cylinder"},
    {"frame_id": "gas_refill_title_08", "boundary": "sold_fuel_gas", "changed_field": "name", "count": 5, "style": "catalogue title for a lighter-refill canister"},
    {"frame_id": "gas_short_description_09", "boundary": "sold_fuel_gas", "changed_field": "description", "count": 5, "style": "brief description of a reusable valve cylinder"},
    {"frame_id": "included_gift_10", "boundary": "included_flammable", "changed_field": "description", "count": 5, "style": "gift fireplace set with a restrained contents list"},
    {"frame_id": "included_bbq_title_11", "boundary": "included_flammable", "changed_field": "name", "count": 5, "style": "compact title for a barbecue accessory kit"},
    {"frame_id": "included_survival_12", "boundary": "included_flammable", "changed_field": "description", "count": 5, "style": "outdoor emergency kit with neutral invariant accessories"},
    {"frame_id": "included_fireplace_13", "boundary": "included_flammable", "changed_field": "name", "count": 5, "style": "catalogue title for a long-lighter fireplace set"},
    {"frame_id": "matches_holder_title_14", "boundary": "matches_present", "changed_field": "name", "count": 5, "style": "short title for a desktop match holder"},
    {"frame_id": "matches_case_description_15", "boundary": "matches_present", "changed_field": "description", "count": 5, "style": "calm description of a sealed travel match case"},
    {"frame_id": "matches_travel_title_16", "boundary": "matches_present", "changed_field": "name", "count": 5, "style": "catalogue title for a pocket match capsule"},
    {"frame_id": "solid_material_spec_17", "boundary": "standalone_solid_fuel", "changed_field": "description", "count": 5, "style": "material specification for a packaged solid-fuel product"},
    {"frame_id": "solid_title_state_18", "boundary": "standalone_solid_fuel", "changed_field": "name", "count": 5, "style": "short title distinguishing the sold solid material role"},
    {"frame_id": "solid_heating_description_19", "boundary": "standalone_solid_fuel", "changed_field": "description", "count": 5, "style": "neutral heating-use description with identical package facts"},
    {"frame_id": "solid_noisy_title_20", "boundary": "standalone_solid_fuel", "changed_field": "name", "count": 5, "style": "marketplace title with invariant size and pack count"},
    # Bucket B: 24 frames, four carry 5 variants and twenty carry 4 = 100.
    {"frame_id": "popper_table_title_21", "boundary": "pyro_popper", "changed_field": "name", "count": 4, "style": "compact title for a tabletop confetti popper"},
    {"frame_id": "popper_party_description_22", "boundary": "pyro_popper", "changed_field": "description", "count": 4, "style": "neutral party-card description with paper confetti"},
    {"frame_id": "popper_wedding_title_23", "boundary": "pyro_popper", "changed_field": "name", "count": 4, "style": "short title for a wedding confetti popper"},
    {"frame_id": "popper_children_description_24", "boundary": "pyro_popper", "changed_field": "description", "count": 4, "style": "restrained description for a celebration popper"},
    {"frame_id": "popper_long_title_25", "boundary": "pyro_popper", "changed_field": "name", "count": 4, "style": "catalogue title for a long foil-confetti popper"},
    {"frame_id": "cake_fountain_title_26", "boundary": "pyro_cake_or_smoke", "changed_field": "name", "count": 5, "style": "compact title for a cake-top celebration item"},
    {"frame_id": "cake_spark_description_27", "boundary": "pyro_cake_or_smoke", "changed_field": "description", "count": 4, "style": "calm description of a cake decoration effect"},
    {"frame_id": "cake_number_title_28", "boundary": "pyro_cake_or_smoke", "changed_field": "name", "count": 4, "style": "catalogue title for a numbered cake decoration"},
    {"frame_id": "cake_party_description_29", "boundary": "pyro_cake_or_smoke", "changed_field": "description", "count": 4, "style": "party-supply description with invariant color and count"},
    {"frame_id": "smoke_decor_title_30", "boundary": "pyro_cake_or_smoke", "changed_field": "name", "count": 4, "style": "short title for a decorative celebration item"},
    {"frame_id": "cake_pack_description_31", "boundary": "pyro_cake_or_smoke", "changed_field": "description", "count": 4, "style": "package-focused cake decoration description"},
    {"frame_id": "roll_integrated_title_32", "boundary": "integrated_match_firestarter", "changed_field": "name", "count": 4, "style": "short title for firestarter rolls"},
    {"frame_id": "roll_fireplace_description_33", "boundary": "integrated_match_firestarter", "changed_field": "description", "count": 4, "style": "fireplace-use description with fixed burn time"},
    {"frame_id": "roll_camp_title_34", "boundary": "integrated_match_firestarter", "changed_field": "name", "count": 4, "style": "catalogue title for camping firestarter rolls"},
    {"frame_id": "roll_pack_description_35", "boundary": "integrated_match_firestarter", "changed_field": "description", "count": 4, "style": "neutral pack specification for firestarter pieces"},
    {"frame_id": "roll_stove_title_36", "boundary": "integrated_match_firestarter", "changed_field": "name", "count": 4, "style": "compact title for stove-lighting rolls"},
    {"frame_id": "grill_case_description_37", "boundary": "grill_or_heater_with_fuel", "changed_field": "description", "count": 5, "style": "portable grill in a case with a contents list"},
    {"frame_id": "heater_bundle_title_38", "boundary": "grill_or_heater_with_fuel", "changed_field": "name", "count": 5, "style": "short title for a compact heater bundle"},
    {"frame_id": "stove_bundle_description_39", "boundary": "grill_or_heater_with_fuel", "changed_field": "description", "count": 5, "style": "technical contents description for a folding stove kit"},
    {"frame_id": "ignition_case_title_40", "boundary": "standalone_ignition_source", "changed_field": "name", "count": 4, "style": "short title for an outdoor ignition-source case"},
    {"frame_id": "ignition_survival_description_41", "boundary": "standalone_ignition_source", "changed_field": "description", "count": 4, "style": "survival-kit description with invariant accessories"},
    {"frame_id": "ignition_pocket_title_42", "boundary": "standalone_ignition_source", "changed_field": "name", "count": 4, "style": "compact title for a pocket ignition-source holder"},
    {"frame_id": "ignition_gift_description_43", "boundary": "standalone_ignition_source", "changed_field": "description", "count": 4, "style": "restrained gift-set description"},
    {"frame_id": "ignition_travel_title_44", "boundary": "standalone_ignition_source", "changed_field": "name", "count": 4, "style": "catalogue title for a sealed travel holder"},
]


CONTROL_TASKS = [
    {
        "pair_id": f"r5-control-{index:03d}",
        "style_frame_id": f"control_{boundary}_{variant}",
        "boundary": boundary,
        "changed_field": "name" if variant == 0 else "description",
        "variant_index": variant,
        "style": ("concise catalogue title" if variant == 0 else "neutral technical marketplace description")
        + f"; fresh control variant {index}",
    }
    for index, (boundary, variant) in enumerate(
        (boundary, variant) for boundary in BOUNDARIES for variant in range(2)
    )
]


def production_tasks() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    index = 0
    for frame in FRAME_REGISTRY:
        for variant in range(int(frame["count"])):
            result.append({
                "pair_id": f"r5-production-{index:03d}",
                "style_frame_id": frame["frame_id"],
                "boundary": frame["boundary"],
                "changed_field": frame["changed_field"],
                "variant_index": variant,
                "style": f"{frame['style']}; catalogue variant {variant + 1} of {frame['count']}",
            })
            index += 1
    return result


AUTHOR_PROMPT = """Ты создаёшь только structured fact frames для синтетических
контрфактуальных пар. Для каждой задачи напиши один общий шаблон с буквальным
placeholder `{{SLOT}}` и два коротких значения этого слота. Код сам подставит
значения и создаст обе карточки.

Жёсткие правила:
1. Placeholder встречается ровно один раз и только в заданном changed_field.
2. Другое поле полностью инвариантно и не утверждает decisive slot. Общий текст
   вокруг placeholder тоже нейтрален к слоту.
3. Оба slot_text явно и самостоятельно сообщают противоположные значения
   одного слота. Молчание не означает отсутствие.
4. Не меняй предмет продажи, назначение, форму, количество, характеристики,
   аксессуары, стиль или подробность. Значения грамматически одинаково
   вставляются в контекст и максимально близки по длине.
5. Варианты одного style_frame должны быть естественными и различаться
   инвариантными поверхностными деталями; не копируй текст другого варианта.
6. Никаких реальных брендов, продавцов, SKU, меток, классов, P/N, organizer,
   соревнования, правил разметки или фразы «является/не является ЛВЖ».
7. Пиши новый естественный русский marketplace-текст. Не копируй реальные
   карточки. Сохрани pair_id, style_frame_id, boundary, changed_slot,
   changed_field и variant_index без изменений. Верни только JSON."""


AUTHOR_SCHEMA = {
    "frames": [{
        "pair_id": "...", "style_frame_id": "...", "boundary": "...",
        "changed_slot": "...", "changed_field": "description", "variant_index": 0,
        "name_template": "...", "description_template": "... {{SLOT}} ...",
        "positive_slot_text": "...", "negative_slot_text": "...",
        "invariant_fact_frame": "...",
    }]
}


REVIEW_PROMPT = """Ты независимый слепой аудитор синтетических пар по
зафиксированной organizer-конвенции категории «Легковоспламеняющиеся». Тебе не
даны авторские роли, boundary, frame или предложенные метки. Сам поставь A/B
метки и назови один наблюдаемый слот.

Локальные направления:
- встроенный резервуар мини-горелки = 1; только внешний баллон = 0;
- заполненная ёмкость с горючим газом = 1; та же пустая ёмкость = 0;
- явно включённые спички/рабочая зажигалка/топливо = 1; тот же набор без них = 0;
- футляр с настоящими спичками = 1; тот же пустой футляр = 0;
- карбонизированное основное твёрдое топливо = 1; древесные роллы только для
  розжига отдельного топлива = 0;
- пиротехническая хлопушка = 1; пневматическая/пружинная без пиросостава = 0;
- пиротехнический искровой/фонтанный/дымовой заряд = 1; обычная восковая свеча
  или декор без пиросостава = 0;
- встроенная в каждый ролл спичка = 1; внешний источник, не входящий в набор = 0;
- гриль/обогреватель с заполненным топливным картриджем в комплекте = 1; тот же
  комплект без топлива = 0;
- рабочие спички/зажигалка/огниво в продаваемой единице = 1; пустой корпус = 0.

Пара usable только если это один product-frame, противоположные значения ровно
одного явного слота, нет второго изменения и обе стороны естественны. Внешнее
топливо не означает комплектность. Отклоняй противоречия title/description,
неявное отсутствие, разные первичные сущности, реальные бренды/SKU и meta-текст.
Верни только JSON."""


REVIEW_SCHEMA = {
    "items": [{
        "pair_id": "...", "label_a": 0, "label_b": 1,
        "same_product_frame": True, "single_causal_boundary": True,
        "pair_usable": True, "observed_slot": "...",
        "evidence_a": "...", "evidence_b": "...", "reject_reason": "NONE",
    }]
}


def author_user(tasks: list[Mapping[str, Any]]) -> str:
    payload = [{**dict(task), **BOUNDARIES[str(task["boundary"])]} for task in tasks]
    return "Построй frames для задач:\n" + canonical_json({"tasks": payload}) + "\n\nСхема:\n" + canonical_json(AUTHOR_SCHEMA)


def validate_frames(payload: Mapping[str, Any], tasks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {str(row["pair_id"]): row for row in tasks}
    frames = payload.get("frames")
    if not isinstance(frames, list) or {str(row.get("pair_id")) for row in frames} != set(expected):
        raise ValueError("frame pair_id coverage drift")
    result: list[dict[str, Any]] = []
    for source in frames:
        row = dict(source)
        task = expected[str(row["pair_id"])]
        for key in ("style_frame_id", "boundary", "changed_field", "variant_index"):
            if row.get(key) != task[key]:
                raise ValueError(f"{key} drift for {row['pair_id']}")
        boundary = str(row["boundary"])
        if row.get("changed_slot") != BOUNDARIES[boundary]["slot"]:
            raise ValueError(f"changed_slot drift for {row['pair_id']}")
        required = ("name_template", "description_template", "positive_slot_text", "negative_slot_text", "invariant_fact_frame")
        if any(not str(row.get(key, "")).strip() for key in required):
            raise ValueError(f"incomplete frame {row['pair_id']}")
        expected_placeholders = {
            "name": int(row["changed_field"] == "name"),
            "description": int(row["changed_field"] == "description"),
        }
        actual = {field: str(row[f"{field}_template"]).count(PLACEHOLDER) for field in ("name", "description")}
        if actual != expected_placeholders:
            raise ValueError(f"placeholder drift for {row['pair_id']}: {actual}")
        result.append(row)
    return sorted(result, key=lambda row: str(row["pair_id"]))


def render_frames(frames: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for frame in frames:
        row: dict[str, Any] = {
            key: frame[key] for key in (
                "pair_id", "style_frame_id", "boundary", "changed_slot",
                "changed_field", "variant_index", "invariant_fact_frame",
            )
        }
        row["positive_first"] = int(str(frame["pair_id"]).rsplit("-", 1)[-1]) % 2 == 0
        for role, slot_key in (("positive", "positive_slot_text"), ("negative", "negative_slot_text")):
            slot = str(frame[slot_key]).strip()
            row[role] = {
                "name": str(frame["name_template"]).replace(PLACEHOLDER, slot),
                "description": str(frame["description_template"]).replace(PLACEHOLDER, slot),
                "decisive_span": slot,
            }
        rendered.append(row)
    return rendered


META_RE = re.compile(r"(?iu)organizer|организатор|target|label|метк[аи]|класс\s*[01]|P[123]|N[1-7]|соревнован|датасет")
SKU_RE = re.compile(r"(?u)\b(?=[A-ZА-Я0-9-]{6,}\b)(?=[A-ZА-Я0-9-]*\d)(?=[A-ZА-Я0-9-]*[A-ZА-Я])[A-ZА-Я0-9-]+\b")
BRAND_RE = re.compile(r"(?iu)\b(weber|zippo|campingaz|kovea|coleman|forester|boyscout|remington|ronson)\b")


def deterministic_audit(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for row in rows:
        changed = str(row["changed_field"])
        invariant = "description" if changed == "name" else "name"
        positive, negative = row["positive"], row["negative"]
        pslot, nslot = str(positive["decisive_span"]), str(negative["decisive_span"])
        combined = " ".join([positive["name"], positive["description"], negative["name"], negative["description"]])
        checks = {
            "invariant_field_byte_identical": positive[invariant] == negative[invariant],
            "changed_field_differs": positive[changed] != negative[changed],
            "positive_slot_binds_once": str(positive[changed]).count(pslot) == 1,
            "negative_slot_binds_once": str(negative[changed]).count(nslot) == 1,
            "masking_slot_makes_identical": str(positive[changed]).replace(pslot, PLACEHOLDER) == str(negative[changed]).replace(nslot, PLACEHOLDER),
            "no_meta_vocabulary": META_RE.search(combined) is None,
            "no_known_real_brand": BRAND_RE.search(combined) is None,
            "no_sku_like_string": SKU_RE.search(combined) is None,
            "balanced_slot_length": abs(len(pslot) - len(nslot)) <= 35,
        }
        audits.append({
            "pair_id": row["pair_id"], "style_frame_id": row["style_frame_id"],
            "boundary": row["boundary"], "checks": checks, "pass": all(checks.values()),
        })
    passed = sum(int(row["pass"]) for row in audits)
    return audits, {"pairs": len(audits), "passed": passed, "pass_rate": passed / max(len(audits), 1)}


def reviewer_user(rows: list[Mapping[str, Any]]) -> str:
    payload = []
    for row in rows:
        swap = int(sha256_text(str(row["pair_id"])), 16) % 2 == 1
        a = row["negative"] if swap else row["positive"]
        b = row["positive"] if swap else row["negative"]
        payload.append({"pair_id": row["pair_id"], "A": {"name": a["name"], "description": a["description"]}, "B": {"name": b["name"], "description": b["description"]}})
    return "Проверь пары:\n" + canonical_json({"pairs": payload}) + "\n\nСхема:\n" + canonical_json(REVIEW_SCHEMA)


def validate_reviews(payload: Mapping[str, Any], expected: set[str]) -> list[dict[str, Any]]:
    items = payload.get("items")
    if not isinstance(items, list) or {str(row.get("pair_id")) for row in items} != expected:
        raise ValueError("review pair_id coverage drift")
    result = []
    for source in items:
        row = dict(source)
        if row.get("label_a") not in (0, 1) or row.get("label_b") not in (0, 1):
            raise ValueError("invalid review labels")
        if any(not isinstance(row.get(key), bool) for key in ("same_product_frame", "single_causal_boundary", "pair_usable")):
            raise ValueError("invalid review booleans")
        if any(not str(row.get(key, "")).strip() for key in ("observed_slot", "evidence_a", "evidence_b", "reject_reason")):
            raise ValueError("incomplete review")
        result.append(row)
    return sorted(result, key=lambda row: str(row["pair_id"]))


def cache_total_cost(cache_dirs: Iterable[Path]) -> float:
    total = 0.0
    for cache_dir in cache_dirs:
        for path in cache_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            total += float((payload.get("usage") or {}).get("cost") or 0.0)
    return total


def all_attempt_cache_dirs() -> list[Path]:
    """Return all paid caches covered by the configured API cap."""
    values: set[Path] = set()
    if BASE_OUTPUT.parent.is_dir():
        values.update(path.resolve() for path in BASE_OUTPUT.parent.rglob("cache") if path.is_dir())
    return sorted(values)


def complete_guarded(
    runner: OpenRouterRunner,
    body: Mapping[str, Any],
    model_config: Mapping[str, Any],
    output: Path,
    *,
    top_level_array_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_hash = sha256_text(canonical_json(body))
    cached = (output / "cache" / f"{request_hash}.json").exists()
    committed = cache_total_cost(all_attempt_cache_dirs())
    projected = 0.0 if cached else estimated_cost(body, model_config)
    if committed + projected > GLOBAL_API_CAP_USD:
        raise RuntimeError(f"global OpenRouter cap would be exceeded: {committed:.6f}+{projected:.6f}")
    try:
        payload, raw = runner.complete(body)
    except ValueError as error:
        # Some OpenRouter providers occasionally return a valid top-level JSON
        # array despite response_format=json_object and the frozen object schema.
        # This is transport normalization only: prompts, tasks, fields and gates
        # remain unchanged, and the untouched raw response stays in the cache.
        raw_path = output / "cache" / f"{request_hash}.json"
        if not raw_path.is_file():
            raise
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        content = str(raw["choices"][0]["message"].get("content") or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
        decoded = json.loads(content)
        if not isinstance(decoded, list):
            raise error
        payload = {top_level_array_key: decoded}
        log_path = output / "transport_normalizations.jsonl"
        record = {
            "request_sha256": request_hash,
            "normalization": f"top_level_array_to_{top_level_array_key}",
            "items": len(decoded),
            "raw_response_sha256": sha256_file(raw_path),
        }
        with TRANSPORT_LOG_LOCK:
            existing = []
            if log_path.is_file():
                existing = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            if not any(row.get("request_sha256") == request_hash for row in existing):
                write_jsonl(log_path, [*existing, record])
    actual = cache_total_cost(all_attempt_cache_dirs())
    if actual > GLOBAL_API_CAP_USD + 1e-12:
        raise RuntimeError("global OpenRouter cap exceeded after response")
    return payload, raw


def population_audit(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def norm(value: object) -> str:
        return re.sub(r"[^0-9a-zа-яё]+", " ", str(value).lower()).strip()

    cards = []
    masked = []
    for row in rows:
        for role in ("positive", "negative"):
            cards.append(norm(row[role]["name"] + " " + row[role]["description"]))
        changed = str(row["changed_field"])
        invariant = "description" if changed == "name" else "name"
        template = str(row["positive"][changed]).replace(str(row["positive"]["decisive_span"]), PLACEHOLDER)
        masked.append(norm(str(row["positive"][invariant]) + " " + template))
    duplicate_cards = len(cards) - len(set(cards))
    duplicate_masked_pairs = len(masked) - len(set(masked))
    max_cross_frame = 0.0
    cross_edges = 0
    if len(masked) > 1:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(masked)
        similarity = (matrix @ matrix.T).toarray()
        frames = [str(row["style_frame_id"]) for row in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if frames[i] != frames[j]:
                    value = float(similarity[i, j])
                    max_cross_frame = max(max_cross_frame, value)
                    cross_edges += int(value >= 0.94)
    deltas = np.array([
        len(row["positive"]["name"] + row["positive"]["description"])
        - len(row["negative"]["name"] + row["negative"]["description"])
        for row in rows
    ], dtype=float)
    by_boundary = {}
    for boundary in BOUNDARIES:
        values = [deltas[index] for index, row in enumerate(rows) if row["boundary"] == boundary]
        by_boundary[boundary] = float(np.mean(values)) if values else None
    return {
        "exact_duplicate_cards": duplicate_cards,
        "exact_duplicate_masked_pairs": duplicate_masked_pairs,
        "cross_frame_near_duplicate_edges_ge_0_94": cross_edges,
        "max_cross_frame_masked_similarity": max_cross_frame,
        "mean_positive_minus_negative_chars": float(np.mean(deltas)),
        "median_positive_minus_negative_chars": float(np.median(deltas)),
        "per_boundary_mean_positive_minus_negative_chars": by_boundary,
    }


def training_cards(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        order = ("positive", "negative") if row["positive_first"] else ("negative", "positive")
        for position, role in enumerate(order):
            label = int(role == "positive")
            result.append({
                "synthetic_id": f"{row['pair_id']}:{position}",
                "pair_id": row["pair_id"], "style_frame_id": row["style_frame_id"],
                "edit_template_family": row["style_frame_id"], "boundary": row["boundary"],
                "pair_position": position, "organizer_label": label,
                "name": row[role]["name"], "description": row[role]["description"],
                "decisive_span": row[role]["decisive_span"],
                "source_id": None, "source_component": None,
                "fully_synthetic": True, "source_surface_preserved": False,
            })
    return result


def freeze_payload(stage: str) -> dict[str, Any]:
    tasks = CONTROL_TASKS if stage == "control" else production_tasks()
    gates = {
        "schema_success_rate_min": 1.0,
        "structural_pass_rate_min": 1.0,
        "blind_direction_accuracy_min": 0.95 if stage == "production" else 0.90,
        "blind_usable_rate_min": 0.95 if stage == "production" else 0.90,
        "exact_train_overlap_max": 0,
        "copied_13_token_train_spans_max": 0,
        "exact_duplicate_cards_max": 0,
        "exact_duplicate_masked_pairs_max": 0,
        "cross_frame_near_duplicate_edges_ge_0_94_max": 0,
        "global_all_attempt_cost_usd_max": GLOBAL_API_CAP_USD,
    }
    return {
        "schema_version": f"openrouter.r5.{stage}.freeze.1",
        "run_id": f"r5_{'control' if stage == 'control' else 'production'}",
        "frozen_before_api_call": True,
        "model": "qwen/qwen3.5-122b-a10b",
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "author_prompt_sha256": sha256_text(AUTHOR_PROMPT),
        "review_prompt_sha256": sha256_text(REVIEW_PROMPT),
        "tasks_sha256": sha256_text(canonical_json(tasks)),
        "boundaries_sha256": sha256_text(canonical_json(BOUNDARIES)),
        "frame_registry_sha256": sha256_text(canonical_json(FRAME_REGISTRY)),
        "tasks": len(tasks), "boundaries": len(BOUNDARIES),
        "style_frames": len({str(row["style_frame_id"]) for row in tasks}),
        "batch_size": 10,
        "gates": gates,
        "forbidden_runtime_inputs": [
            "supplementary training row texts, rationales or predictions",
            "public/private leaderboard rows, labels, predictions or IDs",
            "artifacts not produced by this run",
        ],
        "immutability_rule": "Do not edit prompts, boundaries, frames or gates during a run.",
    }


def run(
    stage: str,
    config_path: Path,
    output: Path,
    source_train: Path,
) -> dict[str, Any]:
    tasks = CONTROL_TASKS if stage == "control" else production_tasks()
    freeze_path = CONTROL_FREEZE if stage == "control" else PRODUCTION_FREEZE
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_freeze = freeze_payload(stage)
    preregistration_checks = {key: freeze.get(key) == value for key, value in expected_freeze.items()}
    if not all(preregistration_checks.values()):
        drift = [key for key, value in preregistration_checks.items() if not value]
        raise RuntimeError(f"R5 generation contract drift: {drift}")
    model_config = load_config(config_path)
    if model_config["model"]["parameter_count"] > 400_000_000_000:
        raise RuntimeError("model exceeds 400B")
    if not source_train.is_file():
        raise FileNotFoundError(
            "Organizer train is not distributed with this package; "
            "pass its FLV Parquet with --source-train"
        )
    source = pd.read_parquet(source_train)
    data_config = model_config["data"]
    required_columns = set(data_config["required_columns"])
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise RuntimeError(f"Organizer train is missing columns: {missing_columns}")
    if len(source) != int(data_config["expected_rows"]):
        raise RuntimeError("Organizer train row count drift")
    if int(source["label"].sum()) != int(data_config["expected_positives"]):
        raise RuntimeError("Organizer train positive count drift")
    output.mkdir(parents=True, exist_ok=True)
    stage_cost_before = cache_total_cost([output / "cache"])
    global_cost_before = cache_total_cost(all_attempt_cache_dirs())
    previous_stages_cost = global_cost_before - stage_cost_before

    task_batches = list(batches(tasks, int(freeze["batch_size"])))

    def author_one(item: tuple[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        batch_index, task_batch = item
        author_runner = OpenRouterRunner(model_config, output, f"r5-{stage}-author-{batch_index}", 4.0)
        body = request_body(model_config, AUTHOR_PROMPT, author_user(task_batch), 20261010 + (0 if stage == "control" else 100) + batch_index, 7200)
        payload, _ = complete_guarded(
            author_runner, body, model_config, output,
            top_level_array_key="frames",
        )
        return validate_frames(payload, task_batch)

    frames: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_API_WORKERS) as pool:
        for values in pool.map(author_one, enumerate(task_batches)):
            frames.extend(values)
    frames = sorted(frames, key=lambda row: str(row["pair_id"]))
    write_jsonl(output / "authored_frames.jsonl", frames)
    rendered = render_frames(frames)
    write_jsonl(output / "rendered_pairs.jsonl", rendered)
    audits, structural = deterministic_audit(rendered)
    write_jsonl(output / "structural_audit.jsonl", audits)

    pair_batches = list(batches(rendered, int(freeze["batch_size"])))

    def review_one(item: tuple[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        batch_index, pair_batch = item
        reviewer_runner = OpenRouterRunner(model_config, output, f"r5-{stage}-blind-review-{batch_index}", 4.0)
        body = request_body(model_config, REVIEW_PROMPT, reviewer_user(pair_batch), 20261110 + (0 if stage == "control" else 100) + batch_index, 5800)
        payload, _ = complete_guarded(
            reviewer_runner, body, model_config, output,
            top_level_array_key="items",
        )
        return validate_reviews(payload, {str(row["pair_id"]) for row in pair_batch})

    reviews: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_API_WORKERS) as pool:
        for values in pool.map(review_one, enumerate(pair_batches)):
            reviews.extend(values)
    reviews = sorted(reviews, key=lambda row: str(row["pair_id"]))
    write_jsonl(output / "blind_reviews.jsonl", reviews)

    direction = usable = 0
    for review in reviews:
        swap = int(sha256_text(str(review["pair_id"])), 16) % 2 == 1
        expected_a, expected_b = ((0, 1) if swap else (1, 0))
        direction += int(review["label_a"] == expected_a and review["label_b"] == expected_b)
        usable += int(review["same_product_frame"] and review["single_causal_boundary"] and review["pair_usable"])

    normalized = [{
        "positive_name": row["positive"]["name"], "positive_description": row["positive"]["description"],
        "negative_name": row["negative"]["name"], "negative_description": row["negative"]["description"],
    } for row in rendered]
    exact_overlap = exact_overlap_count(normalized, source)
    span_hits = copied_train_spans(rendered, source)
    population = population_audit(rendered)
    cards = training_cards(rendered)
    write_jsonl(output / "training_cards.jsonl", cards)

    actual_global_cost = cache_total_cost(all_attempt_cache_dirs())
    stage_cost = cache_total_cost([output / "cache"])
    gates = freeze["gates"]
    checks = {
        "schema_success": len(frames) == len(tasks) and len(reviews) == len(tasks),
        "structural_pass_rate": structural["pass_rate"] >= gates["structural_pass_rate_min"],
        "blind_direction_accuracy": direction / len(tasks) >= gates["blind_direction_accuracy_min"],
        "blind_usable_rate": usable / len(tasks) >= gates["blind_usable_rate_min"],
        "exact_train_overlap": exact_overlap <= gates["exact_train_overlap_max"],
        "copied_13_token_train_spans": span_hits <= gates["copied_13_token_train_spans_max"],
        "exact_duplicate_cards": population["exact_duplicate_cards"] <= gates["exact_duplicate_cards_max"],
        "exact_duplicate_masked_pairs": population["exact_duplicate_masked_pairs"] <= gates["exact_duplicate_masked_pairs_max"],
        "cross_frame_near_duplicate_edges": population["cross_frame_near_duplicate_edges_ge_0_94"] <= gates["cross_frame_near_duplicate_edges_ge_0_94_max"],
        "global_all_attempt_budget": actual_global_cost <= gates["global_all_attempt_cost_usd_max"],
    }
    if stage == "production":
        boundary_counts = {key: sum(int(row["boundary"] == key) for row in rendered) for key in BOUNDARIES}
        frame_counts = {frame["frame_id"]: sum(int(row["style_frame_id"] == frame["frame_id"]) for row in rendered) for frame in FRAME_REGISTRY}
        checks.update({
            "exact_200_pairs": len(rendered) == 200,
            "exact_44_frames": len(frame_counts) == 44 and all(frame_counts[frame["frame_id"]] == frame["count"] for frame in FRAME_REGISTRY),
            "exact_boundary_mass": boundary_counts == {
                "self_contained_torch": 25, "sold_fuel_gas": 20, "included_flammable": 20,
                "matches_present": 15, "standalone_solid_fuel": 20, "pyro_popper": 20,
                "pyro_cake_or_smoke": 25, "integrated_match_firestarter": 20,
                "grill_or_heater_with_fuel": 15, "standalone_ignition_source": 20,
            },
            "exact_label_balance": len(cards) == 400 and sum(row["organizer_label"] for row in cards) == 200,
            "exact_pair_order_balance": sum(int(row["positive_first"]) for row in rendered) == 100,
            "exact_changed_field_mass": sum(int(row["changed_field"] == "name") for row in rendered) == 104,
        })
    verdict = "PASS" if all(checks.values()) else "FAIL"
    normalization_path = output / "transport_normalizations.jsonl"
    transport_normalizations = (
        len([line for line in normalization_path.read_text(encoding="utf-8").splitlines() if line])
        if normalization_path.is_file() else 0
    )
    metrics = {
        "schema_version": f"openrouter.r5.{stage}.metrics.1",
        "experiment_id": freeze["experiment_id"], "stage": stage,
        "pairs": len(tasks), "cards": len(cards), "boundaries": len(BOUNDARIES),
        "style_frames": len({str(row["style_frame_id"]) for row in tasks}),
        "model": model_config["model"],
        "author_prompt_sha256": sha256_text(AUTHOR_PROMPT), "review_prompt_sha256": sha256_text(REVIEW_PROMPT),
        "tasks_sha256": sha256_text(canonical_json(tasks)), "boundaries_sha256": sha256_text(canonical_json(BOUNDARIES)),
        "frame_registry_sha256": sha256_text(canonical_json(FRAME_REGISTRY)),
        "structural": structural,
        "blind_direction_correct": direction, "blind_direction_accuracy": direction / len(tasks),
        "blind_usable": usable, "blind_usable_rate": usable / len(tasks),
        "exact_train_overlap": exact_overlap, "copied_13_token_train_spans": span_hits,
        "population_audit": population,
        "previous_stages_api_cost_usd": previous_stages_cost,
        "this_stage_api_cost_usd": stage_cost,
        "global_api_cost_usd": actual_global_cost,
        "transport_normalizations": transport_normalizations,
        "parallel_api_workers": PARALLEL_API_WORKERS,
        "checks": checks, "preregistration_checks": preregistration_checks,
        "preregistration_path": str(freeze_path.relative_to(ROOT)),
        "preregistration_sha256": sha256_file(freeze_path), "verdict": verdict,
    }
    write_json(output / "metrics.json", metrics)
    manifest = {
        "schema_version": f"openrouter.r5.{stage}.artifacts.1", "verdict": verdict,
        "files": {
            name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in ("authored_frames.jsonl", "rendered_pairs.jsonl", "structural_audit.jsonl", "blind_reviews.jsonl", "training_cards.jsonl", "metrics.json")
        },
    }
    write_json(output / "artifact_manifest.json", manifest)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("control", "production"), required=True)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-train", type=Path)
    parser.add_argument("--print-freeze", action="store_true")
    args = parser.parse_args()
    if args.print_freeze:
        print(json.dumps(freeze_payload(args.stage), ensure_ascii=False, indent=2))
        return
    if args.source_train is None:
        parser.error("--source-train is required because organizer train is not bundled")
    output = args.output or (BASE_OUTPUT / args.stage)
    result = run(
        args.stage,
        args.model_config.resolve(),
        output.resolve(),
        args.source_train.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
