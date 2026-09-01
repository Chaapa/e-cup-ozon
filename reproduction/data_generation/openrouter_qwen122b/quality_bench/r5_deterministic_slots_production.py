#!/usr/bin/env python3
"""Full R5 corpus with LLM-authored neutral frames and code-owned slots."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import quality_bench.r5_openrouter_generation as core
from quality_bench.openrouter_runtime import canonical_json, sha256_file, sha256_text


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "quality_bench/config/r5_deterministic_slots_full_production.json"
OUTPUT = ROOT / "quality_bench/outputs/r5_generation/local_only"
BASE_PRODUCTION_TASKS = core.production_tasks


SLOT_BANKS: dict[str, list[tuple[str, str]]] = {
    "self_contained_torch": [
        ("встроенный заправляемый резервуар", "только отдельный внешний баллон"),
        ("горючий газ хранится во встроенной ёмкости", "внутренней ёмкости нет, нужен внешний баллон"),
        ("интегрированный бак для горючего газа", "подключение лишь к внешнему баллону"),
        ("собственный перезаправляемый топливный бак", "топливо подаётся только из внешней ёмкости"),
        ("встроенная ёмкость заполнена после заправки", "встроенной ёмкости нет, баллон ставится отдельно"),
    ],
    "sold_fuel_gas": [
        ("ёмкость заполнена горючим газом", "ёмкость пустая, горючего газа внутри нет"),
        ("внутри находится газовое топливо", "внутри нет топлива, предусмотрена последующая заправка"),
        ("баллон продаётся заправленным горючим газом", "баллон продаётся пустым, без горючего газа"),
        ("кассета содержит готовое газовое топливо", "кассета пустая и топлива не содержит"),
        ("резервуар поставляется с горючим газом", "резервуар поставляется без газа, пустым"),
    ],
    "included_flammable": [
        ("в комплект входят настоящие спички", "спичек в комплекте нет"),
        ("набор содержит рабочую зажигалку", "зажигалка в набор не входит"),
        ("горючее топливо явно включено в набор", "топливо приобретается отдельно и в наборе отсутствует"),
        ("комплект укомплектован источником огня", "источник огня в комплект не вложен"),
        ("в продаваемой упаковке есть средство розжига", "средство розжига продаётся отдельно"),
    ],
    "matches_present": [
        ("внутри находятся настоящие спички", "футляр пустой, спичек внутри нет"),
        ("комплект содержит рабочие спички", "комплект не содержит спичек"),
        ("спички вложены в продаваемый футляр", "спички в продаваемый футляр не вложены"),
        ("футляр укомплектован настоящими спичками", "футляр продаётся пустым, без спичек"),
        ("в продаваемой капсуле есть спички", "в продаваемой капсуле спичек нет"),
    ],
    "standalone_solid_fuel": [
        ("карбонизированный уголь как основное топливо", "древесные роллы только для розжига отдельного топлива"),
        ("антрацит является продаваемым твёрдым топливом", "деревянные бруски служат лишь стартером для другого топлива"),
        ("упаковка содержит основное угольное топливо", "упаковка содержит только древесную растопку"),
        ("карбонизированный материал предназначен для отопления", "некарбонизированный материал предназначен лишь для розжига"),
        ("продаётся готовое твёрдое топливо из угля", "продаются древесные роллы для поджига отдельного топлива"),
    ],
    "pyro_popper": [
        ("пиротехнический горючий заряд", "пневматический механизм без пиросостава"),
        ("эффект создаёт пороховой пирозаряд", "эффект создаёт сжатый воздух без пирозаряда"),
        ("внутри находится воспламеняемый пиросостав", "внутри только пружина, пиросостава нет"),
        ("срабатывание от пиротехнического заряда", "срабатывание от механической пружины без заряда"),
        ("хлопок создаёт горючая пиротехническая смесь", "хлопок создаёт пневматика без горючей смеси"),
    ],
    "pyro_cake_or_smoke": [
        ("пиротехнический искровой заряд с горючим составом", "обычная восковая свеча без пиротехнического состава"),
        ("горючий пиросостав создаёт фонтан искр", "обычный фитиль даёт только пламя, пиросостава нет"),
        ("внутри находится пиротехнический дымовой заряд", "это простой декор без пиротехнического заряда"),
        ("искры создаёт воспламеняемый пиротехнический состав", "это восковая свеча без искрового пиросостава"),
        ("эффект создаёт горючий фонтанный пирозаряд", "эффект ограничен обычным пламенем свечи без пирозаряда"),
    ],
    "integrated_match_firestarter": [
        ("в каждый ролл встроена спичка", "нужна отдельная спичка, которой в комплекте нет"),
        ("каждый элемент содержит собственный воспламенитель", "воспламенитель приобретается отдельно"),
        ("спичка уже вложена в каждый элемент розжига", "в элементы спички не вложены"),
        ("роллы имеют встроенный источник поджига", "роллы требуют внешней зажигалки"),
        ("в комплект каждого бруска входит спичка", "комплект брусков продаётся без спичек"),
    ],
    "grill_or_heater_with_fuel": [
        ("заполненный газовый картридж с топливом входит в комплект", "топливного картриджа в комплекте нет"),
        ("комплект содержит заправленный баллон горючего газа", "баллон приобретается отдельно и в комплект не входит"),
        ("в набор вложена заполненная топливная кассета", "топливная кассета в наборе отсутствует"),
        ("устройство поставляется вместе с заправленным картриджем", "устройство поставляется без картриджа и топлива"),
        ("готовое топливо включено в продаваемый комплект", "топливо не включено и покупается отдельно"),
    ],
    "standalone_ignition_source": [
        ("в корпусе находится рабочая зажигалка", "корпус пустой, зажигалки внутри нет"),
        ("в комплект входят настоящие спички", "продаётся только пустой футляр без спичек"),
        ("внутри находится рабочее огниво", "внутри нет огнива, продаётся один чехол"),
        ("продаваемая единица содержит источник огня", "продаётся лишь пустой держатель"),
        ("капсула укомплектована рабочими спичками", "капсула продаётся пустой, без спичек"),
    ],
}


COLORS = ("графитовое", "серое", "песочное", "оливковое", "синее", "бордовое", "бежевое", "чёрное", "белое", "терракотовое")
MATERIALS = ("картон", "крафт-бумага", "матовая плёнка", "переработанная бумага", "ламинированная бумага")
SHAPES = ("прямоугольная", "квадратная", "узкая", "плоская")
VARIANT_NOTES = tuple(
    f"Оформление упаковки {color}; вкладыш — {material}; форма этикетки {shape}."
    for color, material, shape in itertools.product(COLORS, MATERIALS, SHAPES)
)
assert len(VARIANT_NOTES) == 200


CAPTIONS = {
    "self_contained_torch": "Подача топлива",
    "sold_fuel_gas": "Состояние содержимого",
    "included_flammable": "Комплектация",
    "matches_present": "Содержимое футляра",
    "standalone_solid_fuel": "Роль материала",
    "pyro_popper": "Механизм срабатывания",
    "pyro_cake_or_smoke": "Тип эффекта",
    "integrated_match_firestarter": "Средство поджига",
    "grill_or_heater_with_fuel": "Комплектация топливом",
    "standalone_ignition_source": "Содержимое корпуса",
}


AUTHOR_PROMPT = """Ты создаёшь только нейтральные товарные fact frames.
Для каждой задачи верни новое естественное русское invariant_name и
invariant_description. Они описывают заданный тип товара и стиль, но вообще не
говорят о decisive slot: не утверждают наличие/отсутствие топлива, спичек,
зажигалки, пиросостава, встроенного бака или комплектного картриджа. Код сам
добавит обе противоположные версии слота и все служебные metadata.

Не используй реальные бренды, продавцов, SKU, organizer/организатор, метки,
классы, P/N, датасет, соревнование, «указано в описании», «уточняется при
заказе» и рекламные оговорки. Не копируй реальные карточки или другие задания.
Верни только JSON; из identifiers повтори только pair_id."""


AUTHOR_SCHEMA = {"frames": [{"pair_id": "...", "invariant_name": "...", "invariant_description": "..."}]}


def production_tasks() -> list[dict[str, Any]]:
    result = []
    for index, source in enumerate(BASE_PRODUCTION_TASKS()):
        row = dict(source)
        row["pair_id"] = f"r5-deterministic-{index:03d}"
        row["global_variant_index"] = index
        row["style"] = str(row["style"]) + "; fresh deterministic-slot corpus"
        result.append(row)
    return result


def author_user(tasks: list[Mapping[str, Any]]) -> str:
    payload = []
    for task in tasks:
        boundary = str(task["boundary"])
        payload.append({
            "pair_id": task["pair_id"], "style": task["style"],
            "product_boundary": boundary,
            "product_subject_guidance": core.BOUNDARIES[boundary]["forbidden_cochanges"],
            "neutrality_requirement": "do not state either positive or negative slot value",
        })
    return "Создай нейтральные frames:\n" + canonical_json({"tasks": payload}) + "\n\nСхема:\n" + canonical_json(AUTHOR_SCHEMA)


def validate_frames(payload: Mapping[str, Any], tasks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {str(row["pair_id"]): row for row in tasks}
    frames = payload.get("frames")
    if not isinstance(frames, list) or {str(row.get("pair_id")) for row in frames} != set(expected):
        raise ValueError("frame pair_id coverage drift")
    result = []
    for source in frames:
        source_row = dict(source)
        pair_id = str(source_row["pair_id"])
        task = expected[pair_id]
        if any(not str(source_row.get(key, "")).strip() for key in ("invariant_name", "invariant_description")):
            raise ValueError(f"incomplete neutral frame {pair_id}")
        index = int(task["global_variant_index"])
        boundary = str(task["boundary"])
        positive_slot, negative_slot = SLOT_BANKS[boundary][int(task["variant_index"]) % 5]
        description = str(source_row["invariant_description"]).strip().rstrip(".!?;:") + ". " + VARIANT_NOTES[index]
        result.append({
            "pair_id": pair_id, "style_frame_id": task["style_frame_id"],
            "boundary": boundary, "changed_field": task["changed_field"],
            "variant_index": task["variant_index"], "changed_slot": core.BOUNDARIES[boundary]["slot"],
            "invariant_name": str(source_row["invariant_name"]).strip(),
            "invariant_description": description,
            "positive_slot_text": positive_slot, "negative_slot_text": negative_slot,
            "invariant_fact_frame": str(source_row["invariant_name"]).strip() + " | " + description,
        })
    return sorted(result, key=lambda row: str(row["pair_id"]))


def render_frames(frames: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Render both pair endpoints from neutral OpenRouter frames."""

    rendered: list[dict[str, Any]] = []
    for frame in frames:
        row = {
            key: frame[key]
            for key in (
                "pair_id",
                "style_frame_id",
                "boundary",
                "changed_slot",
                "changed_field",
                "variant_index",
                "invariant_fact_frame",
            )
        }
        row["positive_first"] = int(str(frame["pair_id"]).rsplit("-", 1)[-1]) % 2 == 0
        changed_field = str(frame["changed_field"])
        for role, slot_key in (
            ("positive", "positive_slot_text"),
            ("negative", "negative_slot_text"),
        ):
            slot = str(frame[slot_key]).strip()
            name = str(frame["invariant_name"]).strip()
            description = str(frame["invariant_description"]).strip()
            if changed_field == "name":
                name = f"{name} — {slot}"
            else:
                description = f"{description} {CAPTIONS[str(frame['boundary'])]}: {slot}."
            row[role] = {
                "name": name,
                "description": description,
                "decisive_span": slot,
            }
        rendered.append(row)
    return rendered


def freeze_payload(stage: str) -> dict[str, Any]:
    if stage != "production":
        raise ValueError("only production is available")
    tasks = production_tasks()
    return {
        "schema_version": "r5.deterministic_slots.production.freeze.1",
        "experiment_id": "quality_r5_deterministic_slots_full_200_production",
        "frozen_before_api_call": True,
        "model": "qwen/qwen3.5-122b-a10b",
        "model_config_sha256": sha256_file(core.MODEL_CONFIG),
        "author_prompt_sha256": sha256_text(AUTHOR_PROMPT),
        "author_schema_sha256": sha256_text(canonical_json(AUTHOR_SCHEMA)),
        "review_prompt_sha256": sha256_text(core.REVIEW_PROMPT),
        "tasks_sha256": sha256_text(canonical_json(tasks)),
        "boundaries_sha256": sha256_text(canonical_json(core.BOUNDARIES)),
        "frame_registry_sha256": sha256_text(canonical_json(core.FRAME_REGISTRY)),
        "captions_sha256": sha256_text(canonical_json(CAPTIONS)),
        "slot_banks_sha256": sha256_text(canonical_json(SLOT_BANKS)),
        "variant_notes_sha256": sha256_text(canonical_json(VARIANT_NOTES)),
        "renderer_version": "deterministic-caption-v2",
        "tasks": 200, "boundaries": 10, "style_frames": 44, "batch_size": 10,
        "gates": {
            "schema_success_rate_min": 1.0,
            "structural_pass_rate_min": 1.0,
            "blind_direction_accuracy_min": 0.95,
            "blind_usable_rate_min": 0.95,
            "exact_train_overlap_max": 0,
            "copied_13_token_train_spans_max": 0,
            "exact_duplicate_cards_max": 0,
            "exact_duplicate_masked_pairs_max": 0,
            "cross_frame_near_duplicate_edges_ge_0_94_max": 0,
            "global_all_attempt_cost_usd_max": core.GLOBAL_API_CAP_USD,
        },
        "forbidden_runtime_inputs": [
            "generated rows, responses, reviews or rendered cards from another run",
            "supplementary training row texts, rationales or predictions",
            "public/private leaderboard rows, labels, predictions or IDs",
            "artifacts not produced by this run",
        ],
        "immutability_rule": "Do not edit prompts, slot banks, variant notes, tasks, renderer or gates during a run.",
    }


def configure_core() -> None:
    core.AUTHOR_PROMPT = AUTHOR_PROMPT
    core.AUTHOR_SCHEMA = AUTHOR_SCHEMA
    core.PRODUCTION_FREEZE = FREEZE
    core.production_tasks = production_tasks
    core.author_user = author_user
    core.validate_frames = validate_frames
    core.render_frames = render_frames
    core.freeze_payload = freeze_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("production",), default="production")
    parser.add_argument("--model-config", type=Path, default=core.MODEL_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-train", type=Path)
    parser.add_argument("--print-freeze", action="store_true")
    args = parser.parse_args()
    configure_core()
    if args.print_freeze:
        print(json.dumps(freeze_payload("production"), ensure_ascii=False, indent=2))
        return
    if args.source_train is None:
        parser.error("--source-train is required because organizer train is not bundled")
    result = core.run(
        "production",
        args.model_config.resolve(),
        args.output.resolve(),
        args.source_train.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
