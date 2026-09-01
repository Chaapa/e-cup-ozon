# Краткая инструкция по обучению

R4 (Round 4) — 100 контрастных ID-отношений organizer train для обеих
LoRA-веток. R5 (Round 5) — 200 синтетических пар и 400 карточек, изначально
сгенерированных через OpenRouter только для pairwise loss. R7 (Round 7) — 100
query-specific ID-отношений organizer train, дедуплицируемых до 62 `mass_key`
для q4/full-fit и используемых только candidate-веткой.

Извне подключается только исходный organizer CE Parquet. JSONL R4/R7 уже
находятся в `organizer_annotations/`; перед созданием manifests скопируйте их в
`REPRO_WORKDIR` рядом с Parquet.

## Данные

- CE: 5 502 строки organizer train, 198 положительных;
- R4: 100 real-real пар для control и candidate;
- R5: 200 синтетических пар OpenRouter для control и candidate;
- R7: 100 query-specific экземпляров / 62 `mass_key` для q4-full-fit, только candidate.

R5 используется только в pairwise loss и не добавляется в CE.

## Параметры

- base model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- LoRA rank 16, alpha 32;
- одна эпоха CE;
- effective batch 32;
- R4/R5 weights: `0.02/0.02`;
- R7 candidate weight: `0.03`;
- checkpoints: 25%, 50% и 100%;
- CatBoost: depth 4, 800 iterations, learning rate 0.03.

## Порядок запуска

1. Создать self-hashed manifests CE/R4/R5/R7.
2. Выполнить `organizer_cpu_dry_run`.
3. Обучить control/candidate LoRA по пяти OOF-фолдам.
4. Проверить q4 gate.
5. Обучить CatBoost-selector на полных OOF-признаках.
6. Выполнить full-fit обеих LoRA-веток.
7. Собрать checkpoints, selector, environment lock и manifests через
   `build_bundle.py`.

В `reference_training_outputs/` уже находятся derivative-only 30 OOF-файлов,
десять receipts, development CPU OOF и готовый selector. Исходных текстов и
изображений organizer train там нет.

CPU-аудит:

```bash
export REPRO_WORKDIR=/absolute/path/to/reproduction_workdir
PYTHONPATH=. python -m training_code.organizer_cpu_dry_run \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --r4-manifest "$REPRO_WORKDIR/r4_manifest.json" \
  --r5-manifest "$REPRO_WORKDIR/r5_manifest.json" \
  --r7-manifest "$REPRO_WORKDIR/r7_manifest.json" \
  --recipe-precommit "$REPRO_WORKDIR/recipe_precommit.json" \
  --output "$REPRO_WORKDIR/organizer_cpu_dry_run.json"
```

Ожидаемый статус:
`PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.

Обучение останавливается при несовпадении SHA-256, неполном OOF-составе,
непройденном q4 gate, изменении base model или попадании данных внешней оценки.
