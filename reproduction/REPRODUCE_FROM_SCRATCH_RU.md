# Воспроизведение

## Что означают R4, R5 и R7

**R4 (Round 4)** — 100 контрастных ID-отношений organizer train для pairwise
loss обеих LoRA-веток.

**R5 (Round 5)** — 200 синтетических пар и 400 карточек, изначально
сгенерированных через OpenRouter только для pairwise loss обеих LoRA-веток.

**R7 (Round 7)** — 100 query-specific ID-отношений organizer train только для
candidate-ветки. В q0–q3 используются 22/33/30/15 экземпляров, а q4 и full-fit
используют 62 дедуплицированных `mass_key`.

## 1. Подставить исходный organizer train

В пакете нет только исходных строк organizer train: названий, описаний, меток и
изображений. Укажите путь к выданному организатором FLV Parquet и подготовьте
локальный рабочий каталог:

```bash
export REPRO_WORKDIR=/absolute/path/to/reproduction_workdir
mkdir -p "$REPRO_WORKDIR"
export ORGANIZER_TRAIN_FLV="$REPRO_WORKDIR/organizer_train_flv.parquet"
```

Обязательные столбцы:

```text
id, blind_uid, label, fold, component_key, clean_name, clean_description,
normalized_name, normalized_description, image_count
```

Ожидается 5 502 строки, включая 198 положительных. Если исходный столбец складки
называется иначе, переименуйте его в `fold` до создания manifest.

## 2. Подключить вложенные ID-разметки

R4 и R7 уже находятся в пакете и не требуют отдельной поставки. Загрузчик
разрешает их конечные точки по `positive_id` и `negative_id` в подставленном
organizer train; тексты и метки в JSONL не хранятся.

```bash
cp organizer_annotations/r4/r4_pairs.jsonl "$REPRO_WORKDIR/r4_pairs.jsonl"
cp organizer_annotations/r7/r7_relations.jsonl "$REPRO_WORKDIR/r7_relations.jsonl"
export ORGANIZER_R4_PAIRS="$REPRO_WORKDIR/r4_pairs.jsonl"
export ORGANIZER_R7_PAIRS="$REPRO_WORKDIR/r7_relations.jsonl"
```

## 3. Использовать или заново сгенерировать OpenRouter R5

Готовый проверенный OpenRouter-run находится в
`reproduction_outputs/openrouter_qwen122b_run/`. Его параметры:

- model: `qwen/qwen3.5-122b-a10b`;
- provider: `novita`;
- quantization: `bf16`;
- temperature: `0`;
- parallel workers: `4`;
- API hard cap: `$10`.

Чтобы использовать готовые данные:

```bash
cp -R reproduction_outputs/openrouter_qwen122b_run \
  "$REPRO_WORKDIR/r5_production"
export R5_ARTIFACT_DIR="$REPRO_WORKDIR/r5_production"
```

Чтобы полностью повторить генерацию через OpenRouter:

```bash
cd data_generation/openrouter_qwen122b
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY='...'
PYTHONPATH=. python -m quality_bench.r5_deterministic_slots_production \
  --stage production \
  --source-train "$ORGANIZER_TRAIN_FLV" \
  --output reproduced_runs/qwen122b-production
```

Ожидаемый результат: 200 пар, 400 карточек, structural `200/200`, blind
direction `200/200`, usable `192/200`. Для нового run задайте его каталог в
`R5_ARTIFACT_DIR`.

## 4. Построить manifests и проверить расписание

Создайте CE/R4/R5/R7 manifests и выполните `organizer_cpu_dry_run` командами из
раздела «Неизменяемая подготовка» в `training_code/README_RU.md`. Проверка
должна завершиться статусом `PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.

## 5. Обучить модели или проверить готовые производные файлы

Базовая модель: `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Полная последовательность команд обучения находится в
`training_code/README_RU.md`:

1. десять control/candidate LoRA OOF-траекторий;
2. OOF-scoring трёх checkpoints каждой траектории;
3. q4 gate;
4. CatBoost-selector на 71 target-free признаке;
5. две full-fit LoRA-траектории;
6. сборка model bundle.

В `reference_training_outputs/` уже включены все несырые производные результаты:

- 30 OOF Parquet и их self-hashed manifests;
- 10 receipts траекторий;
- development и полный пятискладочный CPU OOF;
- переносимые OOF audit и q4 gate report;
- два full-fit receipt, связывающие шесть LoRA-checkpoints внутри ZIP;
- готовый CatBoost-selector и его manifest.

Проверка вложенного OOF набора после создания CE manifest:

```bash
PYTHONPATH=. python -m training_code.fit_selector \
  --phase audit \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --oof-root reference_training_outputs/oof \
  --output-dir "$REPRO_WORKDIR/reference_oof_audit"
```

Финальные LoRA-веса не дублируются отдельным каталогом: они уже находятся в
готовом inference ZIP.

## 6. Проверить итоговый runtime

```bash
python final_runtime/runtime/source/verify_submission.py \
  final_runtime/runtime/ecup_quality_runtime.zip
```

Контракт запуска:

```bash
python3 -u run.py -i TEST_CSV -o SOLUTION_CSV
```
