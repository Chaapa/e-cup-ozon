# ID-разметки organizer train

В каталоге находятся только производные ID-разметки. Здесь нет исходных
названий, описаний, меток, изображений или полных строк organizer train. Текст,
метка, складка и компонент каждой конечной точки разрешаются из подставленного
organizer-train Parquet во время проверки manifest.

## Что такое R4 и какие файлы входят

R4 (Round 4) — 100 контрастных отношений для pairwise loss обеих LoRA-веток.
`r4/r4_pairs.jsonl` хранит только `pair_id`, `positive_id`, `negative_id`,
`boundary_code`, `mass_key`, `review_status` и `eligible_query_folds`.
`r4/annotation_manifest.json` фиксирует размер и SHA-256 файла.

## Что такое R7 и какие файлы входят

R7 (Round 7) — query-specific контрастная разметка только для candidate-ветки.
`r7/r7_relations.jsonl` содержит 100 принятых экземпляров: 22 для q0, 33 для
q1, 30 для q2 и 15 для q3. Один `mass_key` может встречаться в нескольких
query-fold; расписания q4 и full-fit дедуплицируют набор до 62 `mass_key`.
Каждая запись хранит только ID и метаданные отношения.
`r7/annotation_manifest.json` фиксирует размер и SHA-256 файла.

Перед запуском `training_code.prepare_manifests` скопируйте оба JSONL рядом с
organizer-train Parquet по командам из корневой инструкции воспроизведения.
