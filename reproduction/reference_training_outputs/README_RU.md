# Производные результаты обучения

В каталоге находятся эталонные артефакты, производные от organizer train. Здесь
нет исходных названий, описаний, изображений или полных строк organizer train.

## Что такое OOF и какие файлы входят

OOF (out-of-fold) — прогноз для строки, полученный траекторией, в CE-обучение
которой не входила query-складка этой строки. В `oof/` находятся ветки
`control` и `candidate`, пять query-fold для каждой ветки и checkpoints 25%,
50% и 100%: всего 30 Parquet с прогнозами. Каждый Parquet содержит только
`id`, `blind_uid`, `component_key`, `fold`, `label`, `score`, `prediction`.
Self-hashed manifests и десять receipts фиксируют все файлы.

`development_cpu_oof.local_only.parquet` содержит 4 401 производную строку
q0–q3 с CPU-сигналами, которые объединяются с LoRA OOF.
`cpu_oof.local_only.parquet` и `cpu_oof_manifest.json` содержат полный
пятискладочный CPU OOF на 5 502 строки. Исходный текст заменён хешами и
числовыми выходами моделей; текстовых поверхностей в файлах нет.

`audits/oof_audit.json` фиксирует переносимую проверку 30 OOF-файлов.
`gates/q4_gate_report.json` фиксирует результат q4 `GO`, рассчитанный по
вложенному OOF. Оба файла self-hashed и не содержат исходного текста.

## Что такое selector и какие файлы входят

CatBoost-selector объединяет 71 target-free inference-признак базовых сигналов
и control/candidate LoRA-траекторий. Готовая модель находится в
`selector/selector_augmented_fold01234.cbm`; её SHA-256 совпадает с selector
внутри `final_runtime/runtime/ecup_quality_runtime.zip`.

`artifact_manifest.json` фиксирует полный derivative-only состав. Финальные
LoRA-веса уже находятся внутри inference ZIP и здесь не дублируются.
`full_fit/control_fit_receipt.json` и
`full_fit/candidate_fit_receipt.json` связывают шесть checkpoint-файлов внутри
ZIP по размеру и SHA-256.
