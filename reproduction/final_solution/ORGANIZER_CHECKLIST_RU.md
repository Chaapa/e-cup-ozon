# Чек-лист организатора

**Наборы данных.** R4 — 100 ID-only контрастных отношений organizer train для
обеих веток. R5 — 200 синтетических пар и 400
карточек, изначально сгенерированных через OpenRouter, только для общего
pairwise loss. R7 — 100 query-specific ID-only отношений organizer train,
объединённых в 62 `mass_key` q4/full-fit только для candidate-ветки.

1. Подготовить organizer train по CE-схеме из `training_code/README_RU.md`.
2. Скопировать вложенные ID-only разметки R4 и R7 из `organizer_annotations/`.
3. Проверить семь файлов в `reproduction_outputs/openrouter_qwen122b_run/`.
4. Построить самохешируемые manifest-файлы CE/R4/R5/R7.
5. Выполнить `organizer_cpu_dry_run` и получить
   `PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.
6. Зафиксировать ревизию базовой модели и окружение.
7. Проверить вложенные эталонные OOF или заново запустить control/candidate OOF
   по пяти складкам.
8. Обучить selector и две полные LoRA-ветки.
9. Собрать пакет и проверить все SHA-256.
