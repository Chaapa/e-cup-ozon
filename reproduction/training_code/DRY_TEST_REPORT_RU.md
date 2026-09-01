# Проверка кода обучения

Дата: 2026-08-31.

R4 — общий набор из 100 контрастных ID-отношений; R5 — общий синтетический
набор OpenRouter из 200 пар и 400 карточек; R7 — 100 query-specific ID-экземпляров
и 62 `mass_key` q4/full-fit только для candidate. CE содержит 5 502
классификационные строки organizer train.

Проверено:

- импорт и компиляция Python-модулей;
- manifests CE/R4/R5/R7;
- OpenRouter provenance R5;
- расписания control/candidate;
- q4 isolation;
- OOF inventory;
- контракт 71 признака CatBoost-selector;
- CPU-аудит без модели, GPU, API и сети.

Команда:

```bash
python3 -m pytest -q training_code/tests
```

Результат проверки пакета: `11 passed`.

CPU-аудит выполняется командой из `training_code/README_RU.md` и должен
возвращать `PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.
