# E-CUP Quality — пакет воспроизведения

Пакет содержит всё необходимое для генерации синтетических данных, обучения,
проверки производных результатов и запуска готового решения. Не включён только
исходный organizer train как набор строк: исходные названия, описания, метки и
изображения. Получатель подставляет свой экземпляр organizer train по ID.

## Что означают R4, R5 и R7

**R4 (Round 4)** — 100 контрастных ID-отношений между строками organizer train.
R4 используется в pairwise loss обеих LoRA-веток. В пакете находится только
ID-разметка без исходных названий, описаний, меток и изображений.

**R5 (Round 5)** — 200 синтетических контрастных пар, то есть 400 полностью
синтетических карточек. Данные R5 изначально сгенерированы через OpenRouter
моделью `qwen/qwen3.5-122b-a10b` и используются только в pairwise loss обеих
LoRA-веток.

**R7 (Round 7)** — 100 query-specific ID-отношений между строками organizer
train для candidate-ветки. Для q0–q3 используются соответственно 22, 33, 30 и
15 экземпляров; для q4 и full-fit они дедуплицируются до 62 `mass_key`. В пакете
находится только ID-разметка без исходных текстов, меток и изображений.

## Состав пакета

- `organizer_annotations/` — готовые ID-only разметки R4 и R7 с manifests;
- `data_generation/openrouter_qwen122b/` — prompts, JSON schemas, OpenRouter
  runner, model/provider pins и проверки генерации R5;
- `reproduction_outputs/openrouter_qwen122b_run/` — готовый проверенный
  OpenRouter-run R5;
- `training_code/` — подготовка manifests, обучение LoRA, OOF-scoring и
  обучение CatBoost-selector;
- `reference_training_outputs/` — производные OOF-прогнозы обеих веток,
  полный CPU OOF, audit, q4 gate, OOF/full-fit receipts и готовый
  CatBoost-selector;
- `final_runtime/` — готовый inference ZIP с финальными LoRA-весами,
  CatBoost-selector, исходниками runtime и проверкой целостности.
- `RELEASE_MANIFEST.json` — SHA-256 и размер каждого передаваемого файла пакета.

OOF-файлы содержат только производные поля `id`, `blind_uid`,
`component_key`, `fold`, `label`, `score`, `prediction`; исходных названий,
описаний и изображений в них нет.

## Порядок воспроизведения

1. Подставить organizer train по схеме из `REPRODUCE_FROM_SCRATCH_RU.md`.
2. Скопировать вложенные R4/R7 в локальный рабочий каталог.
3. Использовать готовый OpenRouter R5 или запустить его генерацию заново.
4. Построить CE/R4/R5/R7 manifests и выполнить CPU dry-run.
5. При необходимости переобучить LoRA и CatBoost-selector; готовые производные
   OOF и selector можно использовать как эталон для проверки.
6. Проверить и запустить готовый inference runtime.

Точные команды приведены в `REPRODUCE_FROM_SCRATCH_RU.md` и
`training_code/README_RU.md`.
