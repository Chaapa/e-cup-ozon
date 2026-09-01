# Обучение FLV: LoRA и CatBoost-selector

**Что означают R4, R5 и R7.** R4 (Round 4) — 100 контрастных ID-отношений
organizer train для обеих веток. R5 (Round 5) — 200 синтетических пар и 400
карточек, изначально сгенерированных через OpenRouter, только для pairwise loss
обеих веток. R7 (Round 7) — query-specific ID-отношения organizer train только
для candidate-ветки: 100 экземпляров для q0–q3 и 62 дедуплицированных
`mass_key` для q4/full-fit.

Каталог содержит код подготовки данных, обучения двух LoRA-веток, построения
OOF-прогнозов и обучения CatBoost-selector. В пакет уже входят ID-разметки
R4/R7, эталонные OOF-прогнозы, CatBoost-selector и финальные LoRA-веса внутри
inference ZIP. Извне подключаются только исходные строки organizer train,
снимок базовой модели, API caches и секреты.

## Контур обучения

1. Пять складок без общих компонентов из 5 502 FLV-строк organizer train (198
   положительных): q0–q3 — складки разработки, q4 — замороженная
   forward-проверка. OOF-обучение q0–q3 использует только остальные складки
   разработки; метки q4 никогда не входят в траектории разработки. Траектория
   q4 обучается только на q0–q3.
2. Согласованная схема control/candidate с Qwen3.5-4B и rank 16. Обе ветки
   получают одну эпоху текстового CE, пары R4 из organizer train с весом 0,02 и
   синтетические пары R5, сгенерированные через OpenRouter, с весом 0,02. Только
   candidate получает пары R7 из organizer train с весом 0,03; control привязывает тот
   же manifest R7 с нулевой массой.
3. Три контрольные точки на 25%, 50% и 100% каждой траектории.
4. Пятискладочная OOF-оценка обеих веток. Метки запросов никогда не определяют
   порог контрольной точки LoRA: бинарные прогнозы используют ранг
   распространённости положительного класса на соответствующей внешней
   обучающей выборке.
5. Замороженная transfer-проверка q4. Рецепт замораживается до любой оценки q4;
   q4 не выбирает строки R7, вес, rank, prompt или рецепт контрольных точек.
6. Один финальный CatBoost-selector, обучаемый на всех пяти OOF-складках только
   после результата GO на q4. Он получает 71 признак без целевой метки: B1,
   character-SVC, три контрольные точки
   control, три контрольные точки candidate и геометрию траектории
   candidate-minus-control. Признаки P/N и дополнительные rule-based признаки
   в selector не входят.
7. Отдельные полные обучения control и candidate и сборка итогового пакета с
   SHA-256 manifest.

## Необходимые входные данные

Все manifest-файлы хешируют сами себя и привязывают каждый файл данных по пути,
размеру в байтах и SHA-256. CE/R4/R7 объявляют происхождение из organizer train;
R5 объявляет OpenRouter model/provider provenance.

### CE Parquet

Обязательные столбцы:

`id, blind_uid, label, fold, component_key, clean_name, clean_description,
normalized_name, normalized_description, image_count`

Каждый `component_key` должен принадлежать ровно одной складке. Тренер выбирает
только эти столбцы и отклоняет столбцы внешней оценки.

### R4 JSONL

Каждая принятая запись содержит:

`pair_id, positive_id, negative_id, boundary_code, mass_key, review_status,
eligible_query_folds`

Загрузчик получает текст и метки из CE Parquet, требует направление 1→0,
разные компоненты, одну складку для обеих конечных точек в замороженных складках
разработки q0–q3, одно отношение на пару компонентов и ровно 100 принятых пар.
Складка forward-проверки q4 запрещена для всех конечных точек дополнительных
реальных пар.

### Производственные артефакты R5

R5 — производственный контракт только для pairwise-обучения из шести файлов:

- `rendered_pairs.jsonl`: положительные/отрицательные карточки, решающие
  фрагменты, изменённый слот, изменённое поле, инвариантный фактический кадр,
  граница, стилевой кадр и случайный порядок отображения;
- `blind_reviews.jsonl`: метки A/B, процитированные доказательства, решение о
  пригодности и флаг/причина рецензента для каждой пары;
- `structural_audit.jsonl`: структурные проверки каждой пары;
- `training_cards.jsonl`: ровно две полностью синтетические карточки с метками
  организатора, привязанные к каждой отображённой паре;
- `metrics.json`: агрегированная проверка направления, пригодности, пересечений
  и состава;
- `artifact_manifest.json`: SHA-256 и размер остальных пяти файлов.

Повторно созданный производственный результат — ровно 200/200 пар, прошедших
структурную проверку, и 200/200 правильных направлений в слепой проверке. Слепая
проверка признаёт пригодными 192/200 (96%, выше замороженного порога корпуса
95%) и сохраняет восемь флагов рецензента. Восемь отмеченных пар намеренно
**не отфильтровываются**: все 200 пар входят в pairwise-расписание R5, а ID и
причины флагов остаются привязанным хешами аудиторским доказательством. Порог
применяется агрегированно; это не построчное переписывание допуска после
генерации.

Загрузчик отклоняет ID/метки организатора, точное повторное использование
поверхностей organizer train, общие фрагменты train длиной 13 токенов, любую
непройденную структурную проверку или проверку направления, более пяти пар на
namespaced boundary/style frame, менее 40 кадров, расхождение состава между
четырьмя JSONL, а также любую несогласованность артефактов, метрик или хешей. R5
никогда не входит в CE.

### R7 JSONL

Каждая запись содержит ID-поля R4 и дополнительно `query_fold`. Во вложенном
файле ровно 100 query-specific экземпляров: q0=22, q1=33, q2=30, q3=15;
`eligible_query_folds` должен совпадать с `[query_fold]`. Один `mass_key` может
повторяться между разными query-fold, но не внутри одной query-fold. q4 и
full-fit используют по одному экземпляру на `mass_key`, то есть 62 отношения.
Текст, метки, folds и компоненты конечных точек разрешаются по ID из CE Parquet.
В full-fit масса отдельно ограничивается по повторному использованию
положительных и отрицательных компонентов.

Используйте `prepare_manifests.py`, чтобы создать и сразу проверить
manifest-файлы. Входные файлы должны находиться внутри каталога, содержащего их
manifest.

## Неизменяемая подготовка

Подставьте только organizer CE Parquet. R4 и R7 уже вложены как ID-only JSONL;
скопируйте их в тот же новый локальный рабочий каталог, потому что manifest
привязывает файлы внутри своего каталога. Загрузчик получает тексты, метки,
folds и компоненты конечных точек из Parquet. `R5_ARTIFACT_DIR` направьте на
скопированный OpenRouter R5 или на новый production-run OpenRouter.

```bash
export REPRO_WORKDIR=/absolute/path/to/reproduction_workdir
mkdir -p "$REPRO_WORKDIR"
export ORGANIZER_TRAIN_FLV="$REPRO_WORKDIR/organizer_train_flv.parquet"
cp organizer_annotations/r4/r4_pairs.jsonl "$REPRO_WORKDIR/r4_pairs.jsonl"
cp organizer_annotations/r7/r7_relations.jsonl "$REPRO_WORKDIR/r7_relations.jsonl"
cp -R reproduction_outputs/openrouter_qwen122b_run "$REPRO_WORKDIR/r5_production"
export ORGANIZER_R4_PAIRS="$REPRO_WORKDIR/r4_pairs.jsonl"
export ORGANIZER_R7_PAIRS="$REPRO_WORKDIR/r7_relations.jsonl"
export R5_ARTIFACT_DIR="$REPRO_WORKDIR/r5_production"

PYTHONPATH=. python -m training_code.prepare_manifests ce \
  --config training_code/config.json \
  --rows "$ORGANIZER_TRAIN_FLV" \
  --output "$REPRO_WORKDIR/ce_manifest.json"

PYTHONPATH=. python -m training_code.prepare_manifests pairs \
  --kind r4 --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --pairs "$ORGANIZER_R4_PAIRS" --output "$REPRO_WORKDIR/r4_manifest.json"

PYTHONPATH=. python -m training_code.prepare_manifests pairs \
  --kind r7 --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --pairs "$ORGANIZER_R7_PAIRS" --output "$REPRO_WORKDIR/r7_manifest.json"

PYTHONPATH=. python -m training_code.prepare_manifests r5-production \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --artifact-dir "$R5_ARTIFACT_DIR" \
  --output "$REPRO_WORKDIR/r5_manifest.json"

PYTHONPATH=. python -m training_code.freeze_recipe \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --r4-manifest "$REPRO_WORKDIR/r4_manifest.json" \
  --r5-manifest "$REPRO_WORKDIR/r5_manifest.json" \
  --r7-manifest "$REPRO_WORKDIR/r7_manifest.json" \
  --output "$REPRO_WORKDIR/recipe_precommit.json"

# Обязательная предварительная проверка только на CPU: проверяет все строки,
# хеши, агрегированный порог R5, сохранение флагов рецензента, замкнутость
# складок и согласованность расписаний, не импортируя модель и не используя GPU.
PYTHONPATH=. python -m training_code.organizer_cpu_dry_run \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --r4-manifest "$REPRO_WORKDIR/r4_manifest.json" \
  --r5-manifest "$REPRO_WORKDIR/r5_manifest.json" \
  --r7-manifest "$REPRO_WORKDIR/r7_manifest.json" \
  --recipe-precommit "$REPRO_WORKDIR/recipe_precommit.json" \
  --output "$REPRO_WORKDIR/organizer_cpu_dry_run.json"
```

В CPU-отчёте должно быть указано
`PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`, количества R5 `200 / 192 / 8`, привязка
десяти OOF-траекторий и двух расписаний полного обучения, а также
`gpu_loaded=false`, `model_loaded=false`, `fit_executed=false` и ноль вызовов
API/сети. Эту проверку безопасно запускать до оплаты вычислений.

Базовую модель необходимо загрузить на ревизии
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; создайте полный контракт её
локальных файлов командой `prepare_manifests.py base-snapshot`. На изолированной
H100 создавайте `execution-receipt` только после проверки неизменяемого
контейнера и видимого GPU. Команда обучения требует
`network_disabled_during_fit=true`.

## Граф выполнения OOF

Для каждой складки запроса 0…4 дважды выполните следующие команды, меняя только
`--arm` и каталог результата:

```bash
PYTHONPATH=. python -m training_code.train_lora \
  --phase audit --arm control --query-fold 0 \
  --config ... --ce-manifest ... --r4-manifest ... --r5-manifest ... \
  --r7-manifest ... --recipe-precommit ... --output-dir /runs/audit

PYTHONPATH=. python -m training_code.train_lora \
  --phase fit --arm control --query-fold 0 \
  --config ... --ce-manifest ... --r4-manifest ... --r5-manifest ... \
  --r7-manifest ... --recipe-precommit ... --model-path ... \
  --base-snapshot-manifest ... --execution-receipt ... \
  --output-dir /runs/control/fold_0_fit

PYTHONPATH=. python -m training_code.score_lora \
  --config ... --ce-manifest ... --fit-dir /runs/control/fold_0_fit \
  --model-path ... --base-snapshot-manifest ... \
  --output-dir /runs/oof/control/fold_0
```

Когда готовы все десять траекторий и результатов оценки, выполните проверку q4:

```bash
PYTHONPATH=. python -m training_code.evaluate_q4_gate \
  --config ... --ce-manifest ... --recipe-precommit ... \
  --oof-root /runs/oof --output-dir /runs/q4_gate
```

Только отчёт `GO` разрешает финальное обучение selector:

```bash
PYTHONPATH=. python -m training_code.fit_selector \
  --phase fit --config ... --ce-manifest ... --oof-root /runs/oof \
  --gate-report /runs/q4_gate/q4_gate_report.json \
  --output-dir /runs/selector
```

В `reference_training_outputs/` также находится полный derivative-only эталон:
30 OOF Parquet с manifests, десять OOF receipts, development и полный CPU OOF,
переносимые OOF/q4-отчёты, два full-fit receipt и готовый CatBoost-selector.
Эти файлы содержат ID, метки, хеши, scores и predictions, но не содержат
исходных названий, описаний или изображений.

## Полные обучения и сборка пакета

После результата GO запустите `train_lora --full-fit` один раз для каждой
ветки. Заполните `submission_manifest.example.json` фактическими путями и
SHA-256, зафиксируйте manifest через `contracts.bind_self_hash` и вызовите
`build_bundle.py`. В пакет входят шесть checkpoints, CatBoost-selector,
конфигурация, environment lock и manifests. Строки organizer train в пакет не
копируются.

## Границы с закрытием при ошибке

- Ни один обучающий manifest не принимает данные внешней оценки.
- Текст и метки конечных точек R4/R7 берутся из organizer train, а не считаются
  доверенными из дополнительного JSONL.
- R5 никогда не входит в CE.
- После замороженного агрегированного порога 95% R5 использует все 200 пар с
  корректным направлением; 192 решения о пригодности и восемь флагов рецензента
  остаются явными доказательствами.
- Control и candidate привязывают одинаковые снимки CE/R4/R5/R7/config/base.
- Компоненты запросов отсутствуют в CE и каналах реальных пар каждой складки.
- Конечные точки q4 отсутствуют во всех повторно созданных дополнительных
  данных реальных пар.
- Selector нельзя обучить без пяти полных контрольных точек каждой ветки и
  результата GO на q4.
- Для обучения на GPU требуются новый каталог результата, неизменяемые хеши
  файлов базовой модели, квитанция одной H100 и декларация отключённой сети.
- Пакет фиксирует функциональную воспроизводимость; ядра CUDA не гарантируют
  побайтово идентичное повторное обучение на разных конфигурациях
  оборудования/драйверов.
