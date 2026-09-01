# Генерация R5 через OpenRouter

R5 (Round 5) — синтетический набор для pairwise-обучения: 200 контрастных пар
и 400 карточек, используемых обеими LoRA-ветками и не добавляемых в CE.

Директория содержит готовые prompts, schemas, runner, конфигурацию модели,
ограничение бюджета и проверки качества для генерации обучающего корпуса R5.

## Контракт

- OpenRouter model: `qwen/qwen3.5-122b-a10b`;
- Hugging Face model: `Qwen/Qwen3.5-122B-A10B`;
- revision: `dc4d348443bc740c68e2d77492492c11606384d5`;
- provider: `novita`;
- quantization: `bf16`;
- temperature: `0`;
- parallel workers: `4`;
- общий API cap: `$10`.

Pipeline генерирует 200 контрастных пар и 400 карточек. Label-blind reviewer
через OpenRouter проверяет направление и пригодность. Детерминированный
renderer, структурные проверки, overlap-аудит и SHA-256 выполняются локально.

## Prompts и schemas

- author prompt: `AUTHOR_PROMPT` в
  `quality_bench/r5_deterministic_slots_production.py`;
- reviewer prompt: `REVIEW_PROMPT` в
  `quality_bench/r5_openrouter_generation.py`;
- production config:
  `quality_bench/config/r5_deterministic_slots_full_production.json`;
- model/provider config:
  `quality_bench/config/openrouter_model.json`.

## Проверка без API-вызова

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m quality_bench.r5_deterministic_slots_production \
  --print-freeze
```

## Генерация данных

Исходный organizer train в пакет не входит. Передайте выданный организатором
FLV Parquet со столбцами `clean_name`, `clean_description` и `label`. До любого
API-вызова runner проверяет 5 502 строки и 198 положительных, затем использует
текстовые столбцы для проверки пересечения с исходными текстами.

```bash
export OPENROUTER_API_KEY='...'
PYTHONPATH=. python -m quality_bench.r5_deterministic_slots_production \
  --stage production \
  --source-train /absolute/path/to/organizer_train_flv.parquet \
  --output reproduced_runs/qwen122b-production
```

Ожидаемый результат:

- pairs/cards: `200/400`;
- structural pass: `200/200`;
- direction pass: `200/200`;
- usable review: `192/200`;
- reviewer flags: `8`;
- стоимость reference run: `$0.7999292`.

Проверенные артефакты находятся в
`../../reproduction_outputs/openrouter_qwen122b_run/`. Проверка исходников:

```bash
shasum -a 256 -c SOURCE_SHA256SUMS
```
