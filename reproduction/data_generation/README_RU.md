# Генерация данных R5 через OpenRouter

**R5 (Round 5)** — корпус из 200 синтетических контрастных пар и 400 карточек
для pairwise loss обеих FLV-веток. Он не входит в CE.

Весь синтетический корпус R5 изначально создан через OpenRouter. Канонический
production pipeline находится в `openrouter_qwen122b/` и включает model pin,
provider pin, author/reviewer prompts, JSON schemas, frozen gates, бюджет и
детерминированные seeds.

## Конвейер

1. `qwen/qwen3.5-122b-a10b` через OpenRouter создаёт нейтральные style frames.
2. Детерминированный renderer добавляет causal slots и формирует обе стороны
   каждой пары.
3. Label-blind reviewer через OpenRouter проверяет направление и пригодность.
4. Структурные, overlap- и population-аудиты формируют итоговые квитанции.

Эталонные результаты production run находятся в
`../reproduction_outputs/openrouter_qwen122b_run/`:

- `authored_frames.jsonl` — OpenRouter-authored frames;
- `rendered_pairs.jsonl` — 200 контрастных пар;
- `training_cards.jsonl` — 400 синтетических карточек;
- `blind_reviews.jsonl` — независимая OpenRouter-проверка;
- `structural_audit.jsonl` — структурная проверка каждой пары;
- `metrics.json` — агрегированные метрики и model/provider provenance;
- `artifact_manifest.json` — размер и SHA-256 каждого артефакта.

Все `200/200` пар имеют корректное направление и прошли структурный аудит.
Слепая проверка признала пригодными `192/200`; восемь флагов сохранены для
контроля качества. Поскольку доля пригодных пар `96%` превышает порог `95%`, в
pairwise-расписание входят все 200 пар.

Исходный organizer train в пакет не входит. Для нового запуска передайте его
путь через `--source-train`; готовые проверенные R5-артефакты можно использовать
без повторного API-вызова.
