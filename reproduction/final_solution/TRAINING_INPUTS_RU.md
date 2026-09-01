# Обучающие входы

R4 (Round 4) — ID-only контрастная разметка organizer train. R5 (Round 5) —
полностью синтетический корпус, изначально сгенерированный через OpenRouter.
R7 (Round 7) — query-specific ID-only разметка отношений organizer train только
для candidate-ветки.

| Канал | Объём | Использование |
|---|---:|---|
| FLV CE из organizer train | 5 502 строки, 198 положительных | control и candidate LoRA |
| R4 ID-only real-real relations | 100 пар | pairwise loss обеих веток |
| R5 OpenRouter synthetic pairs | 200 пар / 400 карточек | pairwise loss обеих веток |
| R7 ID-only real relations | 100 query-экземпляров / 62 mass key q4-full-fit | pairwise loss только candidate |

R5 не добавляется в CE. Структурную проверку и проверку направления прошли
200/200 пар; usable-review прошли 192/200, восемь reviewer flags сохранены.
