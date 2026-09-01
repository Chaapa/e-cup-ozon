# Training and inference architecture

```text
BAD input ──> pinned Qwen3.5-4B + two fixed prompts ──> threshold

organizer FLV train ──┬──> control LoRA checkpoints ─┐
                      ├──> candidate LoRA checkpoints ─> OOF features ─> CatBoost selector
                      ├── 100 checked train relations ┤
                      ├── 200 OpenRouter synthetic pairs ┤
                      └── 100 relation instances / 62 mass keys ──> candidate only
```

Duplicate organizer components stay in one fold. Each OOF query group is
excluded from its corresponding training trajectory.

Pairwise supervision has a small weight relative to CE.
