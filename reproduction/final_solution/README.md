# Обучение и inference

The runtime has two independent binary branches: BAD and FLV. FLV training uses
organizer train together with OpenRouter-generated pairwise supervision.

R4 (Round 4) is the 100-relation ID-only organizer-train contrastive set. R5
(Round 5) is the 200-pair/400-card fully synthetic corpus generated through
OpenRouter. R7 (Round 7) is 100 query-specific ID-only organizer-train relation
instances, deduplicated to 62 mass keys for q4/full-fit and used only by the
candidate branch.

## FLV

1. Split organizer train into five component-disjoint folds.
2. Train rank-16 control and candidate LoRA branches on `Qwen/Qwen3.5-4B`.
3. Use CE together with 100 R4 real relations and 200 OpenRouter R5 synthetic pairs.
4. Give only the candidate branch 100 R7 relation instances grouped into 62 mass keys.
5. Save checkpoints at 25%, 50% and 100% of each trajectory.
6. Build OOF predictions and 71 target-free features for the CatBoost selector.

## BAD

BAD inference uses the pinned base `Qwen/Qwen3.5-4B` model through vLLM with two
fixed classification prompts. The two first-token margins are summed and
compared with threshold `0.6875`; no BAD adapter training data is distributed or
required by this package.

Exact schemas, commands and checks are documented in `training_code/README.md` and
`ORGANIZER_CHECKLIST.md`.
