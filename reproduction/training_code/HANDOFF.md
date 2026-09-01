# Training quick reference

R4 (Round 4) is the 100-relation ID-only organizer-train contrastive set shared
by both LoRA branches. R5 (Round 5) is the 200-pair/400-card synthetic set
generated through OpenRouter and used only for pairwise loss. R7 (Round 7) is
100 query-specific ID-only organizer-train relation instances, deduplicated to
62 mass keys for q4/full-fit and used only by the candidate branch.

Only the raw organizer CE Parquet is supplied externally. The R4/R7 JSONL files
are included in `organizer_annotations/`; copy them into `REPRO_WORKDIR` beside
the Parquet before building manifests.

## Data

- CE: 5,502 organizer-train rows, 198 positives;
- R4: 100 real-real pairs for control and candidate;
- R5: 200 OpenRouter-generated synthetic pairs for control and candidate;
- R7: 100 query-specific instances / 62 q4-full-fit mass keys for candidate only.

R5 is used only in pairwise loss and is never added to CE.

## Parameters

- base model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- LoRA rank 16, alpha 32;
- one CE epoch;
- effective batch 32;
- R4/R5 weights: `0.02/0.02`;
- candidate R7 weight: `0.03`;
- checkpoints: 25%, 50% and 100%;
- CatBoost: depth 4, 800 iterations, learning rate 0.03.

## Execution order

1. Build self-hashed CE/R4/R5/R7 manifests.
2. Run `organizer_cpu_dry_run`.
3. Train control/candidate LoRA over five OOF folds.
4. Verify the q4 gate.
5. Fit the CatBoost selector on complete OOF features.
6. Full-fit both LoRA branches.
7. Package checkpoints, selector, environment lock and manifests with
   `build_bundle.py`.

`reference_training_outputs/` already contains the derivative-only 30-file OOF
inventory, ten receipts, development CPU OOF, and the ready selector. Raw
organizer text and images are absent.

CPU audit:

```bash
export REPRO_WORKDIR=/absolute/path/to/reproduction_workdir
PYTHONPATH=. python -m training_code.organizer_cpu_dry_run \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --r4-manifest "$REPRO_WORKDIR/r4_manifest.json" \
  --r5-manifest "$REPRO_WORKDIR/r5_manifest.json" \
  --r7-manifest "$REPRO_WORKDIR/r7_manifest.json" \
  --recipe-precommit "$REPRO_WORKDIR/recipe_precommit.json" \
  --output "$REPRO_WORKDIR/organizer_cpu_dry_run.json"
```

Expected status:
`PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.

Training stops on a SHA-256 mismatch, incomplete OOF inventory, failed q4 gate,
base-model drift or external-evaluation data.
