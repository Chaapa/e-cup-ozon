# Training-code verification

Date: 2026-08-31.

R4 is the shared 100-relation ID-only contrastive set; R5 is the shared
OpenRouter-generated 200-pair/400-card synthetic set; R7 contains 100
query-specific ID-only instances and 62 q4/full-fit mass keys for the candidate
branch. CE contains the 5,502 organizer-train classification rows.

Verified:

- Python module imports and compilation;
- CE/R4/R5/R7 manifests;
- OpenRouter R5 provenance;
- matched control/candidate schedules;
- q4 isolation;
- OOF inventory;
- the 71-feature CatBoost-selector contract;
- CPU-only audit without model, GPU, API or network use.

Command:

```bash
python3 -m pytest -q training_code/tests
```

Package verification result: `11 passed`.

The CPU audit command in `training_code/README.md` must return
`PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.
