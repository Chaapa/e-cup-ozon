# Organizer checklist

**Datasets.** R4 is a 100-relation ID-only organizer-train contrastive pool for both
branches. R5 is a 200-pair/400-card synthetic corpus generated through
OpenRouter for pairwise loss in both branches and never added to CE. R7 contains
100 query-specific ID-only organizer-train relation instances grouped into 62
q4/full-fit mass keys for the candidate branch.

1. Prepare organizer train using the CE schema in `training_code/README.md`.
2. Copy the bundled ID-only R4 and R7 annotations from `organizer_annotations/`.
3. Verify the seven files in
   `reproduction_outputs/openrouter_qwen122b_run/`.
4. Build self-hashed CE/R4/R5/R7 manifests.
5. Run `organizer_cpu_dry_run` and obtain
   `PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`.
6. Pin the base-model revision and environment lock.
7. Audit the bundled reference OOF outputs or rerun five-fold control/candidate
   OOF training and scoring.
8. Fit the selector and two full-data LoRA branches.
9. Build the bundle and verify every SHA-256 contract.
