# Organizer ID annotations

This directory contains only derived ID annotations. It contains no organizer
names, descriptions, labels, images, or complete source rows. Endpoint text,
labels, folds, and components are resolved from the organizer-train Parquet at
manifest validation time.

## R4 definition and files

R4 (Round 4) is a set of 100 contrastive relations used in the pairwise loss of
both LoRA branches. `r4/r4_pairs.jsonl` stores only `pair_id`, `positive_id`,
`negative_id`, `boundary_code`, `mass_key`, `review_status`, and
`eligible_query_folds`. `r4/annotation_manifest.json` binds the file by byte
size and SHA-256.

## R7 definition and files

R7 (Round 7) is query-specific contrastive supervision used only by the
candidate branch. `r7/r7_relations.jsonl` contains 100 accepted relation
instances: 22 for q0, 33 for q1, 30 for q2, and 15 for q3. A mass key may occur
in more than one query fold; q4 and full-fit schedules deduplicate the inventory
to 62 mass keys. Each record stores only IDs and relation metadata.
`r7/annotation_manifest.json` binds the file by byte size and SHA-256.

Copy both JSONL files beside the organizer-train Parquet before running
`training_code.prepare_manifests`, as shown in the root reproduction guide.
