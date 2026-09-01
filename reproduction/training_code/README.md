# FLV training: LoRA and CatBoost selector

**Round definitions.** R4 (Round 4) is a 100-relation ID-only organizer-train
contrastive pool shared by both branches. R5 (Round 5) is a 200-pair/400-card
fully synthetic corpus generated through OpenRouter and used by both branches
only in pairwise loss, never in CE. R7 (Round 7) is a query-specific ID-only
organizer-train relation pool used only by the candidate branch: 100 instances
for q0-q3 and 62 deduplicated mass keys for q4/full-fit.

This directory contains code for data preparation, two LoRA branches, OOF
scoring and CatBoost-selector training. The package includes R4/R7 ID
annotations, reference OOF predictions, the CatBoost selector, and final LoRA
weights inside the inference ZIP. Only raw organizer rows, the base-model
snapshot, API caches, and secrets are external.

## Training pipeline

1. Five component-disjoint organizer-train folds over 5,502 FLV rows (198
   positives): q0–q3 are development folds and q4 is the frozen forward gate.
   A q0–q3 OOF fit uses only the other development folds; q4 labels never enter
   development trajectories. The q4 trajectory trains on q0–q3 only.
2. A matched rank-16 Qwen3.5-4B control/candidate design. Both arms receive one
   text-only CE epoch, organizer-train-derived R4 real pairs at weight 0.02 and
   OpenRouter-generated R5 synthetic pairs at weight 0.02. Only the candidate receives
   R7 real pairs at weight 0.03; the control binds the same R7 manifest at zero
   mass.
3. Three checkpoints at 25%, 50%, and 100% of every trajectory.
4. Five-fold OOF scoring for both arms. Query labels never determine a LoRA
   checkpoint threshold: binary predictions use the corresponding outer-train
   prevalence rank.
5. A frozen q4 transfer gate. The recipe is frozen before any q4 score; q4 does
   not select R7 rows, weight, rank, prompt, or checkpoint recipe.
6. One final CatBoost selector trained only after q4 GO on all five OOF folds.
   It receives 71 target-free features: B1,
   character-SVC, three control checkpoints, three candidate checkpoints, and
   candidate-minus-control trajectory geometry. P/N and additional rule-based
   features do not enter this selector.
7. Separate full-data control and candidate fits, followed by a SHA-256-bound
   output bundle.

## Required inputs

All manifests are self-hashed and bind every data file by path, byte size and
SHA-256. CE/R4/R7 declare organizer-train provenance; R5 declares OpenRouter
model/provider provenance.

### CE Parquet

Required columns:

`id, blind_uid, label, fold, component_key, clean_name, clean_description,
normalized_name, normalized_description, image_count`

`component_key` must belong to exactly one fold. The trainer projects only
these columns and rejects external-evaluation columns.

### R4 JSONL

Each accepted record contains:

`pair_id, positive_id, negative_id, boundary_code, mass_key, review_status,
eligible_query_folds`

The loader resolves text and labels from the CE Parquet, requires direction
1→0, different components, same endpoint fold in frozen development folds
q0–q3, one relation per component pair, and exactly 100 accepted pairs. The
forward-gate fold q4 is forbidden in every supplementary real-pair endpoint.

### R5 production artifacts

R5 is a six-file, pairwise-only production contract:

- `rendered_pairs.jsonl`: positive/negative cards, decisive spans, changed slot,
  changed field, invariant fact frame, boundary, style frame and randomized
  display order;
- `blind_reviews.jsonl`: A/B labels, cited evidence, usability decision and the
  reviewer flag/reason for every pair;
- `structural_audit.jsonl`: per-pair structural checks;
- `training_cards.jsonl`: exactly two fully synthetic, organizer-labelled cards
  bound back to each rendered pair;
- `metrics.json`: aggregate direction, usability, overlap and population audit;
- `artifact_manifest.json`: SHA-256 and byte inventory for the other five files.

The regenerated production result is exactly 200/200 structurally passing
pairs and 200/200 correct blind directions. Blind review marks 192/200 usable
(96%, above the frozen 95% corpus gate) and retains eight reviewer flags. The
eight flagged pairs are deliberately **not filtered**: all 200 pairs enter the
R5 pairwise schedule, and their flag IDs/reasons remain hash-bound audit
evidence. The gate is aggregate; it is not a per-row acceptance rewrite after
generation.

The loader rejects organizer IDs/labels, exact organizer surface reuse, shared
13-token train spans, any failed structural/direction audit, more than five
pairs per namespaced boundary/style frame, fewer than 40 frames, roster drift
between the four JSONL files, or any artifact/metric/hash inconsistency. R5
never enters CE.

### R7 JSONL

Each record contains the R4 ID-bound fields plus `query_fold`. The bundled file
contains exactly 100 query-specific instances: q0=22, q1=33, q2=30, q3=15;
`eligible_query_folds` must equal `[query_fold]`. The same `mass_key` may appear
in different query folds but cannot repeat inside one query fold. q4 and
full-fit use one instance per mass key, yielding 62 relations. Endpoint text,
labels, folds, and components are resolved from the CE Parquet by ID. Full-fit
mass is capped independently by positive and negative component reuse.

Use `prepare_manifests.py` to create and immediately validate manifests. Input
files must live below the directory containing their manifest.

## Immutable preparation

Supply only the organizer CE Parquet. R4 and R7 are bundled as ID-only JSONL;
copy them into the same fresh local work directory because each manifest binds
files below its own directory. The loader resolves endpoint text, labels,
folds, and components from the Parquet. Set `R5_ARTIFACT_DIR` to the copied
OpenRouter R5 directory or to a newly generated OpenRouter production run.

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

# Mandatory CPU-only preflight: validates all rows, hashes, aggregate R5 gate,
# reviewer-flag preservation, fold closure, and matched schedules without
# importing a model or touching a GPU.
PYTHONPATH=. python -m training_code.organizer_cpu_dry_run \
  --config training_code/config.json \
  --ce-manifest "$REPRO_WORKDIR/ce_manifest.json" \
  --r4-manifest "$REPRO_WORKDIR/r4_manifest.json" \
  --r5-manifest "$REPRO_WORKDIR/r5_manifest.json" \
  --r7-manifest "$REPRO_WORKDIR/r7_manifest.json" \
  --recipe-precommit "$REPRO_WORKDIR/recipe_precommit.json" \
  --output "$REPRO_WORKDIR/organizer_cpu_dry_run.json"
```

The CPU report must say `PASS_DATA_AND_SCHEDULE_ONLY_NO_FIT`, list R5 counts
`200 / 192 / 8`, bind ten OOF trajectories plus two full-fit schedules, and
declare `gpu_loaded=false`, `model_loaded=false`, `fit_executed=false`, and
zero API/network calls. It is safe to run before any paid compute.

The base model must be downloaded at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; create its complete local file
contract with `prepare_manifests.py base-snapshot`. On the isolated H100,
create `execution-receipt` only after the immutable container and visible GPU
are verified. The fit command requires `network_disabled_during_fit=true`.

## OOF execution DAG

For each query fold 0…4, run the following twice, changing only `--arm` and
the output directory:

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

After all ten trajectories and scores exist, execute the q4 gate:

```bash
PYTHONPATH=. python -m training_code.evaluate_q4_gate \
  --config ... --ce-manifest ... --recipe-precommit ... \
  --oof-root /runs/oof --output-dir /runs/q4_gate
```

Only a `GO` report permits final selector fit:

```bash
PYTHONPATH=. python -m training_code.fit_selector \
  --phase fit --config ... --ce-manifest ... --oof-root /runs/oof \
  --gate-report /runs/q4_gate/q4_gate_report.json \
  --output-dir /runs/selector
```

The package also provides a derivative-only reference inventory in
`reference_training_outputs/`: 30 OOF prediction Parquets with manifests, ten
OOF fit receipts, development and full CPU OOF signals, portable OOF/q4 audit
reports, two full-fit receipts, and the ready CatBoost selector. These files
contain IDs, labels, hashes, scores, and predictions, but no source names,
descriptions, or images.

## Full fits and bundle

Run `train_lora --full-fit` once per arm after gate GO. Fill
`submission_manifest.example.json` with actual paths and SHA-256 values, seal it
with `contracts.bind_self_hash`, and call `build_bundle.py`. The bundle contains
six checkpoints, the CatBoost selector, configuration, environment lock and
manifests. Raw organizer rows are not copied into the bundle.

## Deliberate fail-closed boundaries

- No external-evaluation data is accepted by any training manifest.
- R4/R7 endpoint text and labels are resolved from organizer train, not trusted
  from supplementary JSONL.
- R5 never enters CE.
- R5 uses all 200 direction-correct pairs after the frozen 95% aggregate gate;
  the 192 usable decisions and eight reviewer flags remain explicit evidence.
- Control and candidate bind identical CE/R4/R5/R7/config/base snapshots.
- Query components are absent from every fold's CE and real-pair channels.
- q4 endpoints are absent from all regenerated supplementary real-pair data.
- A selector cannot fit without five complete checkpoints per arm and a q4 GO.
- GPU fitting requires a fresh output directory, immutable base-file hashes,
  one H100 receipt, and a network-disabled declaration.
- The package records functional reproducibility; CUDA kernels do not promise
  bit-identical retraining across different hardware/driver builds.
