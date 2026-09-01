# OpenRouter generation of R5

R5 (Round 5) is the synthetic pairwise-training dataset: 200 contrastive pairs
and 400 product cards used by both LoRA branches and never added to CE.

This directory contains the prompts, schemas, runner, model configuration,
budget guard and quality checks for generating the R5 training corpus.

## Contract

- OpenRouter model: `qwen/qwen3.5-122b-a10b`;
- Hugging Face model: `Qwen/Qwen3.5-122B-A10B`;
- revision: `dc4d348443bc740c68e2d77492492c11606384d5`;
- provider: `novita`;
- quantization: `bf16`;
- temperature: `0`;
- parallel workers: `4`;
- total API cap: `$10`.

The pipeline generates 200 contrastive pairs and 400 cards. A label-blind
OpenRouter reviewer checks direction and usability. Deterministic rendering,
structural checks, overlap checks and artifact hashing run locally.

## Prompts and schemas

- author prompt: `AUTHOR_PROMPT` in
  `quality_bench/r5_deterministic_slots_production.py`;
- reviewer prompt: `REVIEW_PROMPT` in
  `quality_bench/r5_openrouter_generation.py`;
- production configuration:
  `quality_bench/config/r5_deterministic_slots_full_production.json`;
- model/provider configuration:
  `quality_bench/config/openrouter_model.json`.

## Validate without an API call

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m quality_bench.r5_deterministic_slots_production \
  --print-freeze
```

## Generate data

Organizer train is not included. Pass the organizer-provided FLV Parquet with
`clean_name`, `clean_description`, and `label` columns. The runner validates
5,502 rows and 198 positives before making any API call, then uses the text
columns for source-overlap checks.

```bash
export OPENROUTER_API_KEY='...'
PYTHONPATH=. python -m quality_bench.r5_deterministic_slots_production \
  --stage production \
  --source-train /absolute/path/to/organizer_train_flv.parquet \
  --output reproduced_runs/qwen122b-production
```

Expected results:

- pairs/cards: `200/400`;
- structural pass: `200/200`;
- direction pass: `200/200`;
- usable review: `192/200`;
- reviewer flags: `8`;
- production cost recorded in the reference run: `$0.7999292`.

Reference artifacts are stored in
`../../reproduction_outputs/openrouter_qwen122b_run/`. Verify source files with:

```bash
shasum -a 256 -c SOURCE_SHA256SUMS
```
