# E-CUP Quality — inference runtime

This directory contains the complete runtime for the two supported product
categories.

## Decision pipeline

- **БАД:** `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` runs in BF16 through vLLM
  `0.26.0+cu129`. Two fixed prompts produce first-token binary margins. Their
  sum is classified with threshold `0.6875`; a narrow explicit-`не БАД`
  literal veto is applied.
- **Легковоспламеняющиеся:** two CPU text experts, six frozen LoRA checkpoints,
  a CatBoost selector, deterministic rank quota and four sparse text gates
  produce the final decision.

Outputs preserve the input row order and use the required
`<комментарий>...<вердикт>...` format.

After the class labels are fixed, the same ordered vLLM session generates
evidence-bound comments for both categories. It produces two candidates,
checks repaired candidates, selects the stronger supported explanation and
falls back to a fixed comment when the evidence or 50–300 character contract
fails. The comment stage cannot change a class label.

## Run

```bash
python3 -u run.py -i TEST_CSV -o SOLUTION_CSV
```

Input columns:

- unique integer `id`;
- `name`;
- `description`;
- `category`, containing only `БАД` or `Легковоспламеняющиеся`.

The base model is loaded from
`${SHARED_MODELS_PATH:-/shared_models}/Qwen/Qwen3.5-4B`.

`--mock-inference` or `ECUP_MOCK_INFERENCE=1` checks file plumbing only and
must not be used for production predictions.

## Integrity

`member_manifest.json` binds every ZIP member by size and SHA-256.
`artifacts/production_manifest.json` binds all LoRA, CatBoost and base-model
artifacts. The runtime verifies model revision, classification-prompt hashes,
comment-policy hash, token IDs and threshold constants before inference.

The ZIP contains no organizer rows or labels.
