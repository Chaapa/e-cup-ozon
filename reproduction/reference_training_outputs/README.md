# Derived training outputs

This directory contains reference artifacts derived from organizer-train rows.
It does not contain organizer names, descriptions, images, or complete source
rows.

## OOF definition and inventory

OOF means out-of-fold prediction: every row is scored by a trajectory whose CE
training split excludes that row's query fold. `oof/` contains both `control`
and `candidate`, five query folds per arm, and checkpoints at 25%, 50%, and
100%: 30 prediction Parquets in total. Each Parquet has exactly `id`,
`blind_uid`, `component_key`, `fold`, `label`, `score`, and `prediction`.
Self-hashed prediction manifests and ten fit receipts bind all files.

`development_cpu_oof.local_only.parquet` contains the 4,401 q0-q3 derived CPU
signals used with the LoRA OOF predictions. `cpu_oof.local_only.parquet` and
`cpu_oof_manifest.json` contain the complete five-fold 5,502-row CPU OOF view.
Text is represented only by hashes and numeric model outputs; source surfaces
are not present.

`audits/oof_audit.json` records the portable 30-file OOF audit.
`gates/q4_gate_report.json` records the q4 `GO` result computed from the bundled
OOF inventory. Both are self-hashed and contain no source text.

## Selector definition and files

The CatBoost selector combines 71 target-free inference features from the base
signals and the control/candidate LoRA trajectories. The ready selector is
`selector/selector_augmented_fold01234.cbm`; its SHA-256 matches the selector
inside `final_runtime/runtime/ecup_quality_runtime.zip`.

`artifact_manifest.json` binds the complete derivative-only inventory. Final
LoRA weights are already inside the inference ZIP and are not duplicated here.
`full_fit/control_fit_receipt.json` and
`full_fit/candidate_fit_receipt.json` bind their six checkpoint members by byte
size and SHA-256.
