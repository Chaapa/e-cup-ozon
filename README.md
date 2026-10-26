# E-CUP Quality

A reproducible machine-learning solution for the E-CUP product-card quality task. The repository contains LoRA training code, out-of-fold predictions, a CatBoost selector and packaged inference artifacts.

## Approach

- Fine-tune control and candidate LoRA branches on a pinned Qwen3.5-4B base model.
- Combine classification training with contrastive pairs and query-specific relations.
- Score out-of-fold checkpoints and train a CatBoost selector on 71 target-free features.
- Package the selected inference pipeline with integrity manifests and reproduction instructions.

The reproduction package includes ID-based organizer annotations and a recorded synthetic-data generation run. Original organizer training rows and images are **not included**; full retraining requires your authorized copy of that dataset.

## Start here

| Goal | Entry point |
|---|---|
| Run a packaged submission | [Release artifacts](release/README.md) |
| Understand the package | [Reproduction overview](reproduction/README_RU.md) |
| Reproduce training and validation | [Step-by-step guide](reproduction/REPRODUCE_FROM_SCRATCH_RU.md) |
| Inspect the training implementation | [Training code](reproduction/training_code) |
| Inspect saved validation outputs | [Reference outputs](reproduction/reference_training_outputs) |
| Check file integrity | [Release manifest](reproduction/RELEASE_MANIFEST.json) |

Detailed reproduction guides are in Russian.

## Get the artifacts

ZIP archives are tracked with Git LFS. Install Git LFS before cloning, then retrieve the objects:

```sh
git lfs install
git clone https://github.com/Chaapa/e-cup-ozon.git
cd e-cup-ozon
git lfs pull
```

A small text file beginning with `version https://git-lfs.github.com/spec/v1` is an LFS pointer, not the archive itself.

## Repository layout

```text
release/                  Packaged submission and reproduction archives
reproduction/
  data_generation/        Synthetic-pair generation code and configuration
  organizer_annotations/  ID-based contrastive and query-specific relations
  training_code/          LoRA training, OOF scoring and selector training
  reference_training_outputs/
  reproduction_outputs/   Recorded synthetic-data generation outputs
  final_runtime/          Packaged inference runtime
```

Review the dataset requirements and hardware/software setup in the reproduction guide before running training. Model weights and organizer data remain subject to their respective terms.
