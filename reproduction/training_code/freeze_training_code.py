#!/usr/bin/env python3
"""Freeze a data-free, hash-bound snapshot of the V170 training source."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from .contracts import bind_self_hash, sha256_file, write_json_atomic


EXCLUDED_NAMES = frozenset({"__pycache__", ".DS_Store", "training_code_manifest.json"})
ALLOWED_SUFFIXES = frozenset({".py", ".json", ".lock", ".md"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def freeze(output_dir: Path) -> dict:
    source = Path(__file__).resolve().parent
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("V170 training-code snapshot requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    copied = []

    def copy_one(path: Path, relative: Path) -> None:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED_NAMES for part in path.parts):
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        relative = path.relative_to(source)
        copy_one(path, relative)
    if not copied or any(
        marker in item["path"].lower()
        for item in copied
        for marker in ("local_only", "prediction", "checkpoint", "adapter_model")
    ):
        raise RuntimeError("V170 training-code snapshot contains an invalid file roster")
    manifest = bind_self_hash(
        {
            "schema_version": "v170.training_code_snapshot.1",
            "status": "COMPLETE",
            "source_directory": "training_code",
            "files": copied,
            "contains_organizer_rows": False,
            "contains_supplementary_pair_rows": False,
            "contains_model_weights": False,
            "contains_predictions": False,
            "contains_credentials": False,
            "contains_external_eval_data": False,
        }
    )
    write_json_atomic(output / "training_code_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = freeze(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
