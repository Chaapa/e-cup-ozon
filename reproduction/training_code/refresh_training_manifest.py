#!/usr/bin/env python3
"""Refresh the self-hashed inventory of this training-code directory."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import bind_self_hash, sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "training_code_manifest.json"
ALLOWED_SUFFIXES = {".py", ".json", ".lock", ".md"}
EXCLUDED_NAMES = {"training_code_manifest.json", ".DS_Store"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}


def build() -> dict:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        relative = path.relative_to(ROOT)
        if set(relative.parts) & EXCLUDED_PARTS or path.suffix not in ALLOWED_SUFFIXES:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = bind_self_hash(
        {
            "schema_version": "v170.training_code_snapshot.2",
            "status": "COMPLETE",
            "source_directory": "training_code",
            "files": files,
            "contains_organizer_rows": False,
            "contains_supplementary_pair_rows": False,
            "contains_model_weights": False,
            "contains_predictions": False,
            "contains_credentials": False,
            "contains_external_eval_data": False,
        }
    )
    write_json_atomic(MANIFEST, manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
