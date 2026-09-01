#!/usr/bin/env python3
"""Verify the inference ZIP and its non-self member manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def verify(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        if names != sorted(names) or len(names) != len(set(names)):
            raise RuntimeError("ZIP member ordering/uniqueness drift")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise RuntimeError(f"Unsafe member path: {name}")
        manifest = json.loads(archive.read("member_manifest.json"))
        expected_self = manifest.pop("self_sha256")
        if sha256_bytes(canonical(manifest)) != expected_self:
            raise RuntimeError("Member manifest self hash drift")
        expected = manifest["members"]
        actual_names = set(names) - {"member_manifest.json"}
        if actual_names != set(expected):
            raise RuntimeError("Member manifest coverage drift")
        for name, contract in expected.items():
            value = archive.read(name)
            if len(value) != contract["bytes"] or sha256_bytes(value) != contract["sha256"]:
                raise RuntimeError(f"Member content drift: {name}")
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("entry_point") != "python3 -u run.py":
            raise RuntimeError("Entry point drift")
        forbidden = [
            name
            for name in names
            if name.startswith("artifacts/bad/")
            or name in {"src/bad_adapter.py", "src/joint_model.py", "src/pipeline.py"}
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden BAD members leaked: {forbidden[:10]}")
    return {
        "status": "INFERENCE_ZIP_VERIFIED",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": len(names),
        "inventory_members": len(expected),
        "forbidden_members": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.zip_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
