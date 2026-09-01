#!/usr/bin/env python3
"""Build the deterministic inference runtime ZIP from the prepared build tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SOURCE_DIR.parent
BUILD_TREE = RUNTIME_DIR / "build_tree"
OUTPUT = RUNTIME_DIR / "ecup_quality_runtime.zip"
BASE_ARCHIVE = RUNTIME_DIR / "ecup_quality_runtime.zip"
FIXED_TIME = (2026, 8, 30, 6, 0, 0)


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


def verify_member_name(name: str) -> None:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name:
        raise RuntimeError(f"Unsafe ZIP member path: {name}")


def zip_info(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    value = zipfile.ZipInfo(name, FIXED_TIME)
    value.create_system = 3
    mode = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
    value.external_attr = mode
    value.compress_type = zipfile.ZIP_DEFLATED
    return value


def compression_for(name: str) -> int:
    if name.endswith((".safetensors", ".so", ".cbm", ".joblib")):
        return zipfile.ZIP_STORED
    return zipfile.ZIP_DEFLATED


def load_members() -> tuple[dict[str, bytes], dict[str, str]]:
    if not BASE_ARCHIVE.is_file():
        raise FileNotFoundError(f"Bundled runtime ZIP is missing: {BASE_ARCHIVE}")
    members: dict[str, bytes] = {}
    sources: dict[str, str] = {}
    obsolete = {"member_manifest.json"}
    with zipfile.ZipFile(BASE_ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Bundled runtime ZIP CRC failure")
        for name in archive.namelist():
            if name in obsolete or name.endswith("/"):
                continue
            verify_member_name(name)
            members[name] = archive.read(name)
            sources[name] = f"bundled_runtime/{name}"
    for path in sorted(BUILD_TREE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(BUILD_TREE)
        if (
            relative.as_posix() == "member_manifest.json"
            or "__pycache__" in relative.parts
            or path.name == ".DS_Store"
            or path.suffix == ".pyc"
        ):
            continue
        name = relative.as_posix()
        verify_member_name(name)
        members[name] = path.read_bytes()
        sources[name] = f"build_tree/{name}"
    return members, sources


def verify_artifact_contract(members: dict[str, bytes]) -> None:
    manifest = json.loads(members["artifacts/production_manifest.json"])
    if manifest.get("base_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise RuntimeError("Base revision contract drift")
    expected = manifest.get("artifacts")
    if not isinstance(expected, dict):
        raise RuntimeError("Artifact contract is missing")
    for relative, contract in expected.items():
        name = f"artifacts/{relative}"
        value = members.get(name)
        observed = {
            "bytes": len(value) if value is not None else -1,
            "sha256": sha256_bytes(value) if value is not None else None,
        }
        if observed != contract:
            raise RuntimeError(f"Artifact drift: {name}")


def validate_members(members: dict[str, bytes]) -> None:
    required = {
        "run.py",
        "bad_vllm_worker.py",
        "README.md",
        "metadata.json",
        "artifacts/production_manifest.json",
        "src/reasoning_comments.py",
        "src/v116_model.py",
    }
    missing = required - set(members)
    if missing:
        raise RuntimeError(f"Required members missing: {sorted(missing)}")
    if not any(name.startswith("vendor/peft/") for name in members):
        raise RuntimeError("PEFT runtime missing")
    if "vendor_runtime_linux/catboost/_catboost.so" not in members:
        raise RuntimeError("CatBoost runtime missing")
    for name, value in members.items():
        if name.endswith(".py"):
            compile(value, name, "exec")
    metadata = json.loads(members["metadata.json"])
    if metadata != {
        "image": "odshack111/ecup26-quality-vllm:v38-cu129-r6-rocache",
        "entry_point": "python3 -u run.py",
    }:
        raise RuntimeError("Metadata contract drift")
    verify_artifact_contract(members)


def build(output: Path) -> dict[str, Any]:
    members, sources = load_members()
    validate_members(members)

    inventory = {
        name: {
            "bytes": len(value),
            "sha256": sha256_bytes(value),
            "source": sources[name],
        }
        for name, value in sorted(members.items())
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "package_id": "ecup_quality_inference_runtime",
        "inventory_excludes_self": True,
        "member_count_excluding_manifest": len(inventory),
        "members": inventory,
    }
    manifest["self_sha256"] = sha256_bytes(canonical(manifest))
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (BUILD_TREE / "member_manifest.json").write_bytes(manifest_bytes)
    members["member_manifest.json"] = manifest_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name in sorted(members):
            info = zip_info(name, executable=name == "run.py")
            info.compress_type = compression_for(name)
            archive.writestr(info, members[name])
    temporary.replace(output)

    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        names = archive.namelist()
        if names != sorted(names) or len(names) != len(set(names)):
            raise RuntimeError("ZIP member ordering or uniqueness drift")

    return {
        "status": "INFERENCE_ZIP_BUILT",
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "members": len(members),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
