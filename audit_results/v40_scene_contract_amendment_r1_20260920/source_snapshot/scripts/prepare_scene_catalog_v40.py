#!/usr/bin/env python3
"""Prepare seed reservations and six development skeletons. No scene engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nso.scene_contract_v40 import (CATALOG_VERSION, FAMILIES, canonical_json_bytes,
    content_sha256, load_json_strict, make_development_spec, pending_asset,
    public_planner_spec, seed_commitment, validate_catalog, validate_protocol,
    validate_scene_spec)

DEFAULT_PROTOCOL = ROOT / "configs/virtual3d/v40_scene_protocol_20260920.json"
DEFAULT_OUTPUT = ROOT / "audit_results/v40_scene_contract_20260920"


def build_bundle(protocol, seed_factory=None):
    """In-memory preparation. Tests may inject synthetic seed material."""
    validate_protocol(protocol)
    seed_factory = seed_factory or (lambda: secrets.token_hex(32))
    entries, development, escrow, seen_seeds = [], {}, [], set()
    for family in FAMILIES:
        for split, count, prefix in (("development", 1, "DEV"),
                (protocol["families"][family]["test_split"], 6, "TEST")):
            for index in range(count):
                parent = f"{prefix}_{family}_{index:02d}"
                if split == "development":
                    material = hashlib.sha256(f"{protocol['protocol_id']}:{parent}:development".encode()).hexdigest()
                else:
                    material = seed_factory()
                    if material in seen_seeds:
                        raise ValueError("test seed factory repeated seed material")
                    seen_seeds.add(material)
                commitment = seed_commitment(protocol["protocol_id"], parent, split, material)
                lineage = {"status": "reserved_not_materialized", "layout": f"{parent}.layout",
                           "prototypes": [f"{parent}.prototype_pool"], "meshes": [f"{parent}.mesh_pool"],
                           "derived_from_parent": None}
                entry = {"parent_id": parent, "family": family, "split": split, "lineage": lineage,
                         "seed_commitment": commitment, "assets": {key: pending_asset()
                            for key in ("layout", "prototype_bundle", "mesh_bundle")}}
                entries.append(entry)
                if split == "development":
                    development[parent] = make_development_spec(protocol, entry, material)
                else:
                    escrow.append({"parent_id": parent, "split": split, "seed_hex": material,
                                   "commitment": commitment})
    catalog = {"schema_version": CATALOG_VERSION, "protocol_id": protocol["protocol_id"],
               "protocol_sha256": content_sha256(protocol), "status": "parent_and_seed_reservations_only",
               "entries": entries, "geometry_generation_executed": False,
               "policy_or_metric_execution_count": 0}
    validate_catalog(catalog, protocol, scene_specs=list(development.values()))
    return catalog, development, {"schema_version": "v40.private_seed_escrow.v1",
        "protocol_id": protocol["protocol_id"], "seed_bits": 256, "records": escrow}


def write_bundle(protocol_path, output_dir):
    protocol_path, output_dir = Path(protocol_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to replace existing seed commitments or escrow; use --validate-only")
    protocol = load_json_strict(protocol_path)
    catalog, development, escrow = build_bundle(protocol)
    payloads = {".gitignore": b"private/\n", "catalog.json": canonical_json_bytes(catalog),
                "protocol.json": canonical_json_bytes(protocol),
                "private/test_seed_escrow.json": canonical_json_bytes(escrow)}
    for parent, spec in development.items():
        payloads[f"development/{parent}.json"] = canonical_json_bytes(spec)
        payloads[f"public_development/{parent}.json"] = canonical_json_bytes(public_planner_spec(spec, protocol=protocol))
    sources = [ROOT / "nso/scene_contract_v40.py", Path(__file__).resolve(), protocol_path,
               ROOT / "tests/test_scene_contract_v40.py",
               ROOT / "docs/research/V40_SCENE_SPLIT_SPEC_20260920.md"]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    manifest = {"schema_version": "v40.scene_preparation_manifest.v1", "protocol_id": protocol["protocol_id"],
        "counts": {"development_parents": 6, "main_test_parents": 24, "boundary_test_parents": 12,
                   "development_skeletons": 6, "test_seed_commitments": 36,
                   "worlds_created": 0, "observations_created": 0, "policy_runs": 0, "metric_runs": 0},
        "status": "reservation_only_not_formal_test_ready", "source_sha256": source_hashes,
        "file_sha256": hashes, "escrow_relative_path": "private/test_seed_escrow.json",
        "escrow_mode": "0600", "private_directory_mode": "0700",
        "test_seed_values_printed": False, "third_party_or_os_blinding_claimed": False,
        "geometry_assets_frozen": False,
        "notes": ["Asset hashes are null until assets exist and are independently audited.",
                  "Seed commitments bind identity and seed material, not a future generator or mesh.",
                  "Private seed material is not read by public validation."]}
    payloads["manifest.json"] = canonical_json_bytes(manifest)
    size = sum(map(len, payloads.values()))
    if size > protocol["max_output_bytes"]:
        raise ValueError("preparation exceeds protocol output limit")
    output_dir.mkdir(parents=True, mode=0o755)
    try:
        for name, data in payloads.items():
            path = output_dir / name
            private = name.startswith("private/")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
            if private:
                os.chmod(path.parent, 0o700)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        # Keep partial commitments/escrow for explicit recovery; never regenerate silently.
        raise
    return {"output_dir": str(output_dir), "bytes_written": size,
            "files_written": len(payloads), "counts": manifest["counts"],
            "formal_test_ready": False, "test_seed_values_printed": False}


def validate_public_bundle(output_dir):
    """No private seed file is opened, including for hashing."""
    output_dir = Path(output_dir)
    protocol = load_json_strict(output_dir / "protocol.json")
    catalog = load_json_strict(output_dir / "catalog.json")
    validate_catalog(catalog, protocol)
    manifest = load_json_strict(output_dir / "manifest.json")
    parents = {entry["parent_id"]: entry for entry in catalog["entries"] if entry["split"] == "development"}
    expected_names = {".gitignore", "catalog.json", "protocol.json", "private/test_seed_escrow.json"}
    expected_names.update(f"{folder}/{parent}.json" for parent in parents for folder in ("development", "public_development"))
    if set(manifest["file_sha256"]) != expected_names:
        raise ValueError("manifest file allowlist differs from the six development reservations")
    verified = 0
    for name, expected in manifest["file_sha256"].items():
        if name.startswith("private/"):
            continue
        path = (output_dir / name).resolve()
        if not path.is_relative_to(output_dir.resolve()):
            raise ValueError("manifest file escaped output directory")
        if path.is_relative_to((output_dir / "private").resolve()):
            raise ValueError("public manifest path resolves to private seed material")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"public file hash mismatch: {name}")
        verified += 1
    if {path.stem for path in (output_dir / "development").glob("*.json")} != set(parents):
        raise ValueError("development spec count or identities differ from catalog")
    scene_specs = []
    for spec_path in sorted((output_dir / "development").glob("*.json")):
        spec = load_json_strict(spec_path)
        validate_scene_spec(spec, expected_split="development", protocol=protocol)
        scene_specs.append(spec)
        entry = parents[spec_path.stem]
        if (spec["parent_id"] != entry["parent_id"] or spec["family"] != entry["family"]
                or spec["lineage"] != entry["lineage"] or spec["seed_record"]["commitment"] != entry["seed_commitment"]):
            raise ValueError("development spec identity/lineage/commitment mismatch")
        saved = load_json_strict(output_dir / "public_development" / spec_path.name)
        if public_planner_spec(spec, protocol=protocol) != saved:
            raise ValueError("public extraction mismatch")
    validate_catalog(catalog, protocol, scene_specs=scene_specs)
    private = output_dir / "private/test_seed_escrow.json"
    if private.is_symlink() or private.parent.is_symlink():
        raise ValueError("private escrow and directory cannot be symbolic links")
    if private.stat().st_mode & 0o777 != 0o600 or private.parent.stat().st_mode & 0o777 != 0o700:
        raise ValueError("private seed escrow permissions must be 0600/0700")
    return {"public_files_verified": verified, "protocol_bound_scene_specs": len(scene_specs), "private_seed_file_read": False,
            "formal_test_ready": False, "worlds_created": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate_public_bundle(args.output_dir) if args.validate_only else write_bundle(args.protocol, args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
