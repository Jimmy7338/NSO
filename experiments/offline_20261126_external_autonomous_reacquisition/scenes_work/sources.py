"""Geometry-only admission registry fixed before inspecting policy outcomes."""

from pathlib import Path
import json


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
ORIGINAL = ANALYSES / "ras_20261126/heterogeneous"
OLD_REGISTRATION = ORIGINAL / "scenes/STARTS.json"
ADMISSION = json.loads((HERE / "INHERITED_ADMISSION.json").read_text())
CANDIDATES = HERE / "SCENE_SOURCE_ORDER.json"
REGISTRY = json.loads(CANDIDATES.read_text())


def source_faces(details):
    row = {**details, "path": (HERE / details.get("normalized_file", details["file"])).resolve()}
    del row["file"]
    return row


ADMISSION_ORDER = [item["id"] for item in REGISTRY["sources"]]
SOURCES = {item["id"]: source_faces(item) for item in sorted(REGISTRY["sources"], key=lambda x: x["seed_index"])}
