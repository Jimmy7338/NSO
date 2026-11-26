"""Build a local GitHub upload payload from the scan-support continuity study."""

import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PUBLICATION = Path(__file__).resolve().parent
ROOT = PUBLICATION.parent
DEST = "experiments/offline_20261126_external_support_continuity"
EXECUTION = Path("execution/formal")
RESULTS = Path("results/formal")
RECEIPT = EXECUTION / "phase_receipt.json"
REQUIRED = (
    Path("VALIDATION.json"),
    Path("FORMAL_PROTOCOL.json"),
    Path("FORMAL_FREEZE.json"),
    Path("RERUN_COLLISION_AUDIT.json"),
    Path("RESULTS.md"),
    Path("validation/v1_controls.json"),
    Path("validation/v2_controls.json"),
    Path("validation/validate_continuity_controls.py"),
    Path("plateau/README.md"),
    Path("plateau/check_plateau.py"),
    Path("plateau/hm3d_00770_s1_plateau.json"),
    Path("topology/plateau.py"),
    Path("topology/hm3d_start1_plateau.json"),
    RECEIPT,
    RESULTS / "summary.json",
    RESULTS / "measurement_audit.json",
    RESULTS / "episodes.csv",
    RESULTS / "checkpoints.csv",
    RESULTS / "original_mask_matched_paid.csv",
    RESULTS / "original_mask_matched_paid_receipt.json",
)
NATIVE_FILES = {
    "attempt.json", "audit.json", "checkpoint.json", "exit.json", "failure.json",
    "frontier_events.json", "global_choices.json", "guard_events.json", "heartbeat.json",
    "initial_raw_rgbd_scan_sha256.json", "launch.json", "local_choices.json", "manifest.json",
    "regional_choices.json", "regional_guards.json", "runtime_summary.json", "steps.json",
    "worker.log",
}
TRACE_FILES = {"trajectory.csv", "decisions.csv", "room_crossings.csv", "guard_events.csv"}
TEXT_EXTENSIONS = {".md", ".py", ".txt"}
CONTENT_EXTENSIONS = {".json", ".csv", ".md", ".txt", ".sha256"}
UPLOAD_PATH = re.compile(r"[/A-Za-z0-9._-]+")
EXCLUDED_NAME = re.compile(r"(?:^|_)(?:textures?|mesh|navmesh|pointcloud|frames?|assets?|geometry|truth|reference)(?:_|\.|$)", re.IGNORECASE)


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_file(root, relative):
    path = root / relative
    return path.is_file() and path.resolve() == path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_completion(root):
    absent = [str(file) for file in REQUIRED if not source_file(root, file)]
    if absent:
        raise ValueError("Required study records are pending: " + ", ".join(absent))

    validation = read_json(root / "VALIDATION.json")
    if validation["status"] != "verified" or validation["formal_planner_invocations_before_freeze"] != 0:
        raise ValueError("Geometry validation must predate all new formal planner invocations")
    controls = read_json(root / "validation/v2_controls.json")
    if (not controls["all_required_pass"] or controls["total"] != 21
            or controls["adapter_sha256"] != sha256(root / "repair_geometry.py")):
        raise ValueError("Final adapter must match the canonical 21-control receipt")

    protocol = read_json(root / "FORMAL_PROTOCOL.json")
    starts = [start["layout"] for start in protocol["starts"]]
    arms = protocol["arms"]
    expected = [(layout, arm) for layout in starts for arm in arms]
    if len(set(starts)) != 4 or len(set(arms)) != 3 or len(expected) != 12 or protocol["budget"] != 192:
        raise ValueError("Expected the preregistered four starts, three arms and 192-action budget")

    receipt = read_json(root / RECEIPT)
    completed = [(episode["layout"], episode["arm"]) for episode in receipt["episodes"]]
    if sorted(completed) != sorted(expected):
        raise ValueError("The receipt must account for every formal job, including terminal failures")
    frozen_sha = sha256(root / "FORMAL_FREEZE.json")
    if receipt["freeze_sha256"] != frozen_sha:
        raise ValueError("Formal execution receipt does not match the frozen study")

    summary = read_json(root / RESULTS / "summary.json")
    for key, digest in (
        ("run_freeze_sha256", frozen_sha),
        ("protocol_sha256", sha256(root / "FORMAL_PROTOCOL.json")),
        ("phase_receipt_sha256", sha256(root / RECEIPT)),
    ):
        if summary.get(key) != digest:
            raise ValueError(f"Evaluation summary has a stale {key}")
    sensitivity_path = root / RESULTS / "original_mask_matched_paid.csv"
    sensitivity_receipt = read_json(sensitivity_path.with_name("original_mask_matched_paid_receipt.json"))
    if (sensitivity_receipt["csv_sha256"] != sha256(sensitivity_path)
            or sensitivity_receipt["repaired_measurement_audit_sha256"] != sha256(root / RESULTS / "measurement_audit.json")):
        raise ValueError("The common-mask comparison does not match its receipt")


def select_files(root):
    regular = {*REQUIRED, Path("scenes/scene.py")}
    for pattern in ("*_start.json", "*_cache.json"):
        regular.update(p.relative_to(root) for p in (root / "validation/geometry").glob(pattern))
    native = set()
    for file in root.iterdir():
        if file.suffix in TEXT_EXTENSIONS or file.name.endswith("_AUDIT.json"):
            regular.add(file.relative_to(root))
    for file in (root / RESULTS).glob("*"):
        if file.suffix in CONTENT_EXTENSIONS and not EXCLUDED_NAME.search(file.name):
            collection = native if file.name in TRACE_FILES else regular
            collection.add(file.relative_to(root))
    for file in (root / EXECUTION).rglob("*"):
        relative = file.relative_to(root)
        tail = relative.parts[len(EXECUTION.parts):]
        if len(tail) >= 4 and re.fullmatch(r"attempt_\d+", tail[2]) and file.name in NATIVE_FILES:
            native.add(relative)

    return (
        sorted(file for file in regular if source_file(root, file)),
        sorted(file for file in native if source_file(root, file)),
    )


def check_upload_path(path):
    if not UPLOAD_PATH.fullmatch(path) or ".." in Path(path).parts:
        raise ValueError(f"Path is incompatible with the existing GitHub uploader: {path}")


def write_native_archive(root, payload, native):
    archive_path = payload / "native_evidence.zip"
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
        for relative in native:
            member_name = relative.as_posix()
            check_upload_path(member_name)
            member = ZipInfo(member_name)
            member.external_attr = 0o100644 << 16
            archive.writestr(member, (root / relative).read_bytes(), ZIP_DEFLATED, compresslevel=6)
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Native evidence archive failed CRC verification")


def manifest_entries(payload, output):
    entries = []
    for path in sorted(item for item in payload.rglob("*") if item.is_file()):
        relative = path.relative_to(payload)
        repo_path = f"{DEST}/{relative.as_posix()}"
        stage_path = str(output / "payload" / relative)
        check_upload_path(repo_path)
        check_upload_path(stage_path)
        data = path.read_bytes()
        blob = f"blob {len(data)}\0".encode() + data
        entries.append({
            "repo_path": repo_path, "stage_path": stage_path, "size": len(data),
            "git_sha1": hashlib.sha1(blob).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(),
        })
    return entries


def stage(root=ROOT, output=PUBLICATION):
    root, output = root.resolve(), output.resolve()
    check_completion(root)
    if (output / "payload").exists() or (output / "upload.json").exists():
        raise FileExistsError(f"Existing staging output in {output}")

    output.mkdir(parents=True, exist_ok=True)
    regular, native = select_files(root)
    with TemporaryDirectory(prefix=".stage-", dir=output) as temporary:
        payload = Path(temporary) / "payload"
        for relative in regular:
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        packaged_tools = payload / "publication"
        packaged_tools.mkdir()
        shutil.copyfile(Path(__file__), packaged_tools / "stage_publication.py")
        shutil.copyfile(PUBLICATION / "README.md", packaged_tools / "README.md")
        write_native_archive(root, payload, native)

        entries = manifest_entries(payload, output)
        manifest = Path(temporary) / "upload.json"
        write_json(manifest, entries)
        payload.rename(output / "payload")
        manifest.rename(output / "upload.json")
    print(f"{len(entries)} staged GitHub items, {len(native)} compact native files, "
          f"{sum(entry['size'] for entry in entries):,} total bytes")
    return entries


if __name__ == "__main__":
    stage()
