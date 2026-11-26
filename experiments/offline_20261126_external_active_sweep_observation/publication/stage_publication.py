"""Stage either the independent active-observation gate or its qualified formal matrix."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PUBLICATION = Path(__file__).resolve().parent
ROOT = PUBLICATION.parent
DEST = "experiments/offline_20261126_external_active_sweep_observation"
GATE_RECEIPT = Path("execution/gate/phase_receipt.json")
GATE_VALIDATION = GATE_RECEIPT.with_name("GATE_VALIDATION.json")
FORMAL_RECEIPT = Path("execution/formal/phase_receipt.json")
RESULTS = Path("results/formal")
INDEPENDENT = Path("validation/GATE_AUDIT.json")
NATIVE = {"audit.json", "manifest.json", "failure.json", "steps.json", "regional_choices.json", "regional_guards.json",
          "global_choices.json", "local_choices.json", "frontier_events.json", "checkpoint.json", "runtime_summary.json",
          "initial_raw_rgbd_scan_sha256.json", "launch.json", "exit.json", "worker.log"}
FORMAL_TABLES = {"checkpoints.csv", "episodes.csv", "coverage_curve.csv", "external_start_pairs.csv",
                 "external_source_floor_means.csv", "external_source_summary.json", "summary.json", "measurement_audit.json",
                 "original_mask_matched_paid.csv", "original_mask_matched_paid_receipt.json"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, phase):
    gate = read(root / GATE_RECEIPT)
    if gate["phase"] != "gate" or len(gate["episodes"]) != 1 or gate["freeze_sha256"] != sha(root / "GATE_FREEZE.json"):
        raise ValueError("The payload requires one separately sealed gate")
    files = {GATE_RECEIPT, GATE_VALIDATION, INDEPENDENT, Path("GATE_PROTOCOL.json"), Path("GATE_FREEZE.json"),
             Path("active_observation.py"), Path("runner.py"), Path("study.py"), Path("evaluate.py"),
             Path("validation/audit_gate.py"), Path("validation/README.md"),
             Path("validation/SWEEP_GUARD_REGRESSION.json")}
    files |= {path.relative_to(root) for path in (root / "validation").glob("*TERMINATION*")
              if path.is_file() and path.suffix == ".json"}
    if (root / "validation/audit_gate_termination.py").is_file():
        files.add(Path("validation/audit_gate_termination.py"))
    if phase == "formal":
        if not read(root / GATE_VALIDATION)["eligible_for_formal"] or not read(root / INDEPENDENT)["gate_passed"]:
            raise ValueError("Formal payload requires both positive, separate gate records")
        phase_data = read(root / FORMAL_RECEIPT)
        frozen = read(root / "FORMAL_FREEZE.json")
        expected = {(item["layout"], item["arm"]) for item in frozen["jobs"]}
        observed = {(item["layout"], item["arm"]) for item in phase_data["episodes"]}
        metadata = read(root / RESULTS / "summary.json")
        if (phase_data["phase"] != "formal" or len(phase_data["episodes"]) != 12 or len(expected) != 12
                or observed != expected or phase_data["freeze_sha256"] != sha(root / "FORMAL_FREEZE.json")
                or metadata["run_freeze_sha256"] != phase_data["freeze_sha256"]
                or metadata["phase_receipt_sha256"] != sha(root / FORMAL_RECEIPT)):
            raise ValueError("Formal payload requires all twelve newly recorded workers and their matched evaluation")
        sensitivity = read(root / RESULTS / "original_mask_matched_paid_receipt.json")
        if (sensitivity["csv_sha256"] != sha(root / RESULTS / "original_mask_matched_paid.csv")
                or sensitivity["repaired_measurement_audit_sha256"] != sha(root / RESULTS / "measurement_audit.json")):
            raise ValueError("Common-mask sensitivity does not match the formal study")
        files |= {FORMAL_RECEIPT, Path("FORMAL_PROTOCOL.json"), Path("FORMAL_FREEZE.json"), Path("RESULTS.md"),
                  Path("FORMAL_OBSERVATION_AUDIT.json"), Path("audit_formal_observation.py")}
        if (root / "validation/REGIONAL_PAIR_AUDIT.json").is_file():
            files.add(Path("validation/REGIONAL_PAIR_AUDIT.json"))
        files |= {RESULTS / name for name in FORMAL_TABLES}
    if any(not (root / file).is_file() or (root / file).resolve() != root / file for file in files):
        raise ValueError("One or more required staged records are absent or indirect")
    return files


def stage(phase, root=ROOT, publication=PUBLICATION):
    root, publication = root.resolve(), publication.resolve()
    selected = validate(root, phase)
    selected |= {path.relative_to(root) for path in root.iterdir()
                 if path.is_file() and (path.suffix == ".md" and path.name != "RESULTS.md" or path.name.endswith("_AUDIT.json"))}
    phases = ("gate", "formal") if phase == "formal" else ("gate",)
    native = sorted(path.relative_to(root) for name in phases for path in (root / "execution" / name).rglob("*")
                    if path.name in NATIVE and path.is_file() and path.resolve() == path
                    and any(part.startswith("attempt_") for part in path.relative_to(root).parts))
    output = publication / phase
    if output.exists():
        raise FileExistsError(f"A local {phase} payload is already staged at {output}")
    with TemporaryDirectory(dir=publication) as temporary:
        area = Path(temporary) / phase
        payload = area / "payload"
        for relative in sorted(selected):
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        (payload / "publication").mkdir()
        shutil.copyfile(Path(__file__), payload / "publication/stage_publication.py")
        shutil.copyfile(PUBLICATION / "README.md", payload / "publication/README.md")
        archive_path = payload / "native_evidence.zip"
        with ZipFile(archive_path, "x") as archive:
            for name in native:
                entry = ZipInfo(name.as_posix())
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, (root / name).read_bytes(), ZIP_DEFLATED, compresslevel=6)
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Retained native evidence failed CRC validation")
        manifest = []
        for path in sorted(file for file in payload.rglob("*") if file.is_file()):
            relative = path.relative_to(payload)
            repo = f"{DEST}/{relative.as_posix()}"
            stage_path = str(output / "payload" / relative)
            if not all(re.fullmatch(r"[/A-Za-z0-9._-]+", value) for value in (repo, stage_path)):
                raise ValueError(f"Unsupported uploader path: {relative}")
            content = path.read_bytes()
            manifest.append({"repo_path": repo, "stage_path": stage_path, "size": len(content),
                             "git_sha1": hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest(),
                             "sha256": hashlib.sha256(content).hexdigest()})
        (area / "upload.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
        area.rename(output)
    print(f"Staged {len(manifest)} files and {len(native)} native receipts in {output}")


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--phase", choices=("gate", "formal"), required=True)
    stage(cli.parse_args().phase)
