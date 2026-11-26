"""Stage a prospective registration or verified outcome bundle for the experiment branch."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEST = "experiments/offline_20261126_external_autonomous_reacquisition"
PHASES = ("development", "formal")
NATIVE_FILES = {
    "audit.json", "manifest.json", "failure.json", "steps.json", "regional_choices.json",
    "regional_guards.json", "global_choices.json", "local_choices.json", "checkpoint.json",
    "runtime_summary.json", "initial_raw_rgbd_scan_sha256.json", "launch.json", "exit.json",
    "worker.log", "observed_belief_trace.npz",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registration():
    frozen = read(ROOT / "FORMAL_FREEZE.json")
    assert frozen["phase"] == "formal" and frozen["budget"] == 192 and len(frozen["jobs"]) == 8
    layouts = {job["layout"] for job in frozen["jobs"]}
    assert len(layouts) == 4
    assert {tuple((job["layout"], job["arm"])) for job in frozen["jobs"]} == {
        (layout, arm) for layout in layouts for arm in ("A", "B")}
    for job in frozen["jobs"]:
        assert job["native_policy"] == "combined_S"
        assert job["active_ground_reacquisition"] == (job["arm"] == "B")
    files = {
        ROOT / item for item in (
            "runner.py", "study.py", "autonomous_observation.py", "reacquisition_adapter.py", "README.md",
            "DEVELOPMENT_PROTOCOL.json", "DEVELOPMENT_FREEZE.json", "FORMAL_PROTOCOL.json", "FORMAL_FREEZE.json",
            "PARAMETERS_FROZEN.json", "execution/development/phase_receipt.json",
            "validation/DEVELOPMENT_VALIDATION.json", "validation/SWEEP_GUARD_REGRESSION.json",
            "scenes_work/scene.py", "scenes_work/sources.py", "scenes_work/preflight.py",
            "scenes_work/SCENE_SOURCE_ORDER.json", "scenes_work/INHERITED_ADMISSION.json",
            "scenes_work/prepare.py", "scenes_work/register.py", "scenes_work/STARTS.json",
            "scenes_work/FORMAL_SCENE_MANIFEST.json",
            "scenes_work/actor_freeze_holdout.json", "scenes_work/evaluator_freeze_holdout.json",
            "publication/stage.py",
        )
    }
    scene = ROOT / "scenes_work"
    files |= {path for path in (scene / "source_preflights").glob("*.json")}
    for glob in ("*.md", "align_*.py", "normalize_*.py", "materialize.py", "reference_protocol.json",
                 "source_assets/*_alignment.json", "source_assets/*_source_receipt.json"):
        files |= {path for path in scene.glob(glob) if path.is_file()}
    for layout in layouts:
        for entry in (scene / "prepared" / layout / "initial_state.json",
                      scene / "evaluator_references" / layout / "RECEIPT.json"):
            files.add(entry)
    return frozen, files


def archive_native(destination):
    selected = sorted(path for phase in PHASES for path in (ROOT / "execution" / phase).rglob("*")
                      if path.name in NATIVE_FILES and path.is_file()
                      and any(part.startswith("attempt_") for part in path.relative_to(ROOT).parts))
    with ZipFile(destination, "x", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected:
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for phase in PHASES:
            for row in read(ROOT / "execution" / phase / "phase_receipt.json")["episodes"]:
                relative = Path(row["run_path"]).relative_to(ROOT)
                required = "steps.json" if row["exit_code"] == 0 else "failure.json"
                assert (relative / required).as_posix() in archive.namelist()


def result_selection(frozen):
    receipt_path = ROOT / "execution/formal/phase_receipt.json"
    receipt = read(receipt_path)
    assert receipt["freeze_sha256"] == sha(ROOT / "FORMAL_FREEZE.json")
    wanted = {(r["layout"], r["arm"]) for r in frozen["jobs"]}
    received = {(r["layout"], r["arm"]) for r in receipt["episodes"]}
    assert receipt["phase"] == "formal" and len(receipt["episodes"]) == len(received) == 8 and wanted == received
    for episode in receipt["episodes"]:
        source = Path(episode["run_path"]) / ("manifest.json" if episode["exit_code"] == 0 else "failure.json")
        assert episode["result_sha256"] == sha(source)
    include = {receipt_path, ROOT / "FINDINGS.md"}
    for glob in ("evaluate.py", "audit*.py", "figures/*.py", "figures/*.pdf", "figures/*.png", "figures/*.json",
                 "results/**/*.json", "results/**/*.csv", "results/*.md", "validation/*.json",
                 "validation/*.py", "validation/*.md", "publication/preregistration_receipt.json"):
        include |= set(ROOT.glob(glob))
    return include


def staged_manifest(folder, stage_name, *, index):
    manifest = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder)
        raw = path.read_bytes()
        repo = "experiments/RECENT_EXPERIMENTS.md" if index and relative.name == "RECENT_EXPERIMENTS.md" else f"{DEST}/{relative}"
        manifest.append(dict(repo_path=repo, stage_path=str(ROOT / "publication" / stage_name / "payload" / relative),
                             git_sha1=hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
                             sha256=hashlib.sha256(raw).hexdigest(), size=len(raw)))
    return manifest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("preregistered", "results"))
    cli.add_argument("--index", type=Path, help="Prior experiments index downloaded from the expected parent")
    args = cli.parse_args()
    formal, selected = registration()
    is_result = args.phase == "results"
    if is_result:
        selected |= result_selection(formal)
        if not args.index:
            raise ValueError("Final result stage requires the previously published experiments index")
    stage = ROOT / "publication" / args.phase
    if stage.exists():
        raise FileExistsError(stage)
    with TemporaryDirectory(dir=stage.parent) as area:
        temporary = Path(area) / args.phase
        payload = temporary / "payload"
        for src in sorted(selected):
            assert src.is_file(), src
            relative = src.relative_to(ROOT)
            dst = payload / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            copyfile(src, dst)
        if is_result:
            archive_native(payload / "native_evidence.zip")
            index = args.index.read_text()
            marker = "| 11 月 26 日 | HM3D 指定出口主动补测（独立验证＋12 轮）"
            assert index.count(marker) == 1
            path = Path(DEST).name
            row = (f"| 11 月 26 日 | 通用地面补测（四起点配对仿真） | [配对结果]({path}/FINDINGS.md)"
                   f" · [冻结协议]({path}/FORMAL_PROTOCOL.json) |\n")
            (payload / "RECENT_EXPERIMENTS.md").write_text(index.replace(marker, row + marker))
        manifest = staged_manifest(payload, args.phase, index=is_result)
        (temporary / "upload.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(stage)
    print(f"Staged {len(manifest)} files at {stage}", flush=True)


if __name__ == "__main__":
    main()
