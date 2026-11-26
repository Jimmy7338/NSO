#!/usr/bin/env python3
"""Stage the closed Phase A evidence, with native meshes retained on the host."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
FACTOR = ROOT / "factor"
RUNS = FACTOR / "execution/formal"
RESULTS = FACTOR / "results/formal"
PAPER = ROOT / "paper"
PREFIX = "experiments/offline_20261126_ras_regional_factorial"
ARC_OLD = "../offline_20261123_region_independent_floors"
SIMPLE_FILES = (
    "stage_phase_a.py", "factor/PROTOCOL.json", "factor/FORMAL_FREEZE.json", "factor/REUSE_GATE.json",
    "factor/SOURCE_AUDIT.json",
    "factor/factor_study.py", "factor/runner.py", "factor/evaluate_factor.py",
    "factor_audit/replay_rankings.py", "factor_audit/reuse_receipt.json",
    "factor_audit/summarize_factor.py", "factor_audit/README.md",
    "factor_audit/factor_interactions.pdf", "factor_audit/factor_interactions.png", "factor_audit/factor_interactions.svg",
    "paper/CAPTION.md", "paper/render_checkpoints.py", "paper/checkpoints_a96_receipt.json",
    "paper/checkpoints_a96.pdf", "paper/checkpoints_a96.png", "paper/checkpoints_a96.svg",
    "factor/execution/formal/phase_receipt.json",
)
NEEDED_METRICS = {"summary.json", "episodes.csv", "checkpoints.csv", "factor_effects.csv",
                  "coverage_curve.csv", "trajectory.csv", "first_divergence.csv"}
NATIVE_SUFFIXES = {".json", ".npz", ".log", ".csv"}
MAX_INLINE_SIZE = 256 * 1024


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen archive input changed: {path}")


def require(predicate, details):
    if not predicate:
        raise ValueError(details)


def matrix():
    protocol = read(FACTOR / "PROTOCOL.json")
    frozen = read(FACTOR / "FORMAL_FREEZE.json")
    gate = read(FACTOR / "REUSE_GATE.json")
    phase = read(RUNS / "phase_receipt.json")
    summary = read(RESULTS / "summary.json")
    freeze_sha = sha(FACTOR / "FORMAL_FREEZE.json")
    require(gate["actor_freeze_sha256"] == phase["freeze_sha256"] == summary["actor_freeze"] == freeze_sha,
            "Evaluation, actor freeze and reuse gate refer to different experiments")
    reuse_path = ROOT / "factor_audit/reuse_receipt.json"
    verify(reuse_path, gate["verified_archive_receipt_sha256"])
    reuse = read(reuse_path)
    verify(BASE / "FORMAL_FREEZE.json", summary["archived_actor_freeze"])
    layout_arms = set(product(protocol["layouts"], ("D1G0", "D0G1")))
    jobs = {(row["layout"], row["arm"]) for row in frozen["jobs"]}
    exit_rows = {(row["layout"], row["arm"]): row for row in phase["episodes"]}
    require(len(layout_arms) == len(jobs) == len(exit_rows) == len(phase["episodes"]) == 8 and
            layout_arms == jobs == exit_rows.keys() and summary["registered_pairs"] == 16,
            "The protocol and measurements must cover all eight new and eight reused arms")
    require(reuse["status"] == "verified" and len(reuse["runs"]) == summary["completed_reused"] == 8 and
            sum(row["exit_code"] == 0 for row in exit_rows.values()) == summary["completed_new"] and
            summary["completed_reused"] + summary["completed_new"] + summary["failed"] == 16,
            "The reported run counts do not match the native outcomes")
    require(set(summary["results_sha256"]) | {"summary.json"} >= NEEDED_METRICS,
            "A requested outcome CSV is missing from the common evaluator's summary")
    for name, digest in summary["results_sha256"].items():
        verify(RESULTS / name, digest)
    for row in exit_rows.values():
        attempt = Path(row["run_path"]).parent
        require(attempt.is_relative_to(RUNS) and
                attempt.parent == RUNS / row["layout"] / row["arm"],
                f"Worker exited outside the registered arm path: {attempt}")
        require(read(attempt / "exit.json") == row and
                row["started_unix_s"] >= gate["checked_before_start_unix_s"],
                f"Worker outcome or chronology mismatch: {attempt}")
        verify(attempt / "worker.log", row["worker_log_sha256"])
        if row["result_sha256"]:
            verify(Path(row["result_path"]), row["result_sha256"])
    chart = read(PAPER / "checkpoints_a96_receipt.json")
    for name, digest in chart["figures_sha256"].items():
        verify(PAPER / name, digest)
    require([record["layout"] for record in chart["layouts"]] == ["ind26_offset_ring", "ind26_double_loop"]
            and chart["action"] == 96,
            "The required offset-ring and double-loop A96 figure must be complete")
    return protocol, summary, sorted(layout_arms)


def copy(payload, path):
    require(path.is_file() and not path.is_symlink(), f"Missing regular report artifact: {path}")
    local = payload / path.relative_to(ROOT)
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, local)


def package(payload, name, selected, predicate):
    folder = payload / "evidence_bundles"
    folder.mkdir(exist_ok=True)
    archive_path = folder / name
    index = {}
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=8) as archive:
        for path in sorted(selected):
            require(path.is_file() and not path.is_symlink(), f"Invalid evidence source {path}")
            relative = path.relative_to(ROOT).as_posix()
            bundled = predicate(path)
            index[relative] = {"sha256": sha(path), "bytes": path.stat().st_size, "bundled": bundled}
            if bundled:
                archive.write(path, relative)
    with ZipFile(archive_path) as archive:
        for member in archive.namelist():
            with archive.open(member) as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == index[member]["sha256"],
                        f"Invalid bundle SHA for {member}")
    return {"archive": archive_path.relative_to(payload).as_posix(), "archive_sha256": sha(archive_path),
            "files": index}


def native_attempts(payload, layout_arms):
    records = []
    for layout, arm in layout_arms:
        root = RUNS / layout / arm
        attempts = sorted(path for path in root.glob("attempt_*") if path.is_dir())
        require(attempts, f"No recorded worker attempts for {layout}/{arm}")
        selected, exits = [], []
        for attempt in attempts:
            for name in ("launch.json", "exit.json", "worker.log"):
                require((attempt / name).is_file(), f"Missing execution record {attempt/name}")
            receipt = read(attempt / "exit.json")
            require(receipt["layout"] == layout and receipt["arm"] == arm,
                    f"Unexpected actor identity in {attempt}")
            verify(attempt / "worker.log", receipt["worker_log_sha256"])
            exits.append({"attempt": attempt.name, "exit_code": receipt["exit_code"],
                          "result_path": receipt["result_path"], "result_sha256": receipt["result_sha256"]})
            selected.extend(path for path in attempt.rglob("*") if path.is_file())
        packed = package(payload, f"{layout}_{arm}.zip", selected,
                         lambda path: path.suffix in NATIVE_SUFFIXES)
        records.append({"layout": layout, "arm": arm, "attempts": exits, **packed})
    return records


def scene_reference(protocol):
    scene_path = BASE / "scenes/actor_freeze_holdout.json"
    scene = read(scene_path)
    cores = []
    for item in scene["layouts"]:
        if item["layout"] in protocol["layouts"]:
            cores.append({key: item[key] for key in
                          ("layout", "seed", "anchor", "shape", "initial_packet_sha256",
                           "initial_owner_sha256", "actual_config")})
    require([item["layout"] for item in cores] == protocol["layouts"],
            "Archived scene identities do not match the factor protocol")
    return {"archived_scene_freeze_sha256": sha(scene_path),
            "archived_repository_path": f"{ARC_OLD}/scenes/actor_freeze_holdout.json",
            "archived_scene_assets": f"{ARC_OLD}/evidence_bundles",
            "layouts": cores}


def landing_readme(protocol, summary, reports):
    names = [path.relative_to(ROOT).as_posix() for path in reports]
    reports = " · ".join(f"[{Path(name).stem}]({name})" for name in names)
    old_ref = ARC_OLD
    return f"""# NSO regional scoring factorial study, Phase A

Four simulated floors contribute independent paired comparisons across four scoring combinations.
The original regional planner D0G0 and full regional planner D1G1 are archive reuses.
The two single-change arms D1G0 and D0G1 are registered as new closed-loop runs.
The study records {summary['completed_new']} new and {summary['completed_reused']} reused complete runs,
with {summary['failed']} failed terminal dispositions. All arms use the same local S policy,
initial observations, sensors, TSDF and 192-action budget including the return.

- [Protocol](factor/PROTOCOL.json) · [Actor freeze](factor/FORMAL_FREEZE.json) · [Reuse gate](factor/REUSE_GATE.json) · [Source audit](factor/SOURCE_AUDIT.json)
- [Archived score and target replay](factor_audit/reuse_receipt.json)
- {reports}
- [Episodes](factor/results/formal/episodes.csv) · [Checkpoints](factor/results/formal/checkpoints.csv) · [Factor effects](factor/results/formal/factor_effects.csv)
- [A96 figure PDF](paper/checkpoints_a96.pdf) · [PNG](paper/checkpoints_a96.png) · [Editable SVG](paper/checkpoints_a96.svg) · [Caption and source](paper/CAPTION.md)
- [Factor interactions PDF](factor_audit/factor_interactions.pdf) · [PNG](factor_audit/factor_interactions.png) · [Editable SVG](factor_audit/factor_interactions.svg)
- [Native trace and figure evidence index](EVIDENCE_INDEX.json) · [Reused scene identity](SCENE_REUSE.json)

The archive reference commit is `{protocol['archive_head']}`.
The published actor source baseline is `{protocol['execution_source_base']}`.
The [original four-floor archive]({old_ref}/README.md) contains the scene material
and the eight reused trajectories. This package includes the raw outcomes and logs from
all new attempts, with PLY snapshots identified by their authoritative local paths and SHA-256.
Large derived trace tables are stored in the evidence bundle when indicated by the index.

## Reproduction

Use the original Jetson workspace with the frozen four-floor and inherited
source directories available as siblings of this study. Preserve existing
outputs before running into a fresh study directory.

```sh
nso_python=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$nso_python" factor_audit/replay_rankings.py | cmp - factor_audit/reuse_receipt.json
"$nso_python" factor/factor_study.py seal
"$nso_python" factor/factor_study.py run --workers 6
"$nso_python" factor/evaluate_factor.py
"$nso_python" factor_audit/summarize_factor.py
"$nso_python" paper/render_checkpoints.py
```
"""


def manifest(payload):
    rows = []
    for path in sorted(path for path in payload.rglob("*") if path.is_file()):
        data = path.read_bytes()
        rows.append({"stage_path": str(path), "repo_path": f"{PREFIX}/{path.relative_to(payload).as_posix()}",
                     "git_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
                     "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    return rows


def stage(destination, reports):
    protocol, summary, layout_arms = matrix()
    require(reports and all(path.is_relative_to(ROOT / "factor_audit") and path.is_file()
                            for path in reports), "Finalized Phase A report file(s) must exist under factor_audit")
    selected = [*(ROOT / name for name in SIMPLE_FILES), *reports]
    require(all(path.is_file() for path in selected), "One or more finalized Phase A figure or method files are missing")
    payload = destination / "payload"
    payload.mkdir(parents=True, exist_ok=False)
    for path in selected:
        copy(payload, path)
    metrics = [RESULTS / name for name in ("summary.json", *summary["results_sha256"])]
    large_metrics = []
    for path in metrics:
        if path.name in {"trajectory.csv", "measurement_bindings.csv"} and path.stat().st_size > MAX_INLINE_SIZE:
            large_metrics.append(path)
        else:
            copy(payload, path)
    extra = [path for path in sorted(FACTOR.glob("*.log")) if path.is_file()]
    extra.extend(path for path in sorted((FACTOR / "results").glob("evaluation_attempt_*/*"))
                 if path.is_file() and path.suffix in NATIVE_SUFFIXES)
    for path in extra:
        copy(payload, path)
    natives = native_attempts(payload, layout_arms)
    refs = scene_reference(protocol)
    derived = package(payload, "derived_trajectory_tables.zip", large_metrics, lambda _: True) if large_metrics else None
    index = {"native_attempts": natives, "derived_trace_bundle": derived,
             "original_run_archive": old_archive(protocol),
             "measurement_summary_sha256": sha(RESULTS / "summary.json"),
             "figure_receipt_sha256": sha(PAPER / "checkpoints_a96_receipt.json"),
             "retention": "Bundled records are stored under relative paths within ZIPs. Native PLY snapshots are hashed and remain in the original Jetson execution paths."}
    (payload / "EVIDENCE_INDEX.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    (payload / "SCENE_REUSE.json").write_text(json.dumps(refs, indent=2, ensure_ascii=False) + "\n")
    readme = landing_readme(protocol, summary, reports)
    (payload / "README.md").write_text(readme)
    (destination / "README.md").write_text(readme)
    plan = manifest(payload)
    (destination / "upload.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(f"Staged {len(plan)} items totaling {sum(row['size'] for row in plan):,} bytes", flush=True)


def old_archive(protocol):
    return {"commit": protocol["archive_head"], "directory": ARC_OLD,
            "reused_arms": {"D0G0": "region_S", "D1G1": "combined_S"}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ROOT / "publication_a")
    cli.add_argument("--report", type=Path, action="append")
    args = cli.parse_args()
    reports = args.report or [ROOT / "factor_audit/A_FACTOR_RESULTS.md"]
    stage(args.output.resolve(), [path.resolve() for path in reports])
