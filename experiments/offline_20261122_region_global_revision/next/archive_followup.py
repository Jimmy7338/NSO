#!/usr/bin/env python3
"""Index the score repair and compact local diagnostic evidence."""

import hashlib
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PREVIOUS = PROJECT.parent/"region_structure_floor_20261122"
sys.path.insert(0,str(PREVIOUS))
from archive import zip_run
from regression import read, sha, write
from study import verify


REMOTE = "experiments/offline_20261122_region_global_revision/next"


def main():
    stage = PROJECT/"publication"/"followup"
    payload = stage/"payload"
    payload.mkdir(parents=True,exist_ok=False)
    runs = payload/"run_evidence"
    runs.mkdir()
    branch=HERE/"global_attribution"
    direct=HERE/"local_increment"
    tech=HERE/"local_technical"
    stage_files = [Path(__file__),HERE/"README.md",HERE/"access_audit",branch/"results",branch/"scenes/SCENES.md",
                   branch/"scenes/actor_freeze_holdout.json",branch/"scenes/evaluator_freeze_holdout.json",
                   branch/"scenes/formal_geometry_audit.json",branch/"scenes/reference_protocol.json",
                   *branch.glob("*.py"),*branch.glob("*.json"),*(branch/"scenes").glob("*.py"),
                   *direct.glob("*.py"),*direct.glob("*.json"),*direct.glob("*.md"),*direct.glob("*.csv"),
                   *(direct/"scenes").glob("*.py"),*(direct/"scenes").glob("*.json"),
                   direct/"evidence.zip",direct/"exploratory_a19/summary.json",
                   *tech.glob("*.py"),*tech.glob("*.md"),*tech.glob("*.json")]
    for name in ("results","measurement"):
        if (direct/name).is_dir():
            stage_files.append(direct/name)
    stage_files.extend(p for p in tech.glob("*/*")
                       if p.is_file() and p.suffix in (".json",".csv",".md") and p.stat().st_size<1_000_000)
    scene_records = {name:zip_run(branch/"scenes"/name,runs/f"formal_scenes_{name}.zip",include_mesh=True)
                     for name in ("prepared","precomputed_assets","evaluator_references")}
    direct_bundle=direct/"evidence.zip"
    with ZipFile(direct_bundle) as bundle:
        assert bundle.testzip() is None
    local_summary=read(direct/"results/summary.json")
    direct_record={"archive":str(direct_bundle.relative_to(HERE)),"archive_sha256":sha(direct_bundle),
                   "summary_sha256":sha(direct/"results/summary.json"),"independent_layouts":2}
    assert len(list((direct/"runs").rglob("manifest.json")))==6 and local_summary
    dispatched = []
    for phase in ("development","formal"):
        frozen=branch/f"{phase.upper()}_FREEZE.json"
        binding=read(frozen)
        verify(binding["actor_files"]|binding["evaluator_files"])
        summary=read(branch/"results"/phase/"summary.json")
        assert summary["run_freeze_sha256"]==sha(frozen) and summary["pending_trials"]==0
        receipts=branch/"execution"/phase
        stage_files.extend(receipts.glob("phase_receipt*.json"))
        for exit_path in sorted(receipts.glob("*/*/attempt_*/exit.json")):
            status=read(exit_path)
            product=Path(status["result_path"])
            if status["result_sha256"]:
                assert sha(product)==status["result_sha256"]
            trace=exit_path.parent
            key=f"{phase}_{status['layout']}_{status['arm']}_{trace.name}"
            dispatched.append(zip_run(trace,runs/f"{key}.zip")|{
                "layout":status["layout"],"arm":status["arm"],"exit_code":status["exit_code"],
            })
    for entry in sorted(set(p for source in stage_files for p in (source.rglob("*") if source.is_dir() else [source]))):
        if entry.is_file() and "__pycache__" not in entry.parts and entry.suffix not in (".ply",".png",".log"):
            dest=payload/entry.relative_to(HERE)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(entry,dest)
    write(payload/"EVIDENCE_INDEX.json",{
        "main_score_arms":dispatched,"fresh_building_fixtures":scene_records,
        "local_candidate_branches":direct_record,
        "preserved_original_technical_failures":True,
        "raw_native_meshes_held_locally":True,
    })
    plan=[]
    for p in sorted(p for p in payload.rglob("*") if p.is_file()):
        data=p.read_bytes()
        plan.append({"stage_path":str(p),"repo_path":REMOTE+"/"+p.relative_to(payload).as_posix(),
                     "size":len(data),"sha256":hashlib.sha256(data).hexdigest(),
                     "git_sha1":hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()})
    write(stage/"upload.json",plan)
    print(f"Staged {len(plan)} follow-up artifacts")


if __name__=="__main__":
    main()
