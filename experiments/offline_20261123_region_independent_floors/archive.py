#!/usr/bin/env python3
"""Prepare source, measurements and compact native evidence for GitHub."""

import hashlib
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from study import ROOT, read, sha, verify, write


PREFIX="experiments/offline_20261123_region_independent_floors"


def package(folder, target, include_all=False):
    counted={}
    with ZipFile(target,"x",compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            name=path.relative_to(folder).as_posix()
            include=include_all or path.suffix!=".ply"
            counted[name]={"sha256":sha(path),"bytes":path.stat().st_size,"bundled":include}
            if include:
                archive.write(path,name)
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream,"sha256").hexdigest()==counted[name]["sha256"]
    return {"source_directory":folder.relative_to(ROOT).as_posix(),
            "archive":target.name,"archive_sha256":sha(target),"files":counted}


def main():
    checked=read(ROOT/"completion_audit.json")
    assert checked["completed_verified_new_runs"]==12
    assert checked["formal_summary_sha256"]==sha(ROOT/"results/formal/summary.json")
    verify({str(ROOT/"paper"/f):d for f,d in checked["paper_files_sha256"].items()})
    stage=ROOT/"publication"
    payload=stage/"payload"
    bundles=payload/"evidence_bundles"
    bundles.mkdir(parents=True,exist_ok=False)
    selected={
        *(p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".md",".py",".json"}),
        *ROOT.glob("known/*"),
        *ROOT.glob("evaluation/*.py"),
        *ROOT.glob("evaluation/*.md"),
        *ROOT.glob("evaluation/*.json"),
        *ROOT.glob("evaluation/*.log"),
        *ROOT.glob("results/evaluation_attempt_001/*"),
        *ROOT.glob("paper/*"),
        *ROOT.glob("results/formal/*"),
        ROOT/"formal.log",
        ROOT/"evaluation_recovery.log",
        *(ROOT/"execution/formal").glob("phase_receipt*.json"),
    }
    scene_root=ROOT/"scenes"
    for path in scene_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if set(path.relative_to(scene_root).parts)&{"prepared","evaluator_references","precomputed_assets"}:
            continue
        if path.suffix in {".json",".py",".md",".csv",".png",".pdf",".svg"}:
            selected.add(path)
    scene_packages=[]
    for kind in ("prepared","precomputed_assets"):
        scene_packages.append(package(scene_root/kind,bundles/f"scenes_{kind}.zip",include_all=True))
    for layout in read(ROOT/"PROTOCOL.json")["layouts"]:
        scene_packages.append(package(scene_root/"evaluator_references"/layout,
                             bundles/f"offline_reference_{layout}.zip",include_all=True))
    rejected=scene_root/"preregistration_v1"
    if rejected.is_dir():
        scene_packages.append(package(rejected,bundles/"preregistration_v1_geometry.zip",include_all=True))
    native=[]
    for path in sorted((ROOT/"execution/formal").glob("*/*/attempt_*/exit.json")):
        layout,arm,attempt,_=path.relative_to(ROOT/"execution/formal").parts
        native.append(package(path.parent,bundles/f"{layout}_{arm}_{attempt}.zip"))
    for path in sorted(selected):
        if not path.is_file():
            continue
        target=payload/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    write(payload/"EVIDENCE_INDEX.json",{"scene_inputs_and_reference":scene_packages,
         "native_attempts":native,"completion_audit_sha256":sha(ROOT/"completion_audit.json"),
         "native_mesh_retention":"Native PLY snapshots remain on the Jetson at the indexed execution paths."})
    plan=[]
    for target in sorted(path for path in payload.rglob("*") if path.is_file()):
        raw=target.read_bytes()
        plan.append({"stage_path":str(target),"repo_path":PREFIX+"/"+target.relative_to(payload).as_posix(),
                     "size":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
                     "git_sha1":hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()})
    write(stage/"upload.json",plan)
    print(f"Staged {len(plan)} records totaling {sum(p['size'] for p in plan):,} bytes.",flush=True)


if __name__=="__main__":
    main()
