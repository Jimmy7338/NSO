#!/usr/bin/env python3
"""Stage source, manifests and compressed trajectory evidence for the NSO branch."""

import argparse
import hashlib
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from study import ROOT, read, sha, verify, write


PREFIX = "experiments/offline_20261122_shelf_occlusion_pairs"


def package(directory, target, include_mesh=False, subdirectories=None):
    files = {}
    with ZipFile(target,"x",compression=ZIP_DEFLATED,compresslevel=6) as output:
        for source in sorted(p for p in directory.rglob("*") if p.is_file()):
            if "__pycache__" in source.parts:
                continue
            relative=source.relative_to(directory)
            if subdirectories is not None and relative.parts[0] not in subdirectories:
                continue
            name=relative.as_posix()
            included=include_mesh or source.suffix!=".ply"
            files[name]={"sha256":sha(source),"bytes":source.stat().st_size,"bundled":included}
            if included:
                output.write(source,name)
    with ZipFile(target) as bundle:
        assert bundle.testzip() is None
        for name in bundle.namelist():
            with bundle.open(name) as data:
                assert hashlib.file_digest(data,"sha256").hexdigest()==files[name]["sha256"]
    return {"archive_path":target.name,"archive_sha256":sha(target),
            "source_directory":str(directory),"source_files":files}


def main(formal_only):
    audit=read(ROOT/"completion_audit.json")
    assert audit["status"]=="verified_complete_frozen_simulation_matrix"
    assert audit["formal_summary_sha256"]==sha(ROOT/"results/formal/summary.json")
    where=ROOT/("publication_formal" if formal_only else "publication")
    payload=where/"payload"
    payload.mkdir(parents=True,exist_ok=False)
    bundle=payload/"evidence_bundles"
    bundle.mkdir()
    sources={*ROOT.glob("*.py"),*ROOT.glob("*.md"),*ROOT.glob("*.json"),
             *(ROOT/"runtime").glob("*.py"),*(ROOT/"runtime").glob("*.md"),
             *(ROOT/"measurement").glob("*.py"),*(ROOT/"measurement").glob("*.json"),
             *(ROOT/"scenes").glob("*.py"),*(ROOT/"scenes").glob("*.json"),*(ROOT/"scenes").glob("*.md"),
             *(ROOT/"scenes/pair_audits").glob("*.json"),*(ROOT/"results").rglob("*"),
             *(ROOT/"paper").glob("*.pdf"),*(ROOT/"paper").glob("*.svg"),
             *(ROOT/"paper").glob("*.md"),*(ROOT/"paper").glob("*.json")}
    scene_packages = {name:package(ROOT/"scenes"/name,bundle/f"scenes_{name}.zip",include_mesh=True)
                      for name in ("prepared","precomputed_assets")}
    evaluator=ROOT/"scenes/evaluator_references"
    scene_folders=sorted(p.name for p in evaluator.iterdir() if p.is_dir())
    for start in range(0,len(scene_folders),4):
        name=f"evaluator_references_{start//4+1}"
        scene_packages[name]=package(evaluator,bundle/f"scenes_{name}.zip",include_mesh=True,
                                     subdirectories=scene_folders[start:start+4])
    runs=[]
    for phase in ("development","formal"):
        freeze=ROOT/f"{phase.upper()}_FREEZE.json"
        frozen=read(freeze)
        verify(frozen["actor_files_sha256"])
        verify(frozen["evaluator_files_sha256"])
        for layout in frozen["placements"]:
            place=ROOT/"execution"/phase/layout
            sources.add(place/"placement_receipt.json")
            for receipt_path in sorted(place.glob("*/*/exit.json")):
                receipt=read(receipt_path)
                assert receipt["phase_freeze_sha256"]==sha(freeze)
                name=f"{phase}_{layout}_{receipt_path.parent.parent.name}_{receipt_path.parent.name}.zip"
                runs.append(package(receipt_path.parent,bundle/name) | {
                    "physical_actor_arm":receipt["execution_arm"],"placement":layout,
                    "exit_code":receipt["exit_code"],"manifest_sha256":receipt["manifest_sha256"],
                })
        sources.update((ROOT/"execution"/phase).glob("*_receipt.json"))
    followup=ROOT/"followup"
    witness_packages=[]
    projection_packages=[]
    if not formal_only and followup.is_dir():
        sources.update(followup.glob("*.md"))
        for study in sorted(p for p in followup.iterdir() if p.is_dir()):
            sources.update(p for p in study.iterdir()
                           if p.is_file() and p.suffix in {".py",".md",".csv",".json"}
                           and not p.name.endswith(".preflight.json"))
            for name in ("results","formal"):
                sources.update(p for p in (study/name).glob("*")
                               if p.is_file() and p.suffix in {".md",".csv",".json"})
        witness=followup/"local_witness"
        if (witness/"execution_receipt.json").is_file():
            for report in sorted((witness/"runs").glob("*/*/*/exit.json")):
                placement,route,attempt=report.relative_to(witness/"runs").parts[:3]
                name=f"witness_{placement}_{route}_{attempt}.zip"
                witness_packages.append(package(report.parent,bundle/name))
        cold_projection=followup/"west_projection_native"
        for attempt in sorted(cold_projection.glob("attempt_*")):
            if (attempt/"receipt.json").is_file() or (attempt/"failure.json").is_file():
                projection_packages.append(package(attempt,bundle/f"west_projection_{attempt.name}.zip"))
                sources.update(attempt.glob("*.json"))
                sources.add(attempt/"native/manifest.json")
    for path in sorted(sources):
        if path.is_file() and "__pycache__" not in path.parts:
            target=payload/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    write(payload/"EVIDENCE_INDEX.json",{
        "study":"paired frontal occlusion shelf simulation",
        "reference_preparation_bundles":scene_packages,"native_process_attempts":runs,
        "separate_witness_attempts":witness_packages,
        "separate_projection_attempts":projection_packages,
        "completion_audit_sha256":sha(ROOT/"completion_audit.json"),
        "raw_actor_ply_retention":"Complete evaluation meshes retained at the indexed native paths on the Jetson.",
    })
    plan=[]
    for path in sorted(p for p in payload.rglob("*") if p.is_file()):
        bytes_=path.read_bytes()
        plan.append({"stage_path":str(path),"repo_path":PREFIX+"/"+path.relative_to(payload).as_posix(),
                     "size":len(bytes_),"sha256":hashlib.sha256(bytes_).hexdigest(),
                     "git_sha1":hashlib.sha1(f"blob {len(bytes_)}\0".encode()+bytes_).hexdigest()})
    write(where/"upload.json",plan)
    print(f"Staged {len(plan)} files, {sum(r['size'] for r in plan):,} bytes")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-only",action="store_true")
    main(parser.parse_args().formal_only)
