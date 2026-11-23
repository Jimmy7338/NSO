#!/usr/bin/env python3
"""Stage immutable result records, source and compact native traces."""

import hashlib
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from study import ROOT, read, sha, verify, write


REPOSITORY_DIRECTORY = "experiments/offline_20261123_shelf_time_opportunity"


def bundle(folder, target):
    index={}
    with ZipFile(target,"x",compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            relative=path.relative_to(folder).as_posix()
            included=path.suffix!=".ply"
            index[relative]={"sha256":sha(path),"bytes":path.stat().st_size,"bundled":included}
            if included:
                archive.write(path,relative)
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream,"sha256").hexdigest()==index[name]["sha256"]
    return {"relative_source_directory":folder.relative_to(ROOT).as_posix(),
            "bundle":target.name,"sha256":sha(target),"files":index}


def stage():
    paths={
        *(p for p in ROOT.iterdir() if p.suffix in {".json",".py",".md"} and p.is_file()),
        *ROOT.glob("history/*"),
        *ROOT.glob("bank/*"),
        *ROOT.glob("runtime/*.py"),
        *ROOT.glob("runtime/*.md"),
        *ROOT.glob("runtime/preflight/*/diagnosis.json"),
        *ROOT.glob("paper/*"),
        *ROOT.glob("independent/*.py"),
        *ROOT.glob("independent/*.json"),
        *ROOT.glob("recovery/*.json"),
        *ROOT.glob("recovery/results/*"),
        ROOT/"execution/receipt.json",
        ROOT/"recovery/execution/receipt.json",
    }
    publish=ROOT/"publication"
    payload=publish/"payload"
    archives=payload/"evidence_bundles"
    archives.mkdir(parents=True,exist_ok=False)
    diagnostic=read(ROOT/"recovery/results/summary.json")
    verify({str(ROOT/"recovery/results"/p):d for p,d in diagnostic["result_files_sha256"].items()})
    trials=[]
    for phase, directory in (("initial",ROOT/"execution"),("recovery",ROOT/"recovery/execution")):
        for proof in sorted(directory.glob("*/*/attempt_*/exit.json")):
            placement, route, attempt, _=proof.relative_to(directory).parts
            trials.append(bundle(proof.parent,archives/f"{phase}_{placement}_{route}_{attempt}.zip"))
    for source in sorted(paths):
        if not source.is_file():
            continue
        target=payload/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    write(payload/"EVIDENCE_INDEX.json",{"stage_I_native_attempts":trials,
          "meshes":"Native PLY files are retained at their recorded Jetson locations; compact bundles include their hashes.",
          "bank_sha256":sha(ROOT/"bank/BANK_FREEZE.json"),
          "opportunity_summary_sha256":sha(ROOT/"recovery/results/summary.json")})
    records=[]
    for path in sorted(p for p in payload.rglob("*") if p.is_file()):
        raw=path.read_bytes()
        records.append({"repo_path":f"{REPOSITORY_DIRECTORY}/{path.relative_to(payload).as_posix()}",
            "stage_path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),
            "git_sha1":hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest(),"size":len(raw)})
    write(publish/"upload.json",records)
    print(f"Staged {len(records)} files, {sum(r['size'] for r in records):,} bytes.",flush=True)


if __name__=="__main__":
    stage()
