#!/usr/bin/env python3
"""Seal a prospective four-floor matrix and supervise native subprocesses."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/"region_global_revision_20261122/next/global_attribution"
SCENES = ROOT/"scenes"
FREEZE = ROOT/"FORMAL_FREEZE.json"
RUNS = ROOT/"execution/formal"
COMMON = ("budget","capture_paid","local_auxiliary_weights","local_route_invariants")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    staged=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    staged.replace(path)


def verify(files):
    for path,digest in files.items():
        if sha(path)!=digest:
            raise ValueError(f"Frozen file drift: {path}")


def fingerprints(paths):
    return {str(p):sha(p) for p in sorted(set(paths))}


def outside_scene(data):
    return {k:v for k,v in data.items() if not Path(k).is_relative_to(SOURCE/"scenes")}


def seal():
    if RUNS.exists():
        raise ValueError("Register the four floors before opening the execution directory.")
    protocol=read(ROOT/"PROTOCOL.json")
    previous=read(SOURCE/"FORMAL_FREEZE.json")
    settings={k:protocol[k] for k in COMMON}
    assert all(v==previous[k] for k,v in settings.items())
    source_receipt=read(ROOT/"SOURCE_RECEIPT.json")
    assert source_receipt["base_commit"]==protocol["base_commit"]
    assert source_receipt["original_formal_freeze_sha256"]==sha(SOURCE/"FORMAL_FREEZE.json")
    actor_path=SCENES/"actor_freeze_holdout.json"
    eval_path=SCENES/"evaluator_freeze_holdout.json"
    actor,evaluator=read(actor_path),read(eval_path)
    assert actor["layouts"]==evaluator["layouts"]
    assert actor["budget"]==evaluator["budget"]==protocol["budget"]
    assert actor["actors_run"]==evaluator["actors_run"]==0
    layouts=[r["layout"] if isinstance(r,dict) else r for r in actor["layouts"]]
    assert layouts==protocol["layouts"] and len(set(layouts))==4
    jobs=[]
    for layout in layouts:
        for arm in protocol["arms"]:
            treatment={"repair":{"target":True,"outbound":True}} if arm=="combined_S" else {}
            jobs.append({"layout":layout,"arm":arm,"prior":None if arm=="F_strict" else "S",
                         "regional_enabled":arm!="F_strict","literal":True} | treatment)
    initial={SCENES/"scene.py"}
    for layout in layouts:
        scene=SCENES/"prepared"/layout
        initial.update(scene/name for name in ("initial_state.json","initial_frame.npz","initial_scan.npz"))
        cold=read(scene/"initial_state.json")
        assert cold["paid_actions"]==cold["policy_select_calls"]==0 and cold["stop_budget"]==192
    inputs=fingerprints(initial)
    final_protocol=ROOT/"FORMAL_PROTOCOL.json"
    write(final_protocol,protocol | {"source_actor_freeze_sha256":sha(SOURCE/"FORMAL_FREEZE.json"),
                                   "registered_scene_ledger_sha256":sha(actor_path)})
    actor_files=outside_scene(previous["actor_files"]) | actor["files_sha256"] | inputs
    actor_files |= fingerprints([ROOT/"runner.py",Path(__file__),final_protocol,actor_path,ROOT/"SOURCE_RECEIPT.json"])
    cache=actor.get("simulator_only_collision_cache_sha256",{})
    offline=outside_scene(previous["evaluator_files"]) | evaluator["files_sha256"]
    offline |= fingerprints([eval_path,ROOT/"evaluation/evaluate.py",ROOT/"known/audit.json"])
    assert not any("evaluator_ref" in p or "evaluator_truth" in p or "precomputed_assets" in p
                   for p in actor_files)
    for source in (actor_files,cache,offline):
        verify(source)
    write(FREEZE,settings | {"schema":protocol["schema"],"phase":"formal","frozen_unix_s":time.time(),
                            "protocol_relative":final_protocol.name,"protocol_sha256":sha(final_protocol),
                            "actor_files":actor_files,"evaluator_files":offline,"actor_initial_files":inputs,
                            "simulator_only_files":cache,"jobs":jobs,"scene_root":str(SCENES),
                            "base_commit":protocol["base_commit"]})
    print(f"Frozen {len(jobs)} scheduled simulations: {sha(FREEZE)}",flush=True)


def attempt(job, directory):
    directory.mkdir()
    run=directory/"run"
    command=[sys.executable,"-X","faulthandler",str(ROOT/"runner.py"),job["layout"],job["arm"],
             "--freeze",str(FREEZE),"--out",str(run)]
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    log=directory/"worker.log"
    started=time.time()
    with log.open("x") as stream:
        worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        launch=job | {"pid":worker.pid,"started_unix_s":started,"command":command,"freeze_sha256":sha(FREEZE)}
        write(directory/"launch.json",launch)
        code=worker.wait()
    result=run/("manifest.json" if code==0 else "failure.json")
    if code==0 and not result.is_file():
        raise ValueError(f"Actor reported success without a completed manifest: {run}")
    receipt=launch | {"exit_code":code,"observed_exit_unix_s":time.time(),"run_path":str(run),
        "result_path":str(result),"result_sha256":sha(result) if result.is_file() else None,
        "worker_log_sha256":sha(log)}
    write(directory/"exit.json",receipt)
    print(job["layout"],job["arm"],directory.name,"exit",code,flush=True)
    return receipt


def run_job(job):
    root=RUNS/job["layout"]/job["arm"]
    root.mkdir(parents=True,exist_ok=True)
    existing=sorted(root.glob("attempt_*"))
    last=None
    for path in existing:
        last=read(path/"exit.json")
        assert last["freeze_sha256"]==sha(FREEZE) and last["worker_log_sha256"]==sha(path/"worker.log")
        if last["exit_code"]==0:
            assert sha(last["result_path"])==last["result_sha256"]
            return last
    for index in range(len(existing)+1,3):
        last=attempt(job,root/f"attempt_{index:03d}")
        if last["exit_code"]==0:
            break
    return last


def run(workers):
    assert 1<=workers<=6
    frozen=read(FREEZE)
    for files in ("actor_files","evaluator_files","simulator_only_files"):
        verify(frozen[files])
    RUNS.mkdir(parents=True,exist_ok=True)
    final=RUNS/"phase_receipt.json"
    with (RUNS/"supervisor.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if final.is_file():
            raise ValueError("The registered run already has an execution receipt.")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scheduled=[pool.submit(run_job,job) for job in frozen["jobs"]]
            rows=[future.result() for future in scheduled]
        write(final,{"freeze_sha256":sha(FREEZE),"phase":"formal","episodes":rows})
    print(sum(row["exit_code"]==0 for row in rows),"of",len(rows),"native episodes completed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("seal","run"))
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    seal() if args.command=="seal" else run(args.workers)
