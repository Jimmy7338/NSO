#!/usr/bin/env python3
"""Execute each distinct frozen route in an independent simulator process."""

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import time

from run import ROOT, read, sha
from seal import paths, write
from measure import SOURCE


def execute(layout, row, inputs):
    route = row["route_id"]
    target = ROOT/"runs"/layout/route
    target.mkdir(parents=True,exist_ok=False)
    directory = target/"run"
    command = [sys.executable,str(ROOT/"run.py"),layout,route,str(directory)]
    environment = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONFAULTHANDLER="1")
    launch = {
        "layout":layout,"route_id":route,"command":command,"route_freeze_sha256":inputs,
        "started_unix_s":time.time(),
    }
    with (target/"worker.log").open("x") as stream:
        worker = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,cwd=ROOT)
        write(target/"launch.json",launch | {"pid":worker.pid})
        status = worker.wait()
    terminal = directory/"manifest.json"
    evidence = {"status":status,"manifest_sha256":sha(terminal) if terminal.is_file() else None,
                "worker_log_sha256":sha(target/"worker.log"),"finished_unix_s":time.time()}
    write(target/"exit.json",launch | evidence)
    print(layout,route,status,flush=True)
    return launch | evidence


def main():
    freeze = ROOT/"ROUTE_FREEZE.json"
    inputs = sha(freeze)
    plan = read(freeze)
    schedule = [(layout,row) for layout,desc in plan["layouts"].items()
                for row in desc.get("branch_routes",[])]
    assert not (ROOT/"runs").exists()
    write(ROOT/"MEASUREMENT_FREEZE.json",{"files_sha256":paths([
       ROOT/"measure.py",SOURCE,ROOT/"PROTOCOL.json",ROOT/"INPUT_FREEZE.json",freeze
    ])})
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute,layout,row,inputs) for layout,row in schedule]
        receipts = [future.result() for future in futures]
    write(ROOT/"execution_receipt.json",{"route_freeze_sha256":inputs,"episodes":receipts})


if __name__ == "__main__":
    main()
