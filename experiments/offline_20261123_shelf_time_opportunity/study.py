#!/usr/bin/env python3
"""Seal and execute the fixed development route bank."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT.parent
PROTOCOL = ROOT / "PROTOCOL.json"
BANK = ROOT / "bank/BANK_FREEZE.json"
FREEZE = ROOT / "DEVELOPMENT_FREEZE.json"
RESULT = ROOT / "execution"
ACTOR = ROOT / "runtime/run.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Sealed bytes changed: {path}")


def sealed():
    freeze = read(FREEZE)
    verify(freeze["input_sha256"])
    return freeze, read(BANK)


def selected(placement, case):
    if not re.fullmatch(r"[a-z0-9_]+", placement):
        raise ValueError(f"Invalid placement: {placement}")
    rows = case["routes"]
    unique = {row["route_id"] for row in rows}
    if len(unique) != len(rows) or not all(re.fullmatch(r"[A-Za-z0-9_+-]+", key) for key in unique):
        raise ValueError(f"Invalid route identities for {placement}")
    if not set(case["selected"].values()) <= unique or not {"D", "S"} <= case["selected"].keys():
        raise ValueError(f"Missing predeclared routes for {placement}")
    geometric = {v for k, v in case["selected"].items() if k.startswith("geometry_")}
    if len(geometric - {case["selected"]["D"], case["selected"]["S"]}) < 2:
        raise ValueError(f"Fewer than two distinct geometric alternatives for {placement}")
    if any(not row["outbound_actions"] or row["outbound_states"][-1] != case["target_pose"]
           or row["outbound_cost"] != len(row["outbound_actions"])
           or row["reserved_return_cost"] != len(row["return_actions"])
           or row["outbound_cost"] + row["reserved_return_cost"] + case["paid_before"] > 192
           for row in rows):
        raise ValueError(f"Invalid route or return certificate for {placement}")
    return rows


def seal():
    if RESULT.exists():
        raise FileExistsError("Seal before creating execution outputs.")
    protocol, bank = read(PROTOCOL), read(BANK)
    placements = protocol["development_order"]
    if list(bank["placements"]) != placements:
        raise ValueError("Stage-I bank order or placements differ from the protocol.")
    expected = (("native_budget",192),("tie_tolerance_m2",1e-9),("primary_task_paid",96))
    for key, value in expected:
        if protocol.get(key) != value or bank.get(key, bank.get("tolerance_m2")) != value:
            raise ValueError(f"Stage-I bank differs from the protocol in {key}")
    for name, case in bank["placements"].items():
        selected(name, case)
    paths = {PROTOCOL, BANK, ROOT/"SOURCE_RECEIPT.json", Path(__file__), ROOT/"measure.py", ACTOR}
    paths.update((ROOT/"bank").glob("*.py"))
    paths.update((ROOT/"runtime").glob("*.py"))
    keys = ("source_bank_path", "source_candidates_path")
    paths.update(Path(case[k]) for case in bank["placements"].values() for k in keys if case.get(k))
    write(FREEZE, {"phase":"known_development_placements", "placement_order":placements,
                   "frozen_unix_s":time.time(), "base_commit":protocol["base_commit"],
                   "input_sha256":{str(p):sha(p) for p in sorted(paths)}})
    print(f"Frozen {len(placements)} placements, {sha(FREEZE)}", flush=True)


def launch(parent, case, route, attempt):
    attempt.mkdir()
    path = attempt/"run"
    command = [sys.executable, "-X", "faulthandler", str(ACTOR), "--bank", str(BANK),
               "--placement", case, "--route", route, "--out", str(path)]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                       PYTHONDONTWRITEBYTECODE="1")
    with (attempt/"worker.log").open("x") as log:
        task = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT)
        receipt = {"placement":case,"route_id":route,"pid":task.pid,"start_unix_s":time.time(),
                   "command":command,"directory":str(path),"freeze_sha256":sha(FREEZE)}
        write(attempt/"launch.json", receipt)
        code = task.wait()
    valid = code == 0 and (path/"manifest.json").is_file()
    outcome = receipt | {"status":"new_execution" if valid else "technical_failure",
                         "returncode":code, "stop_unix_s":time.time(), "log_sha256":sha(attempt/"worker.log"),
                         "manifest_sha256":sha(path/"manifest.json") if valid else None,
                         "attempt_sha256":{str(p):sha(p) for p in parent.glob("attempt_*/launch.json")}}
    write(attempt/"exit.json", outcome)
    print(f"{case} {route}: {outcome['status']} ({code})", flush=True)
    return outcome


def physical(case, row):
    route = row["route_id"]
    base = RESULT/case/route
    base.mkdir(parents=True, exist_ok=True)
    reuse = row.get("closure_reuse")
    if reuse:
        directory = Path(reuse["run_directory"] if "run_directory" in reuse else reuse["run"])
        manifest = directory/"manifest.json"
        if sha(manifest) != reuse["manifest_sha256"]:
            raise ValueError(f"Changed reused execution: {case}, {route}")
        return {"placement":case,"route_id":route,"status":"reused_closed_loop","directory":str(directory),
                "manifest_sha256":sha(manifest),"freeze_sha256":sha(FREEZE)}
    attempts = sorted(base.glob("attempt_*"))
    previous = None
    for attempt in attempts:
        previous = read(attempt/"exit.json")
        if previous["freeze_sha256"] != sha(FREEZE):
            raise ValueError("Existing attempt has different frozen inputs.")
    for index in range(len(attempts), 2):
        if previous and previous["status"] == "new_execution":
            break
        previous = launch(base,case,route,base/f"attempt_{index+1:03d}")
    return previous | {"attempt_exits_sha256":{str(p):sha(p) for p in (base).glob("attempt_*/exit.json")}}


def run(workers):
    if not 1 <= workers <= 6:
        raise ValueError("Use one to six native processes.")
    freeze, bank = sealed()
    RESULT.mkdir(exist_ok=True)
    receipt = RESULT/"receipt.json"
    with (RESULT/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            print(receipt)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [pool.submit(physical,case,row) for case in freeze["placement_order"]
                     for row in selected(case,bank["placements"][case])]
            outcomes = [future.result() for future in tasks]
        write(receipt, {"freeze_sha256":sha(FREEZE),"outcomes":outcomes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("seal","run"))
    parser.add_argument("--workers",type=int,default=6)
    args = parser.parse_args()
    seal() if args.command=="seal" else run(args.workers)
