#!/usr/bin/env python3
"""Run the same registered development bank with corrected receipt normalization."""

import argparse
from pathlib import Path
import time

import numpy as np

import measure
import study


ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT/"recovery"
ORIGINAL_FREEZE = ROOT/"DEVELOPMENT_FREEZE.json"
ORIGINAL_EXECUTION = ROOT/"execution/receipt.json"
FROZEN = RECOVERY/"DEVELOPMENT_FREEZE.json"
RUNS = RECOVERY/"execution"
ACTOR = ROOT/"runtime/run_recovery.py"


def seal():
    original=study.read(ORIGINAL_FREEZE)
    study.verify(original["input_sha256"])
    attempts=study.read(ORIGINAL_EXECUTION)
    if attempts["freeze_sha256"]!=study.sha(ORIGINAL_FREEZE):
        raise ValueError("Missing original attempt provenance.")
    if any(row["status"] not in {"technical_failure","reused_closed_loop"} for row in attempts["outcomes"]):
        raise ValueError("An actual version-one execution is already available. Audit before changing actor versions.")
    for file in (ROOT/"execution").glob("*/*/attempt_*/exit.json"):
        record=study.read(file)
        run=Path(record["directory"])
        steps=run/"steps.json"
        if steps.exists() and study.read(steps):
            raise ValueError(f"Original branch has charged actions: {run}")
    RECOVERY.mkdir(exist_ok=True)
    paths={ORIGINAL_FREEZE,ORIGINAL_EXECUTION,Path(__file__),ACTOR}
    study.write(FROZEN, {"phase":"known_development_placements",
        "placement_order":original["placement_order"],"base_commit":original["base_commit"],
        "frozen_unix_s":time.time(),"bank_sha256":study.sha(study.BANK),
        "original_sha256":study.sha(ORIGINAL_FREEZE),"original_attempts_sha256":study.sha(ORIGINAL_EXECUTION),
        "scope":"Same selected routes and native experiment, with receipt serialization and area evaluator repair.",
        "input_sha256":original["input_sha256"]|{str(p):study.sha(p) for p in sorted(paths)}})
    print(f"Frozen recovery {study.sha(FROZEN)}",flush=True)


def facilities(before, after, weights):
    weights=np.broadcast_to(np.asarray(weights,dtype=float),before.shape)
    gross=measure.area(after & ~before,weights)
    lost=measure.area(before & ~after,weights)
    return {"gross_new_m2":gross,"lost_m2":lost,"net_m2":gross-lost}


def evaluate():
    original_factory=measure.measurement_engine

    def area_engine():
        meter, weighted, metric=original_factory()
        meter.facility_metrics=facilities
        return meter, weighted, metric

    measure.measurement_engine=area_engine
    measure.ROOT,measure.RESULT,measure.FREEZE=RECOVERY,RUNS,FROZEN
    measure.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("seal","run","measure"))
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    if args.command=="seal":
        seal()
        return
    study.FREEZE,study.RESULT,study.ACTOR=FROZEN,RUNS,ACTOR
    if args.command=="run":
        study.run(args.workers)
    else:
        evaluate()


if __name__=="__main__":
    main()
