"""Freeze the incomplete factorial arms and reuse the archived supervisor."""

import argparse
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1] / "region_independent_floors_20261123"
BASE_FREEZE = BASE / "FORMAL_FREEZE.json"
FREEZE = ROOT / "FORMAL_FREEZE.json"
RUNS = ROOT / "execution/formal"


def supervisor():
    spec = importlib.util.spec_from_file_location("factor_supervisor", BASE / "study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.FREEZE, module.RUNS = ROOT, FREEZE, RUNS
    return module


def seal(native):
    original = native.read(BASE_FREEZE)
    protocol = native.read(ROOT / "PROTOCOL.json")
    initial = native.read(BASE / "scenes/actor_freeze_holdout.json")
    assert [row["layout"] for row in initial["layouts"]] == protocol["layouts"]
    assert original["budget"] == protocol["budget"] and original["capture_paid"] == protocol["capture_paid"]
    assert original["base_commit"] == protocol["execution_source_base"]
    for key in ("actor_files", "evaluator_files", "simulator_only_files"):
        native.verify(original[key])
    jobs = []
    for layout in protocol["layouts"]:
        for arm, factors in protocol["arms"].items():
            if "archive_arm" in factors:
                continue
            jobs.append({"layout": layout, "arm": arm, "prior": "S", "regional_enabled": True, "literal": True,
                         "repair": {key: factors[key] for key in ("target", "outbound")}})
    actor_files = original["actor_files"] | native.fingerprints(
        ROOT / name for name in ("runner.py", "factor_study.py", "PROTOCOL.json"))
    frozen = original | {"schema": protocol["schema"], "frozen_unix_s": time.time(),
                         "protocol_relative": "PROTOCOL.json", "protocol_sha256": native.sha(ROOT / "PROTOCOL.json"),
                         "jobs": jobs, "actor_files": actor_files, "archive_head": protocol["archive_head"],
                         "inherited_scene_freeze_sha256": native.sha(BASE_FREEZE)}
    native.write(FREEZE, frozen)
    print(f"Frozen {len(jobs)} factorial actor jobs at {native.sha(FREEZE)}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("seal", "run"))
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    module = supervisor()
    seal(module) if args.command == "seal" else module.run(args.workers)
