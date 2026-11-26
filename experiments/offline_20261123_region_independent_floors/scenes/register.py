#!/usr/bin/env python3
"""Record the immutable layout recipe before native actor execution."""

import hashlib
import json
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ("LAYOUT_REGISTRATION.json", "scene.py", "prepare_scenes.py", "reference_protocol.json")
    registry = json.loads((HERE / names[0]).read_text())
    output = HERE.parent / "execution"
    layouts = list(registry["layouts"])
    if output.is_dir() and any(path.is_dir() and path.name in layouts for path in output.rglob("*")):
        raise ValueError("Registration must precede all registered actor launches.")
    state = {"frozen_unix_s": time.time(), "registered_layouts": layouts,
             "file_sha256": {name: sha(HERE / name) for name in names},
             "observed_policy_results": 0,
             "reused_unchanged_preparations": sorted(
                 name for name in layouts if (HERE / "prepared" / name / "initial_state.json").is_file())}
    previous = HERE / "preregistration_v1" / "formal_geometry_audit.json"
    if previous.is_file():
        audit = json.loads(previous.read_text())
        if audit["policy_calls"] or len(audit["excluded"]) != 1:
            raise ValueError("Historical registration audit does not match the documented correction.")
        state["previous_rejected_geometry_audit_sha256"] = sha(previous)
    with (HERE / "REGISTRATION_RECEIPT.json").open("x") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(sha(HERE / "REGISTRATION_RECEIPT.json"), flush=True)


if __name__ == "__main__":
    main()
