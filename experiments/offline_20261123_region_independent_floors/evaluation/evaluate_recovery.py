#!/usr/bin/env python3
"""Evaluate sealed runs with the published regional guard schema recognized."""

import argparse
from pathlib import Path

import evaluate as frozen


REGIONAL_FIELDS = {"paid", "option_id", "selection_call_id", "reason", "motion"}
REGIONAL_KIND = "regional_guard_rejection"


def recovered_guards(directory, case, cap, actual):
    observations = []
    for name in ("audit.json", "regional_guards.json", "frontier_events.json"):
        file = directory / name
        if not file.is_file():
            continue
        for index, event in enumerate(frozen.read(file)):
            if "event" in event:
                kind = event["event"]
            else:
                frozen.require(name == "regional_guards.json" and event.keys() == REGIONAL_FIELDS
                               and isinstance(event["paid"], int) and isinstance(event["selection_call_id"], int)
                               and isinstance(event["option_id"], str) and bool(event["option_id"])
                               and isinstance(event["reason"], str) and bool(event["reason"])
                               and event["motion"] in ("forward", "left", "right"),
                               f"Unrecognized untagged native event: {file}/{index}")
                kind = REGIONAL_KIND
            if not any(x in kind for x in ("guard_", "rejection", "rejected", "blocked")) and "motion" not in event:
                continue
            paid = event.get("paid_before", event.get("paid", event.get("frame_id")))
            paid = int(paid.removeprefix("frame-")) if isinstance(paid, str) else paid
            frozen.require(paid is None or 0 <= paid <= actual <= cap,
                           f"Guard evidence clocks outside paid trace: {file}/{index}")
            assess = event.get("assessment", {})
            observations.append(case | {"record_kind": "native_event", "source_file": str(file),
                                 "source_row_index": index, "event": kind, "paid": paid,
                                 "reason": assess.get("reason", event.get("reason")),
                                 "phase": event.get("phase"), "action": assess.get("action", event.get("motion")),
                                 "event_data": event})
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--known-smoke", action="store_true",
                        help="Check the recovered evaluator on the archived floors with a fresh --out.")
    args = parser.parse_args()
    if args.known_smoke and args.out is None:
        parser.error("--known-smoke requires a separate --out")
    root = frozen.ARCHIVE if args.known_smoke else frozen.ROOT
    output = (args.out or root / "results/formal").resolve()
    original_guards = frozen.guards
    try:
        frozen.guards = recovered_guards
        frozen.evaluate(root, output, args.known_smoke)
    finally:
        frozen.guards = original_guards
    frozen.write_json(output / "evaluation_recovery.json", {
        "schema": "nso_evaluation_recovery_v1",
        "failure": "KeyError in frozen evaluator on untagged native regional guard rejections",
        "normalization": {"source": "regional_guards.json", "required_keys": sorted(REGIONAL_FIELDS),
                          "derived_csv_event": REGIONAL_KIND, "raw_event_data_preserved": True},
        "frozen_evaluator_sha256": frozen.sha(Path(frozen.__file__)),
        "recovery_evaluator_sha256": frozen.sha(Path(__file__)),
        "measurement_audit_sha256": frozen.sha(output / "measurement_audit.json"),
    })


if __name__ == "__main__":
    main()
