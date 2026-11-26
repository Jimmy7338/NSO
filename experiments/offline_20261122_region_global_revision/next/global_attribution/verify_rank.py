#!/usr/bin/env python3
"""Check the executable ranking against the independently audited historical pool."""

from collections import Counter
from csv import DictReader
from pathlib import Path
import sys
from types import SimpleNamespace

from runner import PREVIOUS, ROOT, read
from score import rerank


def main():
    binding = read(PREVIOUS/"regression/active_binding.json")
    sys.path.append(str(Path(binding["method_source"]).parents[1]))
    from policy.observed_regions import RegionConfig

    ledger = ROOT.parent/"access_audit"/"dispatches.csv"
    with ledger.open(newline="") as stream:
        events = list(DictReader(stream))
    outcomes = Counter()
    for event in events:
        if event["decision_reason"] != "selected_observed_region_view":
            continue
        before, key = int(event["paid_before"]), int(event["selection_call_id"])
        diagnostic = next(item for item in read(PREVIOUS/event["regional_choice_archive"])
                          if item["selection_call_id"] == key and item["paid_before"] == before)
        native = next(item["outputs"] for item in read(PREVIOUS/event["native_score_archive"])
                      if item["call_id"]==key)
        proposals = {tuple(item["pose"]):item["proposal"] for item in diagnostic["certification"]}
        planner = SimpleNamespace(config=RegionConfig(),latest_proposals=proposals)
        for target,outbound,expected in (
            (False,False,"old_pose"), (True,False,"target_region_only_pose"),
            (False,True,"outbound_measured_only_pose"), (True,True,"both_pose"),
        ):
            chosen, rows = rerank(planner, diagnostic["ranking"], native["candidates"],
                                  native["score_audit"], .20, target, outbound)
            assert chosen["pose"] == list(map(int,event[expected].split(","))), (event["state_uid"],expected)
            assert {tuple(r["pose"]) for r in rows} == {
                tuple(r["pose"]) for r in diagnostic["ranking"]["ranking"]}
            outcomes[expected]+=1
    assert len(outcomes)==4 and set(outcomes.values())=={17},outcomes
    print("Verified all 68 rankings on the 17 factual states",dict(outcomes))


if __name__=="__main__":
    main()
