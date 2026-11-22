#!/usr/bin/env python3
"""Apply the existing independent map evaluator to the revised selector."""

from pathlib import Path
import sys

from runner import PREVIOUS, ROOT


sys.path.insert(0, str(PREVIOUS))
import evaluate as native_evaluation
from regression import read, sha


FORMAL_CONTRASTS = {
    "revision_vs_regional": {"revised_S": 1, "region_S": -1},
    "revision_vs_strict_frontier": {"revised_S": 1, "F_strict": -1},
    "regional_vs_strict_frontier": {"region_S": 1, "F_strict": -1},
}


def main():
    native_evaluation.ROOT = ROOT
    summarize = native_evaluation.contrast_summary

    def contrasts(spec, episodes, phase):
        comparisons = FORMAL_CONTRASTS.copy()
        if phase == "development":
            previous = read(PREVIOUS / "results/formal/summary.json")
            protocol = read(ROOT / "PROTOCOL.json")
            assert previous["run_freeze_sha256"] == protocol["source_formal_freeze_sha256"]
            assert previous["phase_receipt_sha256"] == sha(PREVIOUS / "execution/formal/phase_receipt.json")
            historic = {(row["layout"], row["arm"]): row for row in previous["episodes"]
                        if row["layout"] in spec["layouts"] and row["arm"] in spec["archived_controls"]}
            for key, row in historic.items():
                manifest = Path(row["run_directory"]) / "manifest.json"
                assert sha(manifest) == row["source_manifest_sha256"]
                revised = episodes[(key[0], "revised_S")]
                assert (revised.get("initial_packet_sha256", row["initial_packet_sha256"])
                        == row["initial_packet_sha256"])
            episodes = episodes | historic
            comparisons["revision_vs_archived_native_S"] = {"revised_S": 1, "original_S": -1}
        native_evaluation.FACTORIAL = comparisons
        return summarize(spec, episodes, "formal")

    native_evaluation.contrast_summary = contrasts
    native_evaluation.main()


if __name__ == "__main__":
    main()
