#!/usr/bin/env python3
"""Create a reproducible audit of locally usable learned-model evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    weights = sorted(list((ROOT / "pretrained_models").glob("model_best.*"))
                     + list((ROOT / "trained_models").glob("*/model_best.*"))
                     + [ROOT / "yolov8n.pt"])
    weight_rows = []
    for path in weights:
        prefix = path.read_bytes()[:64]
        weight_rows.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
            git_lfs_pointer=prefix.startswith(b"version https://git-lfs"), sha256=sha(path)))

    rpn_path = ROOT / "eval_results/cpu_rpn_20260910/model.npz"
    with np.load(rpn_path) as model:
        learned_arrays = {name: list(model[name].shape) for name in ("w1", "b1", "w2", "b2")}
        parameter_count = int(sum(model[name].size for name in learned_arrays))
    rpn_report = json.loads((rpn_path.parent / "training_report.json").read_text())
    main_text = (ROOT / "main.py").read_text()
    args_text = (ROOT / "arguments.py").read_text()
    ssc_text = (ROOT / "semantic/ssc_completer.py").read_text()
    v10_text = (ROOT / "nso/cpu_four_modules_v10.py").read_text()

    report = dict(schema_version="network_training_state_audit/1", status="complete",
        checkpoint_files=weight_rows,
        all_listed_large_checkpoints_are_lfs_pointers=all(r["git_lfs_pointer"] for r in weight_rows),
        training_wiring=dict(
            train_semantic_argument_declared="--train_semantic" in args_text,
            train_semantic_referenced_in_main="train_semantic" in main_text,
            deep_ssc_has_not_implemented="DeepSSCCompleter" in ssc_text
                and "NotImplementedError" in ssc_text,
            v10_imports_torch="import torch" in v10_text,
            v10_capability_text="measured RGB-D marker support" in v10_text),
        cpu_candidate_rpn=dict(model=str(rpn_path.relative_to(ROOT)),
            learned_array_shapes=learned_arrays, parameter_count=parameter_count,
            architecture=rpn_report["architecture"], validation=rpn_report["validation"],
            validation_constant_prior=rpn_report["validation_constant_prior"],
            validation_action_budget_rule=rpn_report["validation_action_budget_rule"],
            limitations=rpn_report["limitations"]),
        interpretation=[
            "V10.x is a non-neural measured CPU backend and does not exercise the listed checkpoints.",
            "The semantic-training CLI flag has no corresponding main-loop optimization path.",
            "The available 113-parameter CPU candidate MLP was trained, but the recorded simple action-budget rule outperformed it on its validation split.",
            "Checkpoint pointers and historical training notes are not locally loadable trained-model evidence."
        ],
        audited_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "main.py", ROOT / "arguments.py", ROOT / "semantic/ssc_completer.py",
            ROOT / "nso/cpu_four_modules_v10.py", ROOT / "scripts/train_cpu_rpn.py",
            rpn_path.parent / "training_report.json", rpn_path)})
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False,
        indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], checkpoints=len(weight_rows),
        all_pointers=report["all_listed_large_checkpoints_are_lfs_pointers"],
        cpu_rpn_parameters=parameter_count,
        semantic_training_wired=report["training_wiring"]["train_semantic_referenced_in_main"]),
        indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
