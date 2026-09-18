#!/usr/bin/env python3
"""Apply the frozen information screen with independent reference resampling."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import open3d as o3d
from env.virtual3d_competition_v9 import create_competition_world
from nso.information_value_v13 import summarize_paired_rewards
from scripts.collect_semantic_gain_v13_history import sha,write
from utils.reconstruction_metrics import ReconstructionEvaluator


def read_mesh(path):
    with np.load(path,allow_pickle=False) as data:
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(data["vertices"])
        mesh.triangles=o3d.utility.Vector3iVector(data["triangles"])
        if "vertex_colors" in data:
            mesh.vertex_colors=o3d.utility.Vector3dVector(data["vertex_colors"])
    return mesh


def main(source,output):
    manifest=json.loads((source/"manifest.json").read_text())
    if manifest["status"]!="complete": raise ValueError("acquisition must finish")
    hashes=json.loads((source/"artifact_hashes.json").read_text())
    for name,expected in hashes.items():
        if sha(source/name)!=expected: raise ValueError("acquisition artifact changed")
    for name,expected in manifest["source_sha256"].items():
        if sha(ROOT/name)!=expected: raise ValueError("source changed")
    protocol=manifest["protocol"]
    original=json.loads((source/"partial.json").read_text())
    acquisition=json.loads((source/"summary.json").read_text())
    replay_checks=[]
    for context in manifest["contexts"]:
        replay=json.loads((source/(context+"_replay")/"summary.json").read_text())
        replay_checks.extend(replay["checks"])
        if replay["status"]!="passed": raise ValueError("physical replay incomplete")
    complete=(len(original)==protocol["expected_candidate_outcomes"]==len(replay_checks)
        and acquisition["collisions"]==acquisition["failures"]==acquisition["budget_violations"]==0
        and acquisition["returns"]==acquisition["first_options_complete"]==len(original))
    output.mkdir(parents=True,exist_ok=False)
    write(output/"manifest.json",dict(status="running",source=str(source),source_sha256=manifest["source_sha256"],
        input_artifact_hashes=hashes,physical_screen_passed=complete))
    results=[]; status="running"
    try:
        for seed in protocol["development_training_screen"]["reference_sample_seeds_for_sensitivity"]:
            rewards=[]; all_metrics=[]; errors=[]
            for context in manifest["contexts"]:
                folder=source/context
                rows=json.loads((folder/"partial.json").read_text())
                parent,arrangement=rows[0]["parent"],rows[0]["arrangement"]
                world=create_competition_world(parent,arrangement)
                evaluator=ReconstructionEvaluator(world,count=16000,seed=seed)
                if seed==2026:
                    with np.load(folder/"reference.npz",allow_pickle=False) as data:
                        np.testing.assert_array_equal(data["points"],evaluator.reference)
                        np.testing.assert_array_equal(data["classes"],evaluator.classes)
                np.savez_compressed(output/f"reference_{context}_{seed}.npz",
                    points=evaluator.reference,classes=evaluator.classes)
                prefix={}
                for row in rows:
                    step=row["checkpoint_action"]
                    if step not in prefix:
                        prefix[step]=evaluator.evaluate(read_mesh(folder/f"prefix_step_{step}.npz"),row["before"]["coverage_2d"])
                    before=prefix[step]
                    name=f"step_{step:03d}_candidate_{row['candidate_id']:03d}"
                    after=evaluator.evaluate(read_mesh(folder/name/"final_mesh.npz"),row["after"]["coverage_2d"])
                    reward=dict(parent=parent,arrangement=arrangement,checkpoint_action=step,
                        candidate_id=row["candidate_id"],joint_gain=after["joint_05cm"]-before["joint_05cm"])
                    rewards.append(reward)
                    all_metrics.append(dict(**reward,before=before,after=after))
                    if seed==2026:
                        errors.append(abs(reward["joint_gain"]-row["joint_gain"]))
                        for when,actual in (("before",before),("after",after)):
                            for key in ("precision_05cm","recall_05cm","f1_05cm","joint_05cm"):
                                errors.append(abs(actual[key]-row[when][key]))
            if seed==2026 and max(errors,default=0.)>1e-12:
                raise ValueError("primary resampling failed to reproduce frozen metrics")
            summary=summarize_paired_rewards(protocol,rewards)
            summary.update(reference_seed=seed,physical_screen_passed=complete,
                           primary_max_metric_error=max(errors,default=0.) if seed==2026 else None)
            results.append(summary)
            write(output/f"metrics_seed_{seed}.json",all_metrics)
            write(output/f"information_seed_{seed}.json",summary)
            print("INFORMATION",seed,summary["mean_information_value"],summary["relative_information_value"],
                  "positive parents",summary["positive_parent_count"],"pass",summary["numerical_information_screen_passed"],flush=True)
        passed=complete and all(r["numerical_information_screen_passed"] for r in results)
        write(output/"summary.json",dict(status="complete",development_information_screen_passed=passed,
            eligible_to_design_grouped_training=passed,semantic_model_efficacy_proven=False,
            independent_confirmation=False,reference_seed_results=results,
            boundary="Finite candidates, fixed continuation, artificial marker semantics and four development parents only."))
        status="complete"
    except Exception as error:
        status="failed"; write(output/"failure.json",dict(error=str(error))); raise
    finally:
        write(output/"manifest.json",dict(status=status,source=str(source),source_sha256=manifest["source_sha256"],
            input_artifact_hashes=hashes,physical_screen_passed=complete))
        write(output/"artifact_hashes.json",{str(p.relative_to(output)):sha(p) for p in output.glob("*")
            if p.is_file() and p.name!="artifact_hashes.json"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); main(args.source.resolve(),args.output.resolve())
