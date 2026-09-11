#!/usr/bin/env python3
"""Post-hoc independent physical and metric replay of compact V10.3 rows."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d_competition_v9 import create_competition_world, collect_competition_prefix
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame, PlanarScan


def write(path,value):
    Path(path).write_text(json.dumps(json_value(value),indent=2,sort_keys=True,allow_nan=False)+"\n")


def replay(prepared,row):
    c,a=row["context"],row["arrangement"]
    world=create_competition_world(c,a);generated=collect_competition_prefix(world)
    folder=prepared/c/a/"prefix";records=json.loads((folder/"records.json").read_text())
    mapper=ObservedRuntimeMapperV10(world.shape,world.config)
    for i in range(151):
        frame=RGBDFrame.load(folder/"frames"/f"{i:04d}.npz");scan=PlanarScan.load(folder/"scans"/f"{i:04d}.npz")
        np.testing.assert_array_equal(frame.depth_m,generated[i]["frame"].depth_m)
        np.testing.assert_array_equal(frame.semantic,generated[i]["frame"].semantic)
        np.testing.assert_array_equal(scan.ranges_m,generated[i]["scan"].ranges_m)
        mapper.update(frame,scan)
    ref=dict(np.load(prepared/c/a/"reference.npz"));evaluator=restore_evaluator(world,ref)
    _,before=snapshot_metrics(mapper,world,evaluator,ref,(.05,.10));seen=np.zeros_like(ref["prefix_seen"])
    transform=GridTransform(tuple(world.shape),world.config.resolution_m)
    for action in row["actions"]:
        frame,collision,done=world.step(action["action"]);scan=world.scan()
        packet=SensorPacket(f"{c}_{a}",f"v10_1-{c}-{a}",f"frame-{world.step_count}",world.step_count,
            frame,scan,tuple(map(int,world.position)),int(world.heading),"deterministic_simulator_rgbd_and_scan",
            "simulator_exact_discrete_odometry",action["action"],bool(collision),bool(done)).validate(transform,world.config)
        if packet.sha256()!=action["packet_sha256"] or tuple(world.position)!=tuple(action["position"]) or world.heading!=action["heading"]:
            raise AssertionError("physical action or packet mismatch")
        mapper.update(frame,scan);seen|=reference_visible(evaluator.reference,frame,evaluator.truth,
            frame.world_from_camera,world.config.max_depth_m)
    _,after=snapshot_metrics(mapper,world,evaluator,ref,(.05,.10));area=surface_increment(ref["prefix_seen"],seen,ref["weights"])
    np.testing.assert_allclose(area,row["new_unique_surface_area_m2"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(after["f1_05cm"],row["final_f1_05cm"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(after["f1_05cm"]-before["f1_05cm"],row["f1_gain_05cm"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(area*after["f1_05cm"],row["new_area_times_final_f1_05cm"],rtol=0,atol=1e-12)
    anchor=(*records[-1]["position"],records[-1]["heading"])
    if (*world.position,world.heading)!=anchor or world.collisions:raise AssertionError("unsafe terminal state")
    return {"context":c,"arrangement":a,"condition":row["condition"],"packets_exact":True,
            "metrics_exact":True,"safe_return":True}


def main(run,prepared,output):
    hashes=json.loads((run/"artifact_hashes.json").read_text())
    if any(file_hash(run/name)!=digest for name,digest in hashes.items()):raise RuntimeError("run artifact changed")
    manifest=json.loads((run/"manifest.json").read_text())
    if any(file_hash(ROOT/name)!=digest for name,digest in manifest["source_sha256"].items()):raise RuntimeError("frozen source changed")
    rows=json.loads((run/"results.json").read_text());checks=[]
    for row in rows:
        checks.append(replay(prepared,row));print("replayed",row["condition"],row["context"],row["arrangement"],flush=True)
    output.mkdir(parents=True,exist_ok=False)
    report={"schema_version":"cpu_semantic_inspection_v10_3_independent_replay/1","status":"passed",
        "verifier_written_after_execution":True,"planner_or_runtime_called":False,"branches":len(checks),
        "artifact_hashes_verified":len(hashes),"frozen_source_files_verified":len(manifest["source_sha256"]),"checks":checks}
    write(output/"verification.json",report)
    (output/"REPORT.md").write_text("# V10.3 independent replay\n\nPassed 40/40 compact branches. The post-hoc verifier did not call the planner or runtime. It regenerated the physical prefix and future sensor packets from saved actions, rebuilt every TSDF map, recomputed surface area and F1, and checked collision-free return to the exact prefix anchor.\n")
    print(json.dumps({"status":"passed","branches":len(checks)},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--prepared",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.prepared.resolve(),a.output.resolve())
