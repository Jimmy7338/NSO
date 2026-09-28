#!/usr/bin/env python3
"""Execute the frozen measured-direction-feedback V10.3 comparison."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_cpu_semantic_inspection_v10_2 import execute, comparison, write_json
from utils.cpu_protocol import file_hash


def main(prepared, output):
    output.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/"configs/virtual3d/cpu_four_module_v10_3_development_protocol.json"
    sources=[Path(__file__).resolve(),protocol,ROOT/"nso/cpu_four_modules_v10.py",
        ROOT/"nso/observed_gain_calibration_v10_1.py",ROOT/"nso/hierarchical_options_v10.py",
        ROOT/"nso/runtime_integration.py",ROOT/"scripts/run_cpu_semantic_inspection_v10_2.py"]
    frozen={str(p.relative_to(ROOT)):file_hash(p) for p in sources}
    write_json(output/"manifest.json",{"schema_version":"cpu_semantic_inspection_v10_3/1",
        "status":"running","protocol":json.loads(protocol.read_text()),"source_sha256":frozen,
        "worlds_constructed_before_source_freeze":0,"raw_sensor_files_saved":False})
    rows=[]
    for condition in ("S","G","N","X","S_no_feedback"):
        for context in ("Q0","Q1","Q2","Q3"):
            for arrangement in ("shelf_west","shelf_east"):
                row=execute(prepared,context,arrangement,condition,revision="v10_3",novelty_floor=.25)
                rows.append(row);write_json(output/"partial.json",rows)
                print(condition,context,arrangement,row["new_area_times_final_f1_05cm"],flush=True)
    conditions=("S","G","N","X","S_no_feedback")
    by={c:{"mean_joint":float(np.mean([r["new_area_times_final_f1_05cm"] for r in rows if r["condition"]==c])),
           "mean_area":float(np.mean([r["new_unique_surface_area_m2"] for r in rows if r["condition"]==c])),
           "mean_f1_gain":float(np.mean([r["f1_gain_05cm"] for r in rows if r["condition"]==c]))}
        for c in conditions}
    comparisons={"S_vs_"+b:comparison(rows,"S",b) for b in ("G","N","X","S_no_feedback")}
    gates={"all_safe":all(not r["failed"] and not r["collisions"] and r["returned_to_anchor"] for r in rows),
           "all_four_interfaces":all(r["modules_called"]==["IGCR","OV-SDF","RPN-UQ","STGHP"] for r in rows),
           "S_mean_exceeds_all":all(v["mean_difference"]>0 for v in comparisons.values())}
    write_json(output/"results.json",rows);write_json(output/"summary.json",
        {"by_condition":by,"comparisons":comparisons,"gates":gates,"passed":all(gates.values()),"branches":len(rows)})
    write_json(output/"manifest.json",{"schema_version":"cpu_semantic_inspection_v10_3/1",
        "status":"complete","protocol":json.loads(protocol.read_text()),"source_sha256":frozen,
        "worlds_constructed_before_source_freeze":0,"raw_sensor_files_saved":False})
    if any(file_hash(ROOT/name)!=digest for name,digest in frozen.items()):raise RuntimeError("source changed")
    write_json(output/"artifact_hashes.json",{str(p.relative_to(output)):file_hash(p) for p in output.rglob("*")
        if p.is_file() and p.name!="artifact_hashes.json"})
    print(json.dumps({"by_condition":by,"comparisons":comparisons,"gates":gates},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.prepared.resolve(),a.output.resolve())
