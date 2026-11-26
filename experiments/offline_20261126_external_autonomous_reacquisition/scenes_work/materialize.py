"""Publish selected preregistered scene inputs from conditional or fresh preparation."""

import argparse
import json
import os
import shutil

import prepare
import prepare_motion
from scene import HERE, SPECS, sha


def materialize(floor):
    layouts=[name for name, spec in SPECS.items() if spec["source_floor"]==floor]
    if len(layouts)!=2:
        raise ValueError("Only two starts on an actually preregistered formal floor can be materialized")
    prepare_motion.prepare(floor)
    conditional=HERE/"conditional_fallback"
    if SPECS[layouts[0]]["scene_role"]=="new_starts_on_previously_used_scan":
        expected=conditional/f"precomputed_assets/{floor}.npz"
        if sha(expected)!=sha(HERE/f"precomputed_assets/{floor}.npz"):
            raise ValueError("Conditional cache differs from final simulator cache")
        for layout in layouts:
            ready=conditional/"prepared"/layout
            registered=SPECS[layout]
            state=json.loads((ready/"initial_state.json").read_text())
            if state["seed"]!=registered["seed"] or state["anchor"]!=[*registered["start"],registered["heading"]]:
                raise ValueError(f"Mismatched conditional start: {layout}")
            reference=conditional/"evaluator_references"/layout
            for name,path in (("prepared",ready),("evaluator_references",reference)):
                destination=HERE/name/layout
                destination.mkdir(parents=True,exist_ok=False)
                for item in path.iterdir():
                    target=destination/item.name
                    if item.name=="RECEIPT.json":
                        shutil.copyfile(item,target)
                    else:
                        os.link(item,target)
            final=HERE/"evaluator_references"/layout/"RECEIPT.json"
            receipt=json.loads(final.read_text())
            final_dependencies={}
            for path,digest in receipt["source_inputs_sha256"].items():
                final_path=path.replace(str(conditional)+"/",str(HERE)+"/",1)
                if sha(final_path)!=digest:
                    raise ValueError(f"Conditional reference input bytes changed while promoting {final_path}")
                final_dependencies[final_path]=digest
            receipt["source_inputs_sha256"]=final_dependencies
            final.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    else:
        common=[]
        for layout in layouts:
            config, mask=prepare.initial(layout)
            common.append((layout,mask))
        prepare.reference(common,config,common[0][1])
    print(f"Formal evaluator and actor packets materialized: {floor}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("floor")
    materialize(parser.parse_args().floor)
