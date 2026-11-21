#!/usr/bin/env python3
"""Evaluate finalized predictions only; never starts a sensor or policy rollout."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'audit';OUT=HERE/'particle_replay';OUT.mkdir(exist_ok=True)
STATES=[('shelf_left',24),('shelf_right',24),('shelf_left',54),('shelf_left',109),('shelf_right',53),('shelf_right',110)]
while True:
    pending=[]
    for layout,paid in STATES:
        target=OUT/f'{layout}_{paid}';source=ROOT/'replay'/f'{layout}_{paid}_attempt_001'
        if (target/'result.json').exists():continue
        if not(source/'result.json').exists():pending.append((layout,paid));continue
        assert json.loads((source/'result.json').read_text())['status']=='completed_five_arm_fixed_prefix_reused_branches'
        assert not target.exists(),'Evaluation attempt already exists without final result; inspect before retry'
        print(json.dumps(dict(start_evaluation=[layout,paid])),flush=True)
        with (OUT/f'{layout}_{paid}.log').open('w')as log:
            subprocess.run([sys.executable,str(HERE/'evaluate_particle_replay.py'),'--manifest',str(source/'manifest.json'),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(HERE/'summarize_particle_replay.py')],check=True)
    if not pending:break
    print(json.dumps(dict(waiting=pending)),flush=True);time.sleep(20)
print('all six fixed-replay states independently evaluated',flush=True)
