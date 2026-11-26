#!/usr/bin/env python3
"""Wait for declared prediction receipts and evaluate them; no policy launch."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
pending=[(l,s)for l,ss in {'shelf_left':(24,54,109),'shelf_right':(24,53,110)}.items()for s in ss]
while pending:
    for layout,paid in pending.copy():
        attempt=ROOT/'replay'/f'{layout}_{paid}_attempt_001'
        path=attempt/'result.json'
        if not path.exists():
            if(attempt/'failure.json').exists():raise RuntimeError(f'Prediction failed: {attempt}')
            continue
        assert json.loads(path.read_text())['status']=='completed_parts_fixed_prefix_reused_branches'
        output=HERE/f'{layout}_{paid}'
        with (HERE/f'evaluate_{layout}_{paid}.log').open('w')as log:
            result=subprocess.run([sys.executable,str(HERE/'evaluate_parts.py'),'--manifest',str(attempt/'manifest.json'),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Evaluation failed: {output}; log preserved')
        pending.remove((layout,paid))
        if(HERE/'summarize_parts.py').exists():subprocess.run([sys.executable,str(HERE/'summarize_parts.py')],check=True)
        print(json.dumps(dict(evaluated=[layout,paid],remaining=pending)),flush=True)
    if pending:time.sleep(3)
