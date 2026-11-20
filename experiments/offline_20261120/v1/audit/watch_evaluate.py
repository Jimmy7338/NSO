#!/usr/bin/env python3
"""Only evaluate completed root-owned replay jobs, sequentially."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}


def main():
    done=set()
    while len(done)<6:
        for layout,states in STATES.items():
            for paid in states:
                key=(layout,paid)
                if key in done:continue
                attempt=ROOT/'replay'/f'{layout}_{paid}_attempt_001'
                if not (attempt/'result.json').exists():continue
                if json.loads((attempt/'result.json').read_text())['status']!='completed_staged_fixed_prefix_all_candidate_branches':
                    raise ValueError(f'Unexpected incomplete replay result: {attempt}')
                out=HERE/f'{layout}_{paid}'
                if not (out/'result.json').exists():
                    log=HERE/f'evaluate_{layout}_{paid}.log'
                    with log.open('w') as handle:
                        result=subprocess.run([sys.executable,str(HERE/'evaluate_stage.py'),'--manifest',str(attempt/'manifest.json'),'--output',str(out)],stdout=handle,stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'Evaluation failed; preserved log {log}')
                done.add(key)
                subprocess.run([sys.executable,str(HERE/'summarize_stages.py')],check=True)
                print(json.dumps({'evaluated':key,'completed_states':len(done)}),flush=True)
        if len(done)<6:time.sleep(3)


if __name__=='__main__':main()
