#!/usr/bin/env python3
"""Compile the manuscript with pinned local Tectonic and bounded disk use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--allow-downloads',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'tmp/v38-tex/manuscript')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    task=ROOT/'tmp/v38-tex'; binary=task/'bin/tectonic'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()=='a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da'
    env=dict(os.environ)
    env.update(TECTONIC_CACHE_DIR=str(task/'cache'),XDG_CACHE_HOME=str(task/'cache'),XDG_CONFIG_HOME=str(task/'config'))
    cmd=[str(binary),'--keep-logs','--outdir',str(a.output)]
    if not a.allow_downloads: cmd.append('--only-cached')
    cmd.append(str(ROOT/'Semantic_Enhanced_Active_SLAM_Paper.tex'))
    minimum=shutil.disk_usage(ROOT).free
    assert minimum>76*1024**2,'Insufficient guarded compilation space'
    started=time.monotonic(); reason=None
    with (a.output/'stdout.txt').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            minimum=min(minimum,shutil.disk_usage(ROOT).free)
            if minimum<72*1024**2: reason='72 MiB compilation stop guard preserves 64 MiB reserve'
            if time.monotonic()-started>240: reason='240 second compile limit'
            if reason:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                break
            time.sleep(.05)
    receipt={'command':cmd,'exit_code':proc.returncode,'stop_reason':reason,
             'elapsed_s':time.monotonic()-started,'minimum_free_bytes':minimum,
             'manuscript_sha256':hashlib.sha256((ROOT/'Semantic_Enhanced_Active_SLAM_Paper.tex').read_bytes()).hexdigest(),
             'downloads_allowed':a.allow_downloads}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2)); print((a.output/'stdout.txt').read_text()[-10000:])
    if proc.returncode: raise SystemExit(proc.returncode)

if __name__=='__main__': main()
