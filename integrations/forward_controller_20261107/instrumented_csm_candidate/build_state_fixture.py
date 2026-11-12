"""Offline helper only: access installed class state, never install/replace it."""
import hashlib,json,os,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ['FIXTURE_OUTPUT_DIR']).resolve()
if OUT==ROOT or ROOT in OUT.parents: raise ValueError('output must be separate')
OUT.mkdir(parents=True,exist_ok=True)
flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','roscpp','tf','pcl_ros','gsl'],text=True))
cmd=['g++','-std=c++14','-O2','-fno-access-control',str(ROOT/'state_fixture.cpp'),'-o',str(OUT/'state_fixture'),'-L/opt/ros/noetic/lib','-llaser_scan_matcher']+flags
with (OUT/'state_fixture_build.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(command=cmd,exit_code=result.returncode,scope='private-state access permitted only for offline fixture; links installed matcher conversion/getPrediction/sm_icp; no installed source or binary changed')
if not result.returncode:receipt['binary_sha256']=hashlib.sha256((OUT/'state_fixture').read_bytes()).hexdigest()
(OUT/'state_fixture_build_receipt.json').write_text(json.dumps(receipt,indent=2))
raise SystemExit(result.returncode)
