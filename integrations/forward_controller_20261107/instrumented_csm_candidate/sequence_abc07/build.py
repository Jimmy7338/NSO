"""Compile the offline helper against existing installed ROS/CSM libraries."""
import hashlib,json,os,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ['SEQUENCE_OUTPUT_DIR']).resolve()
if OUT==ROOT or ROOT in OUT.parents:raise ValueError('output must be separate')
OUT.mkdir(parents=True,exist_ok=True)
flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','roscpp','tf','pcl_ros','gsl'],text=True))
cmd=['g++','-std=c++14','-O2','-fno-access-control',str(ROOT/'native_sequence.cpp'),'-o',str(OUT/'native_sequence'),'-L/opt/ros/noetic/lib','-llaser_scan_matcher']+flags
with (OUT/'build.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(command=cmd,exit_code=r.returncode,packages_installed=False)
if not r.returncode:receipt['binary_sha256']=hashlib.sha256((OUT/'native_sequence').read_bytes()).hexdigest()
(OUT/'build_receipt.json').write_text(json.dumps(receipt,indent=2));raise SystemExit(r.returncode)
