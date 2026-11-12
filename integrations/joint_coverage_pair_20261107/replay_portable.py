#!/usr/bin/env python3
"""Stage unchanged historical runner around the previously published vendor code."""
import argparse, pathlib, shutil, subprocess, sys, tempfile
p=argparse.ArgumentParser()
p.add_argument('--arm',choices=('C','S'),required=True)
p.add_argument('--vendor-code',type=pathlib.Path,required=True,help='existing integrations/exploration_resume_20261107/code directory')
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args(); root=pathlib.Path(__file__).resolve().parent
vendor=a.vendor_code.resolve(strict=True); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
if not (vendor/'nso_bridge/joint_engine.py').is_file(): raise ValueError('vendor code missing nso_bridge/joint_engine.py')
with tempfile.TemporaryDirectory(prefix='nso_pair_replay_') as d:
 base=pathlib.Path(d); arm=base/'joint_coverage_pair_20261107/publication'/a.arm; arm.mkdir(parents=True)
 for item in (root/a.arm).iterdir():
  if item.name=='replay.py': shutil.copyfile(item,arm/item.name)
  elif item.name!='replay_output': (arm/item.name).symlink_to(item.resolve(),target_is_directory=item.is_dir())
 code=base/'exploration_resume_20261107/publication/code'; code.parent.mkdir(parents=True);code.symlink_to(vendor,target_is_directory=True)
 raise SystemExit(subprocess.call([sys.executable,str(arm/'replay.py'),'--output',str(out)]))
