#!/usr/bin/env python3
"""Run the retained mapper audit with only path constants redirected."""
import argparse,pathlib,subprocess,sys,tempfile
p=argparse.ArgumentParser();p.add_argument('--vendor-code',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent;vendor=a.vendor_code.resolve(strict=True);output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
for part in ('nso_bridge/joint_engine.py','joint_vendor/nso/mapping3d.py','joint_vendor/nso/observed_runtime_mapper_v10.py'):
 if not (vendor/part).is_file(): raise ValueError('missing vendor component '+part)
source=(here/'mapper_only.py').read_text();old_w="Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')";old_p="Path('/home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/publication/S')";old_out="Path(__file__).resolve().parent"
for needle in (old_w,old_p,old_out):
 if source.count(needle)!=1:raise ValueError('unexpected original audit structure')
with tempfile.TemporaryDirectory(prefix='nso_mapper_audit_') as d:
 stage=pathlib.Path(d);w=stage/'workspace/nso_bridge';w.mkdir(parents=True)
 for part in ('nso_bridge','joint_vendor'): (w/part).symlink_to(vendor/part,target_is_directory=True)
 source=source.replace(old_w,'Path('+repr(str(w.parent))+')').replace(old_p,'Path('+repr(str(here.parent/'S'))+')').replace(old_out,'Path('+repr(str(output))+')')
 runner=stage/'mapper_only_relocated.py';runner.write_text(source)
 raise SystemExit(subprocess.call([sys.executable,str(runner)]))
