#!/usr/bin/env python3
"""Observation-only B9 particle inference subprocess. Never opens evaluator truth."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PARENT=Path('/home/ouc/nso_robot/analyses/dual_template_uncertainty_20261120')
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
INSTANCE=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge/nso_bridge/instance_support.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest_path=args.observations.resolve();output=args.output.resolve()
    assert manifest_path.name=='manifest.json'
    assert (ROOT/'observations') in manifest_path.parents or (ROOT/'counterfactual') in manifest_path.parents
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    forbidden=(ROOT/'evaluator_private',ROOT/'generator',PARENT/'evaluator_private',PARENT/'generator')
    from isolated_io import install_boundary
    blocked,read_files=install_boundary(forbidden)
    try:
        sys.path.insert(0,str(SOURCE))
        import numpy as np
        from utils.rgbd_contract import RGBDFrame
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.cpu_sensor_contract_v10 import json_value
        index=load('B9_acquired_instance_support',INSTANCE)
        inference=load('B9_distribution_inference',PARENT/'method/infer.py')
        manifest=json.loads(manifest_path.read_text())
        assert manifest['schema']=='B9_sanitized_observation_sequence_v1'
        assert not {'truth','construction','split','object_id','reference_geometry_id'}&set(manifest)
        config=SimpleNamespace(**manifest['sensor_config'])
        mapper=ObservedRuntimeMapperV10(tuple(manifest['mapper_shape']),config,config.truncation_m)
        mapper.instance_support_index=index.InstanceSupportIndex();mapper.asset_grouping_profile='measured_instance_support_v1'
        observations=[]
        for row in manifest['frames']:
            fp=Path(row['frame']['path']);op=Path(row['owner']['path'])
            assert fp.parent==manifest_path.parent and op.parent==manifest_path.parent
            assert sha(fp)==row['frame']['sha256'] and sha(op)==row['owner']['sha256']
            frame=RGBDFrame.load(fp)
            with np.load(op,allow_pickle=False)as data:
                assert data.files==['visible_owner'];owner=data['visible_owner'].copy()
            assert owner.shape==frame.depth_m.shape and not owner[frame.depth_m<=0].any()
            event=mapper.instance_support_index.observe(frame,owner,row['frame_id'])
            mapper.update(frame,scan=None)
            observations.append(dict(frame=row['frame'],owner=row['owner'],association=json_value(event)))
        def snapshot():
            values=list(mapper.quality.values());keys=np.asarray(list(mapper.quality),dtype=np.int64).reshape(-1,3)
            arrays={key:np.ascontiguousarray(np.asarray([row[key]for row in values]))for key in values[0]}if values else {}
            arrays['voxel_keys']=keys;arrays['support_track']=mapper.instance_support_index.memberships_for_keys(keys)
            return dict(quality={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in arrays.items()},
                frames=int(mapper.frames),keyframes=[hashlib.sha256(f.world_from_camera.tobytes()+f.intrinsic.tobytes()+f.depth_m.tobytes()).hexdigest()for f in mapper.keyframes])
        before=snapshot()
        prediction=inference.infer_depth_distribution(mapper,neutral_model=None,frozen_scale_config={'sigma_m':.01})
        assert snapshot()==before,'Inference changed acquired evidence'
        prediction=json_value(prediction)
        write(output/'prediction.json',prediction)
        write(output/'observed_input_audit.json',dict(observation_manifest=resource(manifest_path),
            acquired=snapshot(),observations=observations,ideal_instance_mask=True,neutral_structural_prior=True,
            no_truth_or_split_in_inference_api=True,blocked_private_access=blocked,model_input_source='acquired RGB-D and measured instance support only'))
        assert not blocked
        code_files=[Path(path)for path in read_files if path.endswith('.py') and Path(path).exists() and str(Path(path)).startswith('/home/ouc/nso_robot/')]
        write(output/'result.json',dict(status='completed_observation_only_particle_bank',
            prediction=resource(output/'prediction.json'),observed_inputs=resource(output/'observed_input_audit.json'),
            observation_manifest=resource(manifest_path),elapsed_s=time.monotonic()-start,
            code_files_sha256={str(p):sha(p)for p in sorted(code_files)},
            generator_or_truth_files_opened=False,source_actor=str(Path(__file__)),source_actor_sha256=sha(__file__),
            simulator_or_policy_actions=0,temperature_selected=False,sigma_m=.01,existing_B8_model_unchanged=True))
        print(json.dumps({'status':'completed','output':str(output),'elapsed_s':time.monotonic()-start}),flush=True)
    except BaseException as error:
        import traceback
        write(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),blocked_private_access=blocked,
            observation_manifest=str(manifest_path),elapsed_s=time.monotonic()-start,simulator_or_policy_actions=0))
        raise


if __name__=='__main__':main()
