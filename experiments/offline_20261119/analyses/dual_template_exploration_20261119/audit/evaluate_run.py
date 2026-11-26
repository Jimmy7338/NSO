#!/usr/bin/env python3
"""Evaluate only an existing measured TSDF export; never import an actor."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import open3d as o3d
from surface_metrics import evaluate_surface,OLD_EVALUATION

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        manifest=json.loads((args.attempt/'artifact_manifest.json').read_text())
        assert manifest['status']=='completed_execution'
        for name,h in manifest['files_sha256'].items():assert sha(args.attempt/name)==h,name
        receipt=json.loads((args.reference/'RECEIPT.json').read_text())
        for name,key in [('fixed_reference.npz','reference_npz_sha256'),('source_mesh.ply','source_mesh_sha256'),('PROTOCOL.json','protocol_sha256')]:
            assert sha(args.reference/name)==receipt[key]
        for path,h in receipt['source_inputs_sha256'].items():assert sha(path)==h,path
        reference=dict(np.load(args.reference/'fixed_reference.npz',allow_pickle=False))
        prediction=o3d.io.read_triangle_mesh(str(args.attempt/'final_tsdf_mesh.ply'))
        truth=o3d.io.read_triangle_mesh(str(args.reference/'source_mesh.ply'))
        result={'status':'completed_independent_surface_evaluation','source_attempt':str(args.attempt.resolve()),
            'layout':receipt['layout'],'reference_sha256':receipt['reference_npz_sha256'],
            'reference_protocol_sha256':receipt['protocol_sha256'],'metrics':evaluate_surface(prediction,truth,reference),
            'native_TSDF_mesh_sha256':sha(args.attempt/'final_tsdf_mesh.ply'),
            'all_run_artifact_hashes_verified':True,'source_artifact_manifest_sha256':sha(args.attempt/'artifact_manifest.json'),
            'evaluator_source_sha256':{str(p):sha(p)for p in [Path(__file__),Path(__file__).with_name('surface_metrics.py'),OLD_EVALUATION/'surface_reference.py']},
            'no_actor_executed':True,'no_registration':True,'predicted_template_geometry_used':False,
            'target_metric':'Completeness only; source118 actual cabinet replaces historical helper target slot by explicit dictionary rename, not altered distances.',
            'elapsed_s':time.monotonic()-start}
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'output':str(args.output),'F1_5cm':result['metrics']['f1_05cm'],'elapsed_s':result['elapsed_s']}),flush=True)
    except BaseException as exc:
        (args.output/'failure.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise

if __name__=='__main__':main()
