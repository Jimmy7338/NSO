#!/usr/bin/env python3
"""Prepare pinned static references, retaining all failures and measured cost.

This command does not evaluate any episode or construct a World. Endpoint
evaluation is exposed by nso.semantic_scene_evaluation for a future inspected
episode adapter; the currently frozen three-slot runner is not modified.
"""
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.offline_evaluation_v44 import _write_json,sha256
from nso.semantic_scene_assets import checked_manifest
from nso.semantic_scene_evaluation import (
    _source_hashes,load_semantic_scene_reference,prepare_semantic_scene_reference,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--asset-manifest-sha256',required=True)
    parser.add_argument('--asset-id',action='append',dest='asset_ids')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-seconds',type=float,default=600.)
    parser.add_argument('--maximum-seconds-per-asset',type=float,default=60.)
    args=parser.parse_args()
    if not 0<args.maximum_seconds<=1800 or not 0<args.maximum_seconds_per_asset<=300:
        parser.error('finite total limit (0,1800] and per-asset limit (0,300] required')
    output=args.output.resolve()
    if output==args.asset_root.resolve() or output.is_relative_to(args.asset_root.resolve()):
        parser.error('output must be outside immutable asset tree')
    output.mkdir(parents=True,exist_ok=False)
    began,cpu_began=time.perf_counter(),time.process_time()
    sources=_source_hashes()
    started=dict(schema='semantic_scene.reference_preparation_attempt.v1',
        asset_root=str(args.asset_root.resolve()),asset_manifest_sha256=args.asset_manifest_sha256,
        requested_asset_ids=args.asset_ids,source_sha256=sources,cli_source_sha256=sha256(Path(__file__)),
        maximum_seconds=args.maximum_seconds,maximum_seconds_per_asset=args.maximum_seconds_per_asset,
        worlds_created=0,new_sensor_packets=0,policy_trajectories_created=0,quality_evaluations=0,
        automatic_retry=False,primary_matrix_started=False)
    _write_json(output/'started.json',started)
    rows=[];setup_error=None
    def timeout(signum,frame):
        raise TimeoutError('predeclared static reference time limit exceeded; retain failure')
    previous=signal.signal(signal.SIGALRM,timeout)
    try:
        manifest=checked_manifest(args.asset_root,args.asset_manifest_sha256)
        selected=args.asset_ids or sorted(manifest['assets'])
        if len(set(selected))!=len(selected) or not set(selected)<=set(manifest['assets']):
            raise ValueError('distinct manifest-declared asset IDs required')
        for identity in selected:
            wall,cpu=time.perf_counter(),time.process_time()
            row=dict(asset_id=identity,status='started',automatic_retry=False)
            try:
                remaining=args.maximum_seconds-(wall-began)
                if remaining<=0:
                    raise TimeoutError('total static preparation limit reached before this asset')
                if sources!=_source_hashes():
                    raise ValueError('reference sources changed within batch')
                signal.setitimer(signal.ITIMER_REAL,min(args.maximum_seconds_per_asset,remaining))
                result=prepare_semantic_scene_reference(identity,output/identity,
                    asset_root=args.asset_root,asset_manifest_sha256=args.asset_manifest_sha256)
                # Includes a sealed readback before recording success.
                load_semantic_scene_reference(output/identity,manifest_sha256=result['manifest_sha256'],
                    asset_id=identity,asset_root=args.asset_root,asset_manifest_sha256=args.asset_manifest_sha256)
                row.update(status='prepared_and_readback_verified',reference=result)
            except Exception as exc:
                row.update(status='failed_retained',error=dict(type=type(exc).__name__,message=str(exc)))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                row.update(elapsed_s=time.perf_counter()-wall,process_cpu_s=time.process_time()-cpu,
                    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rows.append(row)
            _write_json(output/(identity+'.attempt.json'),row)
            print(json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
    except Exception as exc:
        setup_error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    result=dict(started,status='complete' if rows and setup_error is None
        and all(row['status']=='prepared_and_readback_verified' for row in rows) else 'failures_retained',
        setup_error=setup_error,attempts=rows,attempted_assets=len(rows),
        successful_references=sum(row['status']=='prepared_and_readback_verified' for row in rows),
        failed_assets=[row['asset_id'] for row in rows if row['status']!='prepared_and_readback_verified'],
        elapsed_s=time.perf_counter()-began,process_cpu_s=time.process_time()-cpu_began,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifact_bytes=sum(path.stat().st_size for path in output.rglob('*') if path.is_file()),
        eligibility='Static reference readiness only; no controller or semantic performance evidence.')
    _write_json(output/'result.json',result)
    print(json.dumps(dict(status=result['status'],attempted=result['attempted_assets'],
        successful=result['successful_references'],elapsed_s=result['elapsed_s'],output=str(output))),flush=True)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
