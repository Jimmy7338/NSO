#!/usr/bin/env python3
"""V44 lossless step encoding on the same gated, five-slot V43 development batch."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.development_sensor_v41 import create_development_sensor, storage_report_v41, runtime_counts_v41
from nso.evidence_writer_v44 import CompressedStepWriterV44
from nso.episode_driver_v43 import (DevelopmentStartLedgerV43,
                                    execute_episode_v43, file_sha256)


def source_names_v44():
    names = ['nso/instance_belief_v40.py','nso/observed_instances_v41.py','nso/observed_residual_v41.py',
        'nso/primitive_navigation_v41.py','nso/development_geometry_v40.py','nso/surface_evaluation_v40.py',
        'nso/observed_mapper_v42.py','nso/view_quality_v42.py','env/development_sensor_v41.py',
        'nso/scene_contract_v40.py','utils/rgbd_contract.py',
        'configs/virtual3d/v40_scene_protocol_20260920.json','configs/virtual3d/v43_runtime_protocol_20260921.json',
        'scripts/run_development_v43.py','scripts/run_development_v44.py','nso/evidence_writer_v44.py']
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'nso').glob('*v43.py')))
    return sorted(set(names))


def run_development_v44(run_id, *, output_root, preflight_only=False):
    protocol_path=ROOT/'configs/virtual3d/v43_runtime_protocol_20260921.json'
    protocol=json.loads(protocol_path.read_text())
    if run_id not in protocol['slots']:
        raise ValueError('run must name one of the five predeclared development slots')
    output=Path(output_root).resolve()/run_id
    # No output, ledger reservation, navigation read, sensor, or mapper is
    # created before this mandatory persistent-space gate.
    resource=storage_report_v41(output, protocol['expected_batch_peak_bytes'])
    if not resource['passed']:
        return dict(status='blocked_before_world_creation',run_id=run_id,resource=resource,
                    start_slot_reserved=False,asset_read=False,runtime=runtime_counts_v41())
    from nso.public_navigation_v43 import load_public_navigation_bundle_v43
    from nso.controller_v43 import ANSControllerV43
    from nso.diagnostic_policy_v43 import DiagnosticPolicyV43
    from nso.observed_mapper_v42 import ObservedMapperV42
    slot=protocol['slots'][run_id]
    asset=ROOT/protocol['asset_root']/slot['asset_id']
    nav=ROOT/protocol['navigation_root']/slot['asset_id']
    bundle=load_public_navigation_bundle_v43(nav,asset)
    public=bundle['public_spec']; workspace=bundle['workspace']; graph=bundle['graph']; home=bundle['home_state']
    budget=public['task']['max_actions']
    if budget > 160:
        raise ValueError('V43 bounded driver supports at most 160 paid actions')
    if slot['mode']=='diagnostic':
        controller=DiagnosticPolicyV43(graph,home=home,budget=budget,actions=protocol['diagnostic_actions'])
    else:
        prior=public['structure_prior']
        controller=ANSControllerV43(graph,home=home,budget=budget,palette=workspace['marker_palette'],
            structure_names=prior['abstract_structures'],class_structure_prior=prior['probability_by_category'],
            mode=slot['mode'],**protocol['controller'])
    # Dependency probing does not integrate an observation or instantiate World.
    import open3d as o3d
    if preflight_only:
        return dict(status='ready_without_world_creation',run_id=run_id,resource=resource,
                    start_slot_reserved=False,asset_read='public_only',backend='Open3D '+o3d.__version__,
                    public_graph_sha256=graph.input_sha256,runtime=runtime_counts_v41())
    if output.exists():
        raise FileExistsError('new episode directory required; retained attempts cannot be overwritten')
    sources={name:file_sha256(ROOT/name) for name in source_names_v44()}
    ledger=DevelopmentStartLedgerV43(ROOT/protocol['ledger_relative_path'],maximum_slots=protocol['maximum_new_world_slots'])
    metadata=dict(run_id=run_id,slot=slot,output=str(output),protocol_sha256=file_sha256(protocol_path),
                  public_graph_sha256=graph.input_sha256,source_sha256=sources,
                  evidence_encoding='gzip-json-v1 steps only; metadata and terminal JSON plain',
                  same_v43_start_ledger=True)
    # Reserve durably before any actual World creation attempt. A failed slot
    # remains recorded, including a resource-race failure in the sensor factory.
    ledger.reserve(run_id,metadata)
    output.mkdir(parents=True,exist_ok=False)
    writer=CompressedStepWriterV44(output,maximum_bytes=protocol['maximum_episode_bytes'],
        maximum_file_bytes=protocol['maximum_file_bytes'],terminal_reserve_bytes=protocol['terminal_record_reserve_bytes'])
    sensor=None; world_created=False; before=runtime_counts_v41()
    try:
        writer.json('started.json',dict(metadata,resource=resource,
            task_kind='fixed_motion_diagnostic' if slot['mode']=='diagnostic' else 'autonomous_controller',
            exact_pose_model=True,known_coarse_navigation_prior=True))
        for name,expected in sources.items():
            writer._write('source/'+name,(ROOT/name).read_bytes())
            if file_sha256(output/'source'/name)!=expected:
                raise RuntimeError('source changed during snapshot creation')
        writer.json('public_graph.json',bundle['graph_spec'])
        writer.json('public_spec.json',public); writer.json('public_workspace.json',workspace)
        bounds=workspace['bounds_xy_m']; origin=bounds[0]; resolution=protocol['mapper']['resolution_m']
        shape=[math.ceil((bounds[1][1]-origin[1])/resolution),math.ceil((bounds[1][0]-origin[0])/resolution)]
        mapper=ObservedMapperV42(shape=shape,origin_xy_m=origin,**protocol['mapper'])
        sensor=create_development_sensor(asset,public,episode_id=run_id,noise_seed=slot['noise_seed'],
            persistent_output_root=output,expected_batch_peak_bytes=protocol['expected_batch_peak_bytes'])
        world_created=True
        result=execute_episode_v43(sensor,controller,mapper,writer,budget=budget,
                                  maximum_elapsed_s=protocol['maximum_elapsed_s'])
        changed=[name for name,value in sources.items() if file_sha256(ROOT/name)!=value]
        if changed:
            writer.json('integrity_failure.json',dict(status='source_changed_during_episode',files=changed),terminal=True)
            final_status='source_changed_during_episode'; final_path=output/'integrity_failure.json'
        else:
            final_status=result['status']; final_path=output/'result.json'
        writer.json('encoding.json',dict(schema='v44.step_encoding.v1',steps=writer.step_encoding),terminal=True)
        writer.json('runtime.json',dict(before=before,after=runtime_counts_v41(),world_created=world_created,
            evaluation_executed=False,source_unchanged=not changed),terminal=True)
        writer.json('artifact_manifest.json',dict(scope='complete saved prediction; not an efficacy evaluation',
            files=writer.files,source_sha256=sources),terminal=True)
        ledger.finish(run_id,status=final_status,world_created=world_created,result_sha256=file_sha256(final_path))
        return dict(status=final_status,run_id=run_id,output=str(output),world_created=world_created,
                    result_sha256=file_sha256(final_path),evaluation_executed=False)
    except Exception as exc:
        close_error=None
        if sensor is not None:
            try:
                sensor.close()
            except Exception as close_exc:
                close_error=dict(type=type(close_exc).__name__,message=str(close_exc))
        failure=dict(status='development_attempt_failed',type=type(exc).__name__,message=str(exc),
                     run_id=run_id,world_created=world_created,runtime=runtime_counts_v41(),
                     sensor_close_error=close_error,automatic_retry=False)
        writer.json('attempt_failure.json',failure,terminal=True)
        ledger.finish(run_id,status=failure['status'],world_created=world_created,
                      result_sha256=file_sha256(output/'attempt_failure.json'))
        return failure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'audit_results/v43_development_episodes_20260921')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    result=run_development_v44(args.run_id,output_root=args.output_root,preflight_only=args.preflight_only)
    print(json.dumps(result,ensure_ascii=False))
    sys.exit(2 if result['status']=='blocked_before_world_creation' else
             0 if result['status'] in ('ready_without_world_creation','controller_stop') else 1)
