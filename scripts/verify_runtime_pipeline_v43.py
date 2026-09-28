#!/usr/bin/env python3
"""Seal V43 static navigation and finite observed-packet interface checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from env.development_sensor_v41 import runtime_counts_v41
from nso.analytic_fixture_v42 import analytic_forward_sequence_v42,ANALYTIC_MARKER_COLOR_V42
from nso.controller_v43 import ANSControllerV43
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41,PrimitiveStateV41
from nso.public_navigation_v43 import load_public_navigation_bundle_v43
from run_development_v43 import source_names_v43,run_development_v43


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def old_seals():
    rows=[]
    for name in ('v40_p0_20260920','v40_p1_static_release_20260920','v41_interface_release_20260920','v42_interface_release_20260921'):
        path=ROOT/'audit_results'/name/'manifest.json'; manifest=json.loads(path.read_text())
        for file,entry in manifest['files'].items():
            assert sha(ROOT/file)==(entry['sha256'] if isinstance(entry,dict) else entry),file
        rows.append(dict(manifest=str(path.relative_to(ROOT)),sha256=sha(path),files_checked=len(manifest['files'])))
    return rows


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    names=source_names_v43()+['nso/analytic_fixture_v42.py','scripts/verify_runtime_pipeline_v43.py',
                             'scripts/compile_public_navigation_v43.py']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_*v43.py'))]
    sources={name:sha(ROOT/name) for name in sorted(set(names))}
    write(output/'started.json',dict(scope='static assets, fixed analytic packets, finite driver doubles; no live study rollout',
        source_sha256=sources,maximum_output_bytes=10*1024**2,actual_development_worlds_authorized=0))
    for name,expected in sources.items():
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        assert sha(dest)==expected
    before=old_seals()
    command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_*v43.py','-v']
    start=time.monotonic()
    completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=300,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    log=completed.stdout+completed.stderr;(output/'unittest.txt').write_text(log)
    count=re.findall(r'Ran (\d+) tests?',log)
    tests=dict(exit_code=completed.returncode,tests_run=int(count[-1]) if count else None,
               elapsed_seconds=time.monotonic()-start,log_sha256=sha(output/'unittest.txt'),command=command)
    write(output/'test_result.json',tests)
    if completed.returncode: raise RuntimeError('V43 tests failed; preserve this output and fix before a new revision')

    # Six real static navigation assets are loaded through the PUBLIC runtime
    # API. It never opens their renderer_private/evaluation_private contents.
    navigation=[]
    for family in 'ABCDEF':
        parent=f'DEV_{family}_00';directory=ROOT/'audit_results/v43_public_navigation_r1_20260921'/parent
        bundle=load_public_navigation_bundle_v43(directory,ROOT/'audit_results/v40_p1_development_geometry_20260920'/parent)
        graph=bundle['graph']
        navigation.append(dict(parent_id=parent,coarse_nodes=len(graph.original_nodes),expanded_positions=len(graph.positions),
            coarse_edges=len(bundle['graph_spec']['edges']),graph_sha256=graph.input_sha256,
            bundle_manifest_sha256=sha(directory/'manifest.json'),materialized_public_spec=bundle['public_spec']))
    write(output/'public_navigation_loads.json',navigation)

    # All packets are constructed before either policy call. The stored
    # second packet describes one fixed .25 m move. A matching command is an
    # interface-consistency check, not an executed simulated trajectory.
    packets=analytic_forward_sequence_v42()
    graph_spec=dict(schema_version='v41.public_navigation.v1',source_kind='provided_navigation_prior',
                    nodes={'home':[.75,.75],'advance':[1.,.75]},edges=[['home','advance']])
    controllers={mode:ANSControllerV43(PublicPrimitiveGraphV41(graph_spec),home=PrimitiveStateV41('home',0),budget=32,
        palette={'cabinet':ANALYTIC_MARKER_COLOR_V42},structure_names=('planar','recessed','louvered','open_frame'),
        class_structure_prior={'cabinet':[.4,.3,.2,.1]},mode=mode) for mode in ('G','S')}
    mapper=ObservedMapperV42(shape=(30,30),origin_xy_m=(-.5,-.5))
    all_actions={mode:[] for mode in controllers};scored=[]
    for packet in packets:
        np.savez_compressed(output/f'analytic_packet_{packet.paid_step:03d}.npz',
            **{key:getattr(packet,key) for key in packet.__dataclass_fields__})
        mapping=mapper.update(packet)
        states={};decisions={}
        for mode,controller in controllers.items():
            accepted=controller.accept(packet,mapper)
            decision=controller.choose();state=controller.snapshot()
            states[mode]=state;decisions[mode]=decision;all_actions[mode].append(decision['action'])
            write(output/f'{mode}_step_{packet.paid_step:03d}.json',dict(mapping=mapping,accepted=accepted,
                decision=decision,state=state,candidate_action_physically_executed=False))
        assert states['G']['geometry']==states['S']['geometry']
        assert decisions['G']['candidate_pool']==decisions['S']['candidate_pool']
        assert decisions['G']['instance_candidate_allocations']==decisions['S']['instance_candidate_allocations']
        if packet.paid_step==0:
            assert all(decision['action']=='forward' for decision in decisions.values())
        scored.append(dict(paid_step=packet.paid_step,packet_sha256=packet.sha256(),
            semantic_active={mode:state['observed_instances']['instances'][0]['semantic_conditioning_used']
                             for mode,state in states.items()},
            actions={mode:decision['action'] for mode,decision in decisions.items()},
            targets={mode:decision['routing'].get('target') for mode,decision in decisions.items()}))
    assert scored[1]['semantic_active']=={'G':False,'S':True}
    np.savez_compressed(output/'analytic_tsdf_mesh.npz',**mapper.mesh_arrays())
    write(output/'mapper_snapshot.json',mapper.snapshot())
    # Test the actual entry point's negative resource path. preflight_only is
    # true so a future large disk still cannot create World from this script.
    gate=run_development_v43('R3_A_G',output_root=output/'not_started_episodes',preflight_only=True)
    write(output/'actual_entrypoint_preflight.json',gate)
    assert not gate['start_slot_reserved'] and runtime_counts_v41()['worlds_created']==0
    assert not (output/'not_started_episodes').exists()
    assert before==old_seals()
    assert sources=={name:sha(ROOT/name) for name in sources},'source changed during final verification'
    import scipy,open3d
    result=dict(status='passed',phase='V43_static_and_analytic_interfaces',contract_tests=tests['tests_run'],
        static_public_navigation_parents=len(navigation),static_coarse_nodes=sum(r['coarse_nodes'] for r in navigation),
        static_coarse_edges=sum(r['coarse_edges'] for r in navigation),
        main_analytic_predeclared_packets=2,main_analytic_TSDF_integrations=mapper.snapshot()['tsdf_integration_count'],
        two_packet_controller_actions=all_actions,controller_receipts=scored,
        s_g_geometry_candidate_pool_and_allocations_equal=True,semantic_active_after_second_fixed_packet=True,
        four_module_cpu_controller_composed=True,full_autonomous_development_rollout_validated=False,
        driver_verified_with_finite_packet_doubles=True,actual_development_worlds=0,
        actual_development_sensor_trajectories=0,actual_development_planner_rollouts=0,
        forecast_selected_actions_physically_executed=0,semantic_performance_claim_added=False,
        physical_batch_storage_ready=gate['resource']['passed'],runtime_counts=runtime_counts_v41(),
        old_seals=before,source_sha256=sources,source_unchanged_during_verification=True,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,open3d=open3d.__version__),
        limitations=['provided coarse navigation prior','artificial marker frontend','greedy bounded candidate approximation',
            'uncalibrated area proxies and default zero uncertainty penalty','no actual driver World run',
            'no cross-scene quality or semantic efficacy conclusion'])
    write(output/'result.json',result)
    write(output/'artifact_sha256.json',{str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(output.rglob('*')) if p.is_file()})
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    assert total<=10*1024**2,'declared analytic output cap exceeded; retain for audit'
    print(json.dumps({key:result[key] for key in ('status','contract_tests','static_public_navigation_parents',
        'two_packet_controller_actions','actual_development_worlds','physical_batch_storage_ready')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'audit_results/v43_runtime_pipeline_20260921')
    run(parser.parse_args().output.resolve())
