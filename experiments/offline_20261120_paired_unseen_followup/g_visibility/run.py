"""Four neutral-prior visibility runs on immutable B11 worlds; no robot access."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
B11=HERE.parents[1]/'paired_unseen_gain_20261120'
EXEC=B11/'execution'
LAYOUTS=('chain_a','cycle_a','chain_b','cycle_b')
sys.path.insert(0,str(EXEC))
import runtime_binding as R
_installed=False

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);R.write(path,data)
def protocol():return read(HERE/'protocol.json')
def prepared(layout):return EXEC/'layouts'/layout/'prepared'
def baseline(layout):return EXEC/'layouts'/layout/'runs/S_visibility/attempt_001'
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def frozen_paths():
    paths=[HERE/'run.py',HERE/'supervisor.py',HERE/'protocol.json',EXEC/'freeze.json']
    paths += [Path(p)for p in read(EXEC/'freeze.json')['files_sha256']]
    for layout in LAYOUTS:
        paths += list(prepared(layout).glob('*'))
        paths += [baseline(layout)/n for n in ('artifact_manifest.json','result.json','template_audit.json','choices.json','effective_configuration.json')]
    return sorted(set(p.resolve()for p in paths if p.is_file()))

def freeze():
    assert not (HERE/'freeze.json').exists()
    assert protocol()['status']=='frozen_before_four_G_visibility_runs'
    # Verify the parent evidence before sealing the extension.
    for p,h in read(EXEC/'freeze.json')['files_sha256'].items():assert R.sha(p)==h,p
    write(HERE/'freeze.json',dict(status='frozen_four_G_visibility_extension',
        created_unix_s=time.time(),parent_execution_freeze=R.resource(EXEC/'freeze.json'),
        files_sha256={str(p):R.sha(p)for p in frozen_paths()},actor_actions_before_freeze=0))

def verify(layout=None):
    frozen=read(HERE/'freeze.json')
    for p,h in frozen['files_sha256'].items():assert R.sha(p)==h,p
    if layout is None:return frozen
    assert layout in LAYOUTS
    f=read(prepared(layout)/'freeze.json')
    assert f['execution_freeze_sha256']==R.sha(EXEC/'freeze.json')
    for n,h in f['prepared_files_sha256'].items():assert R.sha(prepared(layout)/n)==h,n
    return read(prepared(layout)/'initial_state.json')

def install():
    global _installed,old
    if _installed:return old
    # Execute the exact frozen B11 install, then only expose the declared G arm.
    old=R.install()
    R.ARMS=protocol()['arms']
    old.method.ARMS=tuple(R.ARMS)
    old.method.MODE={arm:cfg['score_mode']for arm,cfg in R.ARMS.items()}
    R.protocol=protocol;old.protocol=protocol
    old.HERE=HERE;old.prepared=prepared;old.verify=verify
    _installed=True
    return old

def selfcheck(layout):
    first=verify(layout);old=install()
    import numpy as np
    from nso.cpu_sensor_contract_v10 import digest
    shared=R.world_and_packet(layout);rows={};audits={};plans={}
    for arm in ('S_visibility','G_visibility'):
        world,tr,p,comp,runtime,effective,owner=old.bootstrap(layout,arm,shared)
        mapper=runtime.states[0]['mapper'];state=comp._cpu_backend.scenes[0]
        routes,common,_=R.native_candidates(dict(state=state,backend=comp._cpu_backend))
        comp._cpu_backend.select_target(0)
        plans[arm]=state['last_selection']
        audits[arm]=old.method.audit_log(mapper)[-1]
        assert world.step_count==0 and p.sha256()==first['initial_packet_sha256']
        rows[arm]=dict(candidate_sha256=digest(routes),common_sha256=digest(common),
            initial_packet_sha256=p.sha256(),owner_sha256=owner['owner_sha256'],
            belief_sha256=digest(mapper.belief),points_sha256=audits[arm]['points_sha256'],
            G_base_weights_sha256=audits[arm]['G_base_weights_sha256'],
            S_base_weights_sha256=audits[arm]['S_base_weights_sha256'],
            history_keep_sha256=audits[arm]['history_keep_sha256'],simulator_actions=0)
    assert rows['G_visibility']==rows['S_visibility']
    historical=read(baseline(layout)/'template_audit.json')[0]
    assert audits['S_visibility']==historical,'Extended S binding differs from recorded B11 first decision'
    g=dict(audits['G_visibility']);g['arm']='S_visibility'
    assert g==historical,'G must only choose from the existing neutral scores'
    saved=read(baseline(layout)/'choices.json')[0]['outputs']
    for key in ('candidates','scores','score_audit','selected'):
        assert plans['S_visibility'][key]==saved[key],key
    for key in ('candidates','scores'):
        assert plans['G_visibility'][key]==plans['S_visibility'][key],key
    verify(layout)
    write(HERE/'layouts'/layout/'selfcheck.json',dict(status='passed_zero_action_shared_input_and_exact_historical_S_binding',
        layout=layout,arms=rows,selected_score_modes={'S_visibility':'S','G_visibility':'G'},
        historical_S_template_audit=R.resource(baseline(layout)/'template_audit.json'),
        historical_S_choices=R.resource(baseline(layout)/'choices.json'),
        historical_S_audit_exact=True,historical_S_candidates_scores_selection_exact=True,
        shared_G_S_diagnostic_scores_exact=True,selected_G_candidate=plans['G_visibility']['selected']['candidate_id'],
        extension_freeze=R.resource(HERE/'freeze.json'),simulator_actions=0))
    print(json.dumps(dict(status='selfcheck_passed',layout=layout,simulator_actions=0)),flush=True)

def run(layout):
    verify(layout)
    assert read(HERE/'layouts'/layout/'selfcheck.json')['status']=='passed_zero_action_shared_input_and_exact_historical_S_binding'
    old=install();arm='G_visibility';parent=HERE/'layouts'/layout/'runs'/arm
    before=set(parent.glob('attempt_*'))if parent.exists()else set()
    old.run(layout,arm)
    new=set(parent.glob('attempt_*'))-before;assert len(new)==1;out=new.pop()
    result=read(out/'result.json')
    assert result['effective_args']['cpu_score_mode']=='G' and result['effective_args']['dual_template_arm']==arm
    write(out/'B11_execution_binding.json',dict(wrapper=R.resource(Path(__file__)),
        execution_freeze=R.resource(EXEC/'freeze.json'),extension_freeze=R.resource(HERE/'freeze.json'),
        new_protocol=R.resource(HERE/'protocol.json'),predictor='visibility',selected_score_mode='G',
        full_posterior_same_quadrature=True,short_cost_band_used=False,geometry_from_acquired_frames_only=True,
        parent_S_visibility=R.resource(baseline(layout)/'artifact_manifest.json'),
        only_policy_change='Select neutral structural-prior G scores instead of S scores; shared ideal-instance frontend retained.'))
    write(out/'artifact_manifest.json',dict(status='completed_execution',files_sha256={p.name:R.sha(p)for p in out.iterdir()if p.is_file()and p.name!='artifact_manifest.json'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','verify','selfcheck','run'))
    p.add_argument('--layout',choices=LAYOUTS);a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='verify':verify(a.layout)
    elif a.layout is None:p.error('layout required')
    elif a.command=='selfcheck':selfcheck(a.layout)
    else:run(a.layout)
if __name__=='__main__':main()
