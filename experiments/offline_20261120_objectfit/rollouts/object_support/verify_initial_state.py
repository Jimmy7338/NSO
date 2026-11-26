#!/usr/bin/env python3
"""Read-only zero-action initial-state verification of the frozen wrapper.

The wrapper's original selfcheck accessed audit['objects'] even when no object
was observed. This separate verifier accepts the model's actual empty objects
list. It does not change any policy, model, freeze or run entry point.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('B5_frozen_object_rollout',HERE/'run.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
r.HERE=HERE
r.dependencies('object_support')
from nso.cpu_sensor_contract_v10 import digest

for layout in ('shelf_left','shelf_right'):
    first=r.verify(layout);shared=r.old.world_and_packet(layout);rows={}
    for arm in r.protocol()['arms']:
        world,tr,p,comp,runtime,effective,owner=r.old.bootstrap(layout,arm,shared)
        assert p.sha256()==first['initial_packet_sha256']
        comp._cpu_backend.select_target(0)
        state=comp._cpu_backend.scenes[0];plan=state['last_selection']
        models=state['mapper']._dual_template_state['cache']['models']
        audits={name:dict(model.audit,objects=model.objects)for name,model in models.items()}
        for name,audit in audits.items():
            assert audit['object_support'] and not audit['full_observed_support']
            assert not audit['geometry_fit'] and not audit['free_space_likelihood'] and not audit['applicability']
            assert audit['history_rejection'] and audit['applied_global_point_cap'] is None
            assert audit['fit_update']==(name=='G' or arm!='S_no_fit')
            assert audit['measured_surface_consumption']==(name=='G' or arm!='S_no_measured_consumption')
            if name=='S' and arm=='S_no_fit':
                assert all(o['consumed_geometry_log_likelihood_ratio']==0.0 for o in audit['objects'])
        rows[arm]=dict(model_audits=audits,candidate_sha256=digest(plan['candidates']),belief_sha256=digest(state['mapper'].belief),first_plan=plan,paid_actions=world.step_count)
        assert world.step_count==0
    assert len({v['candidate_sha256']for v in rows.values()})==1
    assert len({v['belief_sha256']for v in rows.values()})==1
    r.verify(layout)
    result=dict(status='passed',shared_initial_sensor_geometry_candidates=True,arms=rows,
                independent_verifier_sha256=r.sha(Path(__file__)),frozen_wrapper_unchanged=True,
                initial_empty_objects_allowed=True,scope='zero-action initial sensors/candidates and configured module switches, not policy effectiveness')
    r.write(HERE/'layouts'/layout/'selfcheck.json',result)
    print(json.dumps(dict(status='passed',layout=layout,arms=list(rows),paid_actions=0)),flush=True)
