"""Common task-aware geometric continuation for the V18 information pilot.

No fitted or semantic score is deployed here. The pilot measures whether
observed categories can select among physically executed common options.
"""
from copy import deepcopy
import numpy as np
from nso.cpu_four_modules_v17 import CPUFourModulesV17
from nso.cpu_four_modules_v16 import axis_components_v16
from nso.observed_option_runtime_v17 import ObservedOptionRuntimeV17
from nso.recovery_runtime_v14 import RecoveryRuntimeV14


class FacilityBackendV18(CPUFourModulesV17):
    def __init__(self,args,num_scenes,shape):
        if getattr(args,'cpu_score_mode',None)!='G':
            raise ValueError('V18 pilot has only a common geometric continuation; semantic head is not fitted')
        super().__init__(args,num_scenes,shape)

    def _scores_v10_1(self,state,routes,attempted):
        _,rows=super()._scores_v10_1(state,routes,attempted)
        # Target presence is common task information; class vote never enters.
        # The pilot uses visible markers, not hidden simulator IDs or masks.
        target=np.array([a['marked_points']>0 for a in state['assets']],bool)
        total=max(1,state['mapper'].belief.size)
        scores={k:[] for k in ('N','G','O','S','X','M')}
        for row in rows:
            coverage=(row['predicted_radar_cells']*row['gain_posterior']['radar']+
                      row['predicted_camera_cells']*row['gain_posterior']['camera'])/total
            support=np.asarray(row['corrected_aperture'])
            asset=float(support[target].mean()) if target.any() else 0.
            value=float(coverage+asset)
            row['v18_task_proxy']=dict(expected_new_coverage_fraction=float(coverage),
                mean_visible_target_direction_support=asset,weights=[1.,1.],
                calibrated_asset_f1=False,class_vote_used=False,full_reference_used=False)
            for mode in scores:scores[mode].append(float(coverage) if mode=='N' else value)
        return scores,rows


class FacilityRuntimeV18(ObservedOptionRuntimeV17):
    # Preserve the already validated common denial recovery. This calls our
    # normal choose_goal, so a missed first-stage arrival gets an explicit
    # geometric fallback, while every recovered action remains guarded.
    next_local_action=RecoveryRuntimeV14.next_local_action

    def install_pilot_option(self,option_name):
        state=self.states[0]
        if state['pending'] is not None or state['active_actions'] or state['closed']:
            raise RuntimeError('pilot intervention requires an idle observation boundary')
        self.choose_goal(0,list(state['packet'].position),(0,self.full_shape[0],0,self.full_shape[1]))
        backend=self.components._cpu_backend;b=backend.scenes[0]
        selection=b['last_selection'];routes=selection['candidates'];nominal=deepcopy(state['option'])
        if nominal is None:raise ValueError('pilot state has no feasible positive common option')
        if option_name=='continue_coverage':
            ids=[i for i,r in enumerate(routes) if r['group'].startswith('coverage_')]
            if not ids:raise ValueError('no shared coverage candidate')
            index=min(ids,key=lambda i:(-selection['scores']['N'][i],routes[i]['cost'],routes[i]['candidate_id']))
        else:
            # A/B denote left-to-right OBSERVED targets, not true object IDs.
            targets=sorted([i for i,a in enumerate(b['assets']) if a['marked_points']>0],
                           key=lambda i:tuple(b['assets'][i]['aabb_center']))
            if len(targets)!=2:raise ValueError('pilot requires exactly two observed task targets')
            if option_name not in ('observe_asset_A','observe_asset_B'):raise ValueError('unknown pilot option')
            ai=targets[option_name=='observe_asset_B']
            ids=[i for i,r in enumerate(routes) if r['asset_index']==ai and '_corner_' in r['group']]
            if not ids:raise ValueError('no common observed approach for target')
            index=min(ids,key=lambda i:(routes[i]['outbound_cost'],routes[i]['cost'],routes[i]['candidate_id']))
        route=deepcopy(routes[index]);option=deepcopy(route)
        for key in ('option_id','selection_call_id','selected_map_version','selected_feedback_version','parent_region'):
            option[key]=nominal[key]
        state.update(option=option,active_actions=list(option['outbound_actions']),phase='outbound')
        b['selected']=deepcopy(option)
        if option_name!='continue_coverage':backend.arm_region_continuation(0,route)
        receipt=dict(event='v18_declared_option',name=option_name,action_id=state['packet'].action_id,
            selected=deepcopy(option),nominal_candidate_id=nominal['candidate_id'],class_used=False)
        self.audit.append(receipt)
        return deepcopy(selection),receipt


def facility_components_v18(args,shape):
    comp=axis_components_v16(args,shape)
    comp._cpu_backend=FacilityBackendV18(args,1,shape)
    comp.capabilities=dict(comp._cpu_backend.capabilities)
    comp.capabilities['v18_role']='unfitted task-aware geometric continuation for information-value pilot'
    return comp
