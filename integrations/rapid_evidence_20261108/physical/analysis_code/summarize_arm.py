#!/usr/bin/env python3
"""Read actual terminal result and logged same-state choices; no replay."""
import argparse,hashlib,json
from pathlib import Path
from compare_pair import arm

def summarize(report,output):
    report=Path(report).resolve();metrics,settings,_,_=arm(report)
    rows=[];seen=set();events=[];paid=0;pose=None
    for lineno,line in enumerate((report/'joint/decisions.jsonl').read_text().splitlines(),1):
        d=json.loads(line)
        if d['request']['method']=='observe':
            paid=d['request']['params']['action_id'];pose=d['response']['result'].get('state',{}).get('pose')
        if d['request']['method']!='next_action':continue
        x=d['response']['result'];runtime=x.get('runtime_events',[])
        for event in runtime:
            if event.get('event') in ['exploration_joint_replan_after_rejection','close_after_final_observation']:
                events.append(dict(source_line=lineno,paid_actions_before=paid,current_grid_pose=pose,event=event))
        q=x.get('selection')
        if not q or q['plan_number'] in seen:continue
        seen.add(q['plan_number']);routes=q['candidates'];choices={}
        for mode in ['C','G','S','X']:
            scores=q['scores'][mode];i=min(range(len(routes)),key=lambda i:(-scores[i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
            if i is None or scores[i]<=0:choices[mode]=None
            else:
                c=routes[i];choices[mode]=dict(candidate_id=c['candidate_id'],group=c['group'],target=c['pose'],cost=c['cost'],score=scores[i],first_action=c['outbound_actions'][0])
        selected=q['selected']
        global_event=next((e for e in runtime if e.get('event')=='global_choice'),{})
        rows.append(dict(source_line=lineno,paid_actions_before=paid,current_grid_pose=pose,
            plan_number=q['plan_number'],selected_map_version=global_event.get('map_version'),
            selection_call_id=None if selected is None else selected.get('selection_call_id'),
            belief_sha256=q['candidate_audit'].get('belief_sha256'),safe_sha256=q['candidate_audit'].get('safe_sha256'),camera_seen_sha256=q['candidate_audit'].get('camera_seen_sha256'),
            candidate_targets=[dict(candidate_id=c['candidate_id'],group=c['group'],target=c['pose'],cost=c['cost']) for c in routes],
            candidate_count=len(routes),actual_mode=q['mode'],actual_selected_candidate=None if selected is None else selected['candidate_id'],
            within_state_choices=choices,scores={m:q['scores'][m] for m in ['C','G','S','X']},
            score_components=[dict(candidate_id=a['candidate_id'],coverage=a['coverage_only_proxy'],quality=a['quality_total_proxy'],common=a['common_total_proxy'],prior_G=a['prior_potentials']['G'],prior_S=a['prior_potentials']['S'],prior_X=a['prior_potentials']['X'],incremental_aperture=a['incremental_aperture']) for a in q['score_audit']],
            G_equals_S_per_candidate=q['scores']['G']==q['scores']['S'],candidate_generation_status=q['candidate_audit']['status']))
    def ident(x):return None if x is None else x['candidate_id']
    planning=dict(new_planning_states=len(rows),multi_candidate_states=sum(z['candidate_count']>1 for z in rows),
        C_S_selected_option_differences=sum(ident(z['within_state_choices']['C'])!=ident(z['within_state_choices']['S']) for z in rows),
        different_candidate_when_both_positive=sum(z['within_state_choices']['C'] is not None and z['within_state_choices']['S'] is not None and ident(z['within_state_choices']['C'])!=ident(z['within_state_choices']['S']) for z in rows),
        C_no_positive_option_S_positive=sum(z['within_state_choices']['C'] is None and z['within_state_choices']['S'] is not None for z in rows),
        G_S_equal_all_candidates_all_new_states=all(z['G_equals_S_per_candidate'] for z in rows),states=rows)
    summary=dict(schema='nso.joint_coverage_pair.arm_terminal_summary.v2',status='actual_terminal_episode',report=str(report),
        source_sha256={f:hashlib.sha256((report/f).read_bytes()).hexdigest() for f in ['result.json','joint_configuration.json','joint/decisions.jsonl']},
        metrics=metrics,planning=planning,original_runtime_rejections_and_close=events,
        limits=['Same-state C/G/S/X scores come from original recorded states, not separately executed trajectories.',
            'C/S differences remove both surface quality and object prior; they are not semantic-effect evidence.',
            'Cross-run route differences can reflect different actual map/candidate sets; do not attribute them to score mode alone.',
            'Each quality proxy uses its own arm bootstrap support; different support forbids matched-quality benefit from absolute cross-arm values.',
            'Original circular known-safe rejection and failed/incomplete returns are retained; no new policy gate was added.'])
    output=Path(output);output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();summarize(a.report,a.output)
