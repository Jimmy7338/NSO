#!/usr/bin/env python3
"""Complete-table information screen; failures remain visible and unfiltered."""
import argparse,csv,hashlib,json,math,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.collect_semantic_gain_v13_history import sha,write

OPTIONS=('continue_coverage','observe_asset_A','observe_asset_B')
REFERENCES=('2026','2027','2028')
PARENTS=('D19-P00','D19-P01')
ASSIGNMENTS=('A_complex_B_simple','A_simple_B_complex')


def _unit(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError(name+' must be a finite number in [0,1]')


def _close(left,right,name):
    if not math.isclose(left,right,rel_tol=0.,abs_tol=1e-10):raise ValueError(name+' differs')


def parent_value(records,reference):
    assignments=sorted({r['case']['assignment'] for r in records})
    options=OPTIONS
    table={(r['case']['assignment'],r['case']['option']):r for r in records}
    if len(records)!=6 or len(table)!=6 or len(assignments)!=2 or set(table)!={(a,o) for a in assignments for o in options}:
        raise ValueError('complete unique two-by-three table required')
    groups={}
    for assignment in assignments:
        votes=tuple(table[assignment,options[0]]['case']['observed_votes'])
        if len(votes)!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in votes):
            raise ValueError('two finite observed category votes required')
        for option in options:
            if tuple(table[assignment,option]['case']['observed_votes'])!=votes:
                raise ValueError('initial observed groups depend on the chosen option')
            metric=table[assignment,option]['after'][reference]
            if type(metric['eligible']) is not bool:raise ValueError('eligibility must be Boolean')
            _unit(metric['05cm']['joint_external'],'joint_external')
        groups.setdefault(votes,[]).append(assignment)
    def feasible(option,group):return all(table[a,option]['after'][reference]['eligible'] for a in group)
    def value(option,group):return sum(table[a,option]['after'][reference]['05cm']['joint_external'] for a in group)/len(group)
    common=[o for o in options if feasible(o,assignments)]
    result=dict(reference=reference,observed_groups=list(groups.values()),
        observed_group_cues=[dict(votes=list(v),assignments=g) for v,g in groups.items()],common_feasible_options=common,
        eligibility={a:{o:table[a,o]['after'][reference]['eligible'] for o in options} for a in assignments},
        unfiltered_rewards={a:{o:table[a,o]['after'][reference]['05cm']['joint_external'] for o in options} for a in assignments})
    if not common:return dict(result,status='no_common_feasible_baseline',screen_passed=False)
    blind=max(common,key=lambda o:value(o,assignments));blind_value=value(blind,assignments)
    group_rows=[];conditional=0.
    for group in groups.values():
        allowed=[o for o in options if feasible(o,group)]
        if not allowed:return dict(result,status='observed_group_infeasible',screen_passed=False)
        best=max(allowed,key=lambda o:value(o,group));v=value(best,group)
        conditional+=len(group)/len(assignments)*v
        group_rows.append(dict(assignments=group,option=best,value=v))
    relative=(conditional-blind_value)/blind_value if blind_value>1e-12 else None
    return dict(result,status='evaluated',blind_option=blind,blind_value=blind_value,
        conditional_value=conditional,absolute_information=conditional-blind_value,
        absolute_information_percentage_points=100.*(conditional-blind_value),
        relative_information=relative,conditional_choices=group_rows,
        conditional_choice_differs_from_common=any(row['option']!=blind for row in group_rows),
        optimistic_unit_score_absolute_ceiling=1.-blind_value,
        optimistic_unit_score_relative_ceiling=(1./blind_value-1.) if blind_value>1e-12 else None,
        strict_relative_screen_threshold=.05,screen_numerical_tolerance=1e-12,
        screen_passed=relative is not None and relative>.05 and not math.isclose(relative,.05,rel_tol=0.,abs_tol=1e-12))


def validate_metric(metric):
    """Reject inconsistent bookkeeping without filtering poor physical results."""
    if metric['mission_asset_count']!=6 or len(metric['instances'])!=6 or {x['id'] for x in metric['instances']}!=set(range(6)):
        raise ValueError('all six fixed external facilities must remain in the metric')
    _unit(metric['coverage_2d'],'coverage')
    for key in ('returned','failed','eligible'):
        if type(metric[key]) is not bool:raise ValueError(key+' must be Boolean')
    collisions=metric['collisions']
    if type(collisions) is not int or collisions<0:raise ValueError('invalid collision count')
    eligible=metric['coverage_2d']>=.8 and metric['returned'] and collisions==0 and not metric['failed']
    if eligible!=metric['eligible']:raise ValueError('failure/coverage/return contradicts eligibility')
    for tag in ('02cm','05cm','10cm'):
        for item in metric['instances']:
            score=item[tag]
            for name in ('precision','recall','f1'):_unit(score[name],name)
            p,r=score['precision'],score['recall'];_close(score['f1'],2*p*r/(p+r) if p+r else 0.,'instance F1')
            if item.get('missing') and any(score[name]!=0. for name in ('precision','recall','f1')):
                raise ValueError('missing facility has nonzero credit')
        macro=sum(x[tag]['f1'] for x in metric['instances'])/6
        _close(metric[tag]['external_macro_f1'],macro,'external six-facility macro')
        _close(metric[tag]['asset_macro_f1'],macro,'legacy macro alias')
        _close(metric[tag]['joint_external'],metric['coverage_2d']*macro,'joint external')
        _close(metric[tag]['joint_asset'],metric[tag]['joint_external'],'joint alias')


def validate_record(row):
    if set(row['before'])!=set(REFERENCES) or set(row['after'])!=set(REFERENCES):
        raise ValueError('three predeclared reference samples required at both endpoints')
    count=row['paid_actions']
    if type(count) is not int or count<0 or len(row['actions'])!=count:
        raise ValueError('paid action count differs from physical action record')
    if row['primitive_budget_compliant']!=(count<=row['case']['budget']):
        raise ValueError('budget compliance flag differs from action count')
    collisions=sum(bool(action['collision']) for action in row['actions'])
    if collisions!=row['collisions']:raise ValueError('physical collision count differs')
    termination=row['termination']
    if termination['final_action_id']!=count:raise ValueError('termination frame does not match paid action count')
    for ref in REFERENCES:
        for endpoint in ('before','after'):validate_metric(row[endpoint][ref])
        metric=row['after'][ref]
        if metric['failed']!=termination['failed'] or metric['returned']!=termination['returned_to_anchor']:
            raise ValueError('terminal failure/return contradicts final metric')
        if metric['collisions']!=row['collisions']:raise ValueError('metric collisions differ from physical trace')
        if metric['eligible'] and not row['primitive_budget_compliant']:
            raise ValueError('over-budget outcome cannot be called eligible')
        for name in ('coverage_2d','returned','failed','collisions','eligible'):
            if metric[name]!=row['after']['2026'][name]:raise ValueError('reference sampling changed physical eligibility')
    expected=[0]+list(range(20,count+1,20))
    if expected[-1]!=count:expected.append(count)
    curve=row['quality_curve']
    if row['curve_action_stride']!=20 or [point['action_id'] for point in curve]!=expected:
        raise ValueError('declared periodic quality curve is incomplete')
    for point in curve:validate_metric(point['metrics'])
    if count>0 and curve[0]['metrics']!=row['before']['2026']:
        raise ValueError('quality curve initial point differs')
    if curve[-1]['metrics']!=row['after']['2026']:raise ValueError('quality curve terminal point differs')


def load_records(source):
    """Read only a completed, sealed twelve-case experiment with fresh replays."""
    manifest=json.loads((source/'manifest.json').read_text())
    cases=manifest['cases']
    if manifest['status']!='complete' or len(cases)!=12:raise ValueError('complete12-case run required')
    for case in cases:
        if case['sensor_model']!='iid_025px' or case['noise_seed']!=1901:
            raise ValueError('pilot sensor condition differs from the declared analysis scope')
        if type(case['budget']) is not int or case['budget']<1:raise ValueError('positive integer common budget required')
    combinations={(c['parent'],c['assignment'],c['option']) for c in cases}
    if {c['index'] for c in cases}!=set(range(12)) or combinations!={(p,a,o) for p in PARENTS for a in ASSIGNMENTS for o in OPTIONS}:
        raise ValueError('unique declared two-parent, two-assignment, three-option matrix required')
    inventory=json.loads((source/'artifact_hashes.json').read_text())
    files={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    if set(inventory)!=files:raise ValueError('sealed artifact inventory does not cover exact current files')
    for name,wanted in inventory.items():
        if sha(source/name)!=wanted:raise ValueError('source artifacts changed')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        if set(archive.namelist())!=set(manifest['source_sha256']):raise ValueError('frozen source archive inventory differs')
        for name,wanted in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=wanted:raise ValueError('frozen source archive hash differs')
    references=manifest.get('reference_sha256',{})
    if len(references)!=12:raise ValueError('twelve frozen reference caches required')
    for name,wanted in references.items():
        if sha(source/name)!=wanted:raise ValueError('frozen evaluator reference changed')
    records=[]
    for case in cases:
        folder=source/f'case_{case["index"]:02d}'
        verification=json.loads((folder/'verification.json').read_text())
        if verification['status']!='passed' or verification.get('fresh_world_sensor_action_metric_replay') is not True:
            raise ValueError('fresh physical replay missing')
        row=json.loads((folder/'result.json').read_text())
        if row['case']!=case:raise ValueError('declared case differs')
        if row['trained'] is not False:raise ValueError('restricted option pilot must not claim a trained policy')
        if verification['actions']!=row['paid_actions']:raise ValueError('replay action count differs')
        packets={path.name for path in (folder/'packets').glob('*.npz')}
        if packets!={f'{index:04d}.npz' for index in range(row['paid_actions']+1)}:
            raise ValueError('raw packet sequence is incomplete')
        validate_record(row);records.append(row)
    for parent in PARENTS:
        subset=[r for r in records if r['case']['parent']==parent]
        if len({r['case']['budget'] for r in subset})!=1:raise ValueError('common parent budget differs by option/assignment')
        for option in OPTIONS:
            paired=[r for r in subset if r['case']['option']==option]
            if len({r['case']['shared_geometry_sha256'] for r in paired})!=1:
                raise ValueError('paired option nonsemantic initial observation differs')
        for reference in REFERENCES:parent_value(subset,reference)
    return manifest,records


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest,records=load_records(a.source)
    a.output.mkdir(parents=True,exist_ok=False)
    names=['scripts/analyze_facility_v19.py','tests/virtual3d/test_facility_information_v19.py',
           'scripts/collect_semantic_gain_v13_history.py']
    sources={name:sha(ROOT/name) for name in names}
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    results=[]
    for parent in sorted({r['case']['parent'] for r in records}):
        subset=[r for r in records if r['case']['parent']==parent]
        results.append(dict(parent=parent,references={str(seed):parent_value(subset,str(seed)) for seed in (2026,2027,2028)}))
    write(a.output/'result.json',dict(status='complete',unique_physical_outcomes=12,independent_replays=12,
        parent_layouts=2,primary_reference=2026,results=results,
        companion_references=[2027,2028],reference_sampling_repeats_are_independent_experiments=False,
        primary_screen_passed=all(r['references']['2026']['screen_passed'] for r in results),
        training_allowed=False,full_architecture_efficacy_proven=False,
        task='six_facility_external_documentation',sensor_model='iid_025px',
        note='Restricted development option oracle only; passing is not a learned policy, significance or training qualification. All six assets equal; external/all-facility targets coincide for this solid-body scene.',
        input_inventory_sha256=sha(a.source/'artifact_hashes.json')))
    write(a.output/'manifest.json',dict(status='complete',source_sha256=sources,
        input_inventory_sha256=sha(a.source/'artifact_hashes.json'),training_allowed=False))
    rows=[]
    for record in records:
        for ref,metrics in record['after'].items():
            rows.append(dict(parent=record['case']['parent'],assignment=record['case']['assignment'],option=record['case']['option'],
                reference=ref,coverage=metrics['coverage_2d'],asset_f1=metrics['05cm']['asset_macro_f1'],joint=metrics['05cm']['joint_external'],
                completed=metrics['05cm']['completion_fraction'],eligible=metrics['eligible'],global_joint=metrics['global_legacy']['joint_05cm'],
                failed=metrics['failed'],returned=metrics['returned'],
                external_error_mean_m=metrics['continuous_external']['accuracy']['complete_mission_macro_mean_m'],
                external_error_rmse_m=metrics['continuous_external']['accuracy']['complete_mission_macro_rmse_m'],
                external_error_p95_m=metrics['continuous_external']['accuracy']['complete_mission_macro_p95_m'],
                actions=record['paid_actions'],path_m=record['path_distance_m'],collisions=record['collisions']))
    with (a.output/'outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    curves=[]
    for record in records:
        for point in record['quality_curve']:
            metric=point['metrics']
            curves.append(dict(parent=record['case']['parent'],assignment=record['case']['assignment'],
                option=record['case']['option'],reference='2026',action_id=point['action_id'],
                coverage=metric['coverage_2d'],asset_f1=metric['05cm']['external_macro_f1'],
                joint=metric['05cm']['joint_external'],completed=metric['05cm']['completion_fraction'],
                eligible=metric['eligible'],failed=metric['failed'],returned=metric['returned']))
    with (a.output/'quality_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    write(a.output/'artifact_hashes.json',{p.name:sha(p) for p in a.output.iterdir() if p.is_file()})

if __name__=='__main__':main()
