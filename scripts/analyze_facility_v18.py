#!/usr/bin/env python3
"""Complete-table information screen; failures remain visible and unfiltered."""
import argparse,csv,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.collect_semantic_gain_v13_history import sha,write


def parent_value(records,reference):
    assignments=sorted({r['case']['assignment'] for r in records})
    options=('continue_coverage','observe_asset_A','observe_asset_B')
    table={(r['case']['assignment'],r['case']['option']):r for r in records}
    if len(table)!=6 or len(assignments)!=2:raise ValueError('complete two-by-three table required')
    groups={}
    for assignment in assignments:
        votes=tuple(table[assignment,options[0]]['case']['observed_votes'])
        groups.setdefault(votes,[]).append(assignment)
    def feasible(option,group):return all(table[a,option]['after'][reference]['eligible'] for a in group)
    def value(option,group):return sum(table[a,option]['after'][reference]['05cm']['joint_asset'] for a in group)/len(group)
    common=[o for o in options if feasible(o,assignments)]
    result=dict(reference=reference,observed_groups=list(groups.values()),common_feasible_options=common,
        eligibility={a:{o:table[a,o]['after'][reference]['eligible'] for o in options} for a in assignments},
        unfiltered_rewards={a:{o:table[a,o]['after'][reference]['05cm']['joint_asset'] for o in options} for a in assignments})
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
        conditional_value=conditional,relative_information=relative,conditional_choices=group_rows,
        optimistic_unit_score_relative_ceiling=(1./blind_value-1.) if blind_value>1e-12 else None,
        screen_passed=relative is not None and relative>.05)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.source/'manifest.json').read_text())
    if manifest['status']!='complete' or len(manifest['cases'])!=12:raise ValueError('complete12-case run required')
    for name,wanted in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/name)!=wanted:raise ValueError('source artifacts changed')
    records=[]
    for case in manifest['cases']:
        folder=a.source/f'case_{case["index"]:02d}'
        if json.loads((folder/'verification.json').read_text())['status']!='passed':raise ValueError('physical replay missing')
        row=json.loads((folder/'result.json').read_text())
        if row['case']!=case:raise ValueError('declared case differs')
        records.append(row)
    a.output.mkdir(parents=True,exist_ok=False)
    names=['scripts/analyze_facility_v18.py','tests/virtual3d/test_facility_information_v18.py',
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
        primary_screen_passed=all(r['references']['2026']['screen_passed'] for r in results),
        training_allowed=False,full_architecture_efficacy_proven=False,
        note='Restricted development option oracle only; passing is not a learned policy, significance or training qualification.',
        input_inventory_sha256=sha(a.source/'artifact_hashes.json')))
    write(a.output/'manifest.json',dict(status='complete',source_sha256=sources,
        input_inventory_sha256=sha(a.source/'artifact_hashes.json'),training_allowed=False))
    rows=[]
    for record in records:
        for ref,metrics in record['after'].items():
            rows.append(dict(parent=record['case']['parent'],assignment=record['case']['assignment'],option=record['case']['option'],
                reference=ref,coverage=metrics['coverage_2d'],asset_f1=metrics['05cm']['asset_macro_f1'],joint=metrics['05cm']['joint_asset'],
                completed=metrics['05cm']['completion_fraction'],eligible=metrics['eligible'],global_joint=metrics['global_legacy']['joint_05cm'],
                actions=record['paid_actions'],path_m=record['path_distance_m'],collisions=record['collisions']))
    with (a.output/'outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(a.output/'artifact_hashes.json',{p.name:sha(p) for p in a.output.iterdir() if p.is_file()})

if __name__=='__main__':main()
