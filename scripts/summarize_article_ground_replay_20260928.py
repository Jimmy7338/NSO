#!/usr/bin/env python3
"""Read-only tables for sealed, successful old/new fixed-path evidence replay.

Consumes replay records only. No scene asset, sensor, World, controller, mapper,
residual evaluator, or hypothesis scorer is imported or invoked here.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'audit_results/article_stage_20260928'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def zipped(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def dump(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def csv_table(path, rows):
    if not rows:
        raise ValueError('Refusing a silently empty diagnostic table')
    with path.open('x', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, separators=(',', ':'))
                if isinstance(v, (list, dict)) else v for k,v in row.items()})


def run(base, output):
    if output.exists() or not output.resolve().is_relative_to(STAGE):
        raise ValueError('Require a new exclusive analysis path under article stage')
    declaration = read(base/'predeclaration.json')
    inputs = {str((base/'predeclaration.json').relative_to(ROOT)):sha(base/'predeclaration.json')}
    stages = {}
    for name in ('control', 'new_ground'):
        path = base/name
        manifest = read(path/'manifest.json')
        if manifest['predeclaration_sha256'] != sha(base/'predeclaration.json'):
            raise ValueError('Predeclaration seal mismatch')
        for rel, info in manifest['files'].items():
            item = path/rel
            if item.stat().st_size != info['bytes'] or sha(item) != info['sha256']:
                raise ValueError('Replay artifact mismatch: '+str(item))
            inputs[str(item.relative_to(ROOT))] = sha(item)
        inputs[str((path/'manifest.json').relative_to(ROOT))] = sha(path/'manifest.json')
        summary = read(path/'summary.json')
        if summary['status'] != 'passed' or summary['processed_frames'] != 161:
            raise ValueError('Both sealed full 161-frame stages must pass')
        stages[name] = dict(summary=summary, records=zipped(path/'evidence.json.gz'),
            matches={r['paid_step']:{m['new_instance_id']:m for m in r['matches']}
                     for r in zipped(path/'observed_anchor_matches.json.gz')})
    if any(a['observation_sha256'] != b['observation_sha256']
           for a,b in zip(stages['control']['records'],stages['new_ground']['records'])):
        raise ValueError('Old/new raw paid packets differ')
    frame_rows=[];instance_rows=[];event_rows=[];details={}
    for stage,data in stages.items():
        previous={};reasons=Counter();ground_stats=Counter()
        for record in data['records']:
            step=record['paid_step'];association=record['association']
            beliefs={method:{r['instance_id']:r for r in record['all_method_beliefs'][method]['instances']}
                     for method in ('G','B','S')}
            feedback={r['instance_id']:r for r in record['geometry_feedback']}
            rejected=Counter(r['reason'] for r in association['rejected']);reasons.update(rejected)
            ground=association.get('article_ground_association',{})
            ground_stats[ground.get('reason','original_no_ground_revision')]+=1
            max_difference=0.;informative_peer_edges=0;peer_edges=0
            for row in association['instances']:
                key=row['instance_id'];s=beliefs['S'][key];b=beliefs['B'][key]
                difference=max(abs(x-y) for x,y in zip(s['structure_probabilities'],b['structure_probabilities']))
                max_difference=max(max_difference,difference)
                peer_edges+=len(s['peer_instance_ids'])
                informative_peer_edges+=sum(max(beliefs['S'][peer]['geometry_log_evidence'])-
                    min(beliefs['S'][peer]['geometry_log_evidence'])>1e-12 for peer in s['peer_instance_ids'])
                match=data['matches'][step][key]
                state=dict(qualified=row['semantic_conditioning_used'],uncertain=row['association_uncertain'],
                    class_conflict=row['class_conflict'],observed_class=row['observed_class'])
                old=previous.get(key)
                if old!=state:
                    event_rows.append(dict(stage=stage,paid_step=step,instance_id=key,
                        matched_old_instance_id=match['matched_old_instance_id'],previous_state=old,
                        new_state=state,current_rejection_reasons=dict(rejected)))
                previous[key]=state
                instance_rows.append(dict(stage=stage,paid_step=step,instance_id=key,
                    matched_old_instance_id=match['matched_old_instance_id'],identity_status=match['status'],
                    observed_anchor_distance_m=match['distance_m'],**state,
                    support_points_before_current_feedback=row['support_points'],
                    near_ground_support_points_z_below_015m=row['near_ground_support_points_z_below_015m'],
                    geometry_feedback_applied=feedback.get(key,{}).get('applied',False),
                    geometry_log_evidence_after_feedback=s['geometry_log_evidence'],
                    G_structure_probabilities=beliefs['G'][key]['structure_probabilities'],
                    B_structure_probabilities=b['structure_probabilities'],S_structure_probabilities=s['structure_probabilities'],
                    S_rho=s['rho'],S_peer_ids=s['peer_instance_ids'],S_B_max_absolute_posterior_difference=difference))
            frame_rows.append(dict(stage=stage,paid_step=step,observation_sha256=record['observation_sha256'],
                observed_instances=len(association['instances']),qualified_instances=sum(r['semantic_conditioning_used'] for r in association['instances']),
                uncertain_instances=sum(r['association_uncertain'] for r in association['instances']),
                applied_feedback=sum(r['applied'] for r in record['geometry_feedback']),
                qualified_peer_edges=peer_edges,informative_peer_edges=informative_peer_edges,
                S_B_max_absolute_posterior_difference=max_difference,association_rejection_reasons=dict(rejected),
                ground_accepted=ground.get('accepted',False),ground_reason=ground.get('reason','original_no_ground_revision'),
                ground_excluded_pixels=ground.get('excluded_pixels',0),ground_support_pixels=ground.get('validated_support_pixels'),
                ground_standardized_rms=ground.get('standardized_residual_rms'),ground_measured_z_m=ground.get('measured_ground_z_m'),
                ground_xy_span_m=ground.get('validated_xy_span_m')))
        details[stage]=dict(first_steps=data['summary']['first_steps'],per_instance_first_steps=data['summary']['per_instance_first_steps'],
            counters=data['summary']['counters'],association_rejection_reasons=dict(reasons),ground_reasons=dict(ground_stats))
    output.mkdir(parents=True,exist_ok=False)
    csv_table(output/'frames.csv',frame_rows)
    csv_table(output/'instances.csv',instance_rows)
    csv_table(output/'qualification_events.csv',event_rows)
    summary=dict(schema='article.ground_replay.comparison.v2',stages=details,
        raw_packets_identical=True,frames_per_stage=161,rows=dict(frames=len(frame_rows),instances=len(instance_rows),qualification_events=len(event_rows)),
        original_return_boundary_paid_step=declaration['reporting_boundary_paid_step'],
        interpretation='Conditional association and belief mechanism along one original paid path; no action rescore, new World, trajectory, TSDF integration, or quality claim.',
        scope='Only observed marker anchors establish identity correspondence. Fields ending before_current_feedback are association snapshots; beliefs are after current feedback.',
        input_sha256=inputs,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    dump(output/'summary.json',summary)
    dump(output/'manifest.json',dict(schema='article.readonly.analysis.manifest.v1',
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()},
        predeclaration_sha256=sha(base/'predeclaration.json')))
    print(json.dumps(dict(output=str(output),stages=details),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=STAGE/'ground_association_replay_v2_attempt03')
    parser.add_argument('--output',type=Path,default=STAGE/'analysis_v1/ground_v2_attempt03')
    args=parser.parse_args();run(args.input.resolve(),args.output.resolve())
