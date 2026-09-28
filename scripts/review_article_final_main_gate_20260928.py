#!/usr/bin/env python3
"""Independent final necessary-condition review; never launches or measures."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'audit_results/article_stage_20260928'
PINS={}


def read(path,pin=None):
    payload=path.read_bytes();actual=hashlib.sha256(payload).hexdigest()
    if pin is not None and actual!=pin:raise ValueError('SHA mismatch '+str(path))
    PINS[str(path.relative_to(ROOT))]=dict(sha256=actual,bytes=len(payload))
    return payload


def obj(path,pin=None):return json.loads(read(path,pin))


def folder(path):
    manifest=obj(path/'manifest.json')
    for name,record in manifest['files'].items():
        payload=read(path/name,record['sha256']);assert len(payload)==record['bytes']
    sources=manifest.get('reviewer_source_sha256',{})
    if isinstance(sources,str):sources={'scripts/review_article_episode_20260928.py':sources}
    for name,pin in sources.items():read(ROOT/name,pin)
    return manifest


def main(output):
    assert not output.exists()
    snapshot=STAGE/'analysis_v1/exposure12_complete';sm=folder(snapshot)
    for name,pin in sm['source_sha256'].items():read(ROOT/name,pin)
    summary=obj(snapshot/'summary.json');assert summary['findings']==0 and obj(snapshot/'findings.json')==[]
    gate=obj(snapshot/'main_gate_witnesses.json')
    declaration=gate['declaration'];read(ROOT/declaration['path'],declaration['sha256'])
    pairs=list(csv.DictReader(io.StringIO(read(snapshot/'paired_effects.csv').decode())))
    pairs={r['scene_id']:r for r in pairs if r['comparison']=='ExposureV3/S-B'}
    witnesses={r['scene_id']:r for r in gate['pairs'] if r['comparison']=='ExposureV3/S-B'}
    assert set(pairs)==set(witnesses)=={'ART1_AISLE_DEV','ART1_CELL_DEV','ART1_LOOP_DEV'}
    rows=[];arms={};cohort={};results={};manifests={};common=[];frozen_sources={}
    for phase,config,review_root in (
        ('development_v1','article_development_v1_20260928.json','episode_reviews_v1'),
        ('ground_ablation_v2','article_ground_ablation_v2_20260928.json','episode_reviews_ground_v2'),
        ('exposure_ablation_v3','article_exposure_ablation_v3_20260928.json','episode_reviews_exposure_v3')):
        ledger_path=STAGE/phase/'start_ledger.json';ledger=obj(ledger_path)
        protocol=obj(ROOT/'configs/virtual3d'/config,ledger['protocol_sha256'])
        assert ledger['slots']==protocol['slots'] and len(ledger['entries'])==len(protocol['slots'])==12
        assert {e['run_id'] for e in ledger['entries']}==set(protocol['slots'])
        assert protocol['status']=='frozen'
        archive=read(ROOT/protocol['source_archive'],protocol['source_archive_sha256'])
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(protocol['source_sha256'])
            for name,pin in protocol['source_sha256'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==pin
                read(ROOT/name,pin)
        frozen_sources[phase]=dict(source_count=len(protocol['source_sha256']),archive_sha256=protocol['source_archive_sha256'])
        phase_rows=[]
        for entry in ledger['entries']:
            run=entry['run_id'];assert entry['status']!='reserved' and entry.get('finished_unix_s') is not None
            review_folder=STAGE/review_root/run;rm=folder(review_folder);review=obj(review_folder/'review.json')
            assert review['run_id']==run and review['online_status']==entry['status'] and review['all_checks_passed']
            episode=STAGE/phase/'episodes'/run
            if entry['status']=='attempt_failed':
                assert run=='dev_CELL_G_b160_n92801' and review['status']=='failed_attempt_preserved'
                assert entry.get('qualified',False) is False and review['qualified'] is False
                failure=episode/'attempt_failure.json'
                if not failure.exists():failure=STAGE/phase/'failures'/(run+'.json')
                assert obj(failure,entry['result_sha256'])==review['failure']
            else:
                assert entry['status']=='controller_stop' and entry['qualified'] and review['qualified']
                assert review['status']=='reviewed'
                manifest=obj(episode/'artifact_manifest.json',entry['artifact_manifest_sha256'])
                assert rm['input_episode_manifest_sha256']==entry['artifact_manifest_sha256']
                result=obj(episode/'result.json',entry['result_sha256'])
                assert manifest['files']['result.json']['sha256']==entry['result_sha256']
                results[run]=result;manifests[run]=manifest
            row=dict(run_id=run,phase=phase,status=entry['status'],original_qualified=entry.get('qualified',False),
                independent_review_status=review['status'],review_checks_passed=review['all_checks_passed'])
            rows.append(row);phase_rows.append(row);cohort[run]=row
            if phase=='development_v1':
                supplement=STAGE/'common_evaluation_v1'/run
                folder(supplement/'measurement');measurement=obj(supplement/'measurement/result.json')
                assert measurement['original_end_to_end_status']==entry['status']
                assert measurement['original_end_to_end_qualified']==entry.get('qualified',False)
                assert measurement['motion_completion_verified'] and measurement['quality_measurement_available']
                assert measurement['measurement_version']=='article.common_numeric_face_evaluation.v1'
                common.append(dict(run_id=run,mode=measurement['mode'],original_qualified=entry.get('qualified',False)))
                if entry['status']=='attempt_failed':
                    folder(supplement/'prepared');capture=obj(supplement/'prepared/input_snapshot.json')
                    assert capture['forensic_capture_after_terminal_failure'] and not capture['original_artifact_manifest_available']
                    assert capture['original_end_to_end_status']=='attempt_failed' and not capture['original_end_to_end_qualified']
                    assert measurement['mode']=='new_derived_measurement' and measurement['original_failed_attempt_remains_failed']
                    failed=dict(ledger=row,original_message=review['failure']['message'],
                        original_artifact_manifest_available=False,forensic_capture_after_terminal_failure=True,
                        motion_completion_verified=True,measurement_mode=measurement['mode'],
                        original_qualified=False,derived_J_nav=measurement['metrics']['J_nav'])
        arms[phase]=dict(attempts=12,terminal_status_counts=dict(Counter(r['status'] for r in phase_rows)),
            independent_reviews_passed=sum(r['review_checks_passed'] for r in phase_rows))
    evidence=[]
    for scene,pair in sorted(pairs.items()):
        witness=witnesses[scene]
        assert witness['status']=='no_actual_action_divergence'
        assert witness['common_actual_prefix_frames']==161 and witness['first_actual_action_divergence_paid_step'] is None
        assert pair['actions_identical']=='True' and int(pair['common_observation_prefix_frames'])==161
        assert int(pair['common_selected_target_differences'])==0 and float(pair['delta_J_nav'])==0.
        left,right=pair['left_run'],pair['right_run']
        # Independent check of every executed action, without policy replay.
        actions={run:[a['sensor_action'] for a in results[run]['actions']] for run in (left,right)}
        assert len(actions[left])==len(actions[right])==160 and actions[left]==actions[right]
        for method,run in (('S',left),('B',right)):
            assert witness['side_status'][method]['artifact_manifest_sha256']==PINS[str((STAGE/'exposure_ablation_v3/episodes'/run/'artifact_manifest.json').relative_to(ROOT))]['sha256']
        evidence.append(dict(scene_id=scene,left_run=left,right_run=right,physical_prefix_frames_from_sealed_snapshot=161,
            independently_checked_action_pairs=160,all_executed_actions_equal=True,selected_target_differences=0,
            first_actual_action_divergence=None,first_changed_posterior=pair['first_changed_posterior_paid_step'] or None,
            first_changed_score=pair['first_changed_score_paid_step'] or None,
            changed_common_candidate_scores=int(pair['changed_common_candidate_scores']),delta_J_nav=0.))
    live_ledgers=[]
    for path in sorted(STAGE.rglob('start_ledger.json')):
        ledger=obj(path);live_ledgers.append(dict(path=str(path.relative_to(ROOT)),phase=ledger['phase'],entries=len(ledger['entries'])))
    assert len(live_ledgers)==3 and sum(x['entries'] for x in live_ledgers)==36
    assert not any(x['phase']=='main' for x in live_ledgers)
    verdict=dict(schema='article.final_main_gate_verdict.v1',gate='necessary_action_mechanism_condition_not_met',
        main48_status='not_executed',main48_results=None,main48_slots_executed=0,
        automatic_main_launch=False,additional_development_rounds_authorized=0,
        reason='All three Exposure S/B pairs have identical complete executed action sequences; necessary causal action divergence is absent.',
        declaration=declaration,cohorts=arms,attempts=rows,frozen_execution_archives=frozen_sources,
        common_evaluation_modes=dict(Counter(x['mode'] for x in common)),original_failed_attempt_preserved=failed,
        exposure_SB_evidence=evidence,ledger_inventory=live_ledgers,
        verification_scope='Sealed snapshot outputs and analysis sources; all three original execution archives/live-source pins; 36 terminal ledgers and independent review artifacts; all 12 common supplements; independent comparison of all 480 executed S/B action pairs. Exact RGBD/scan equality is inherited from the verified sealed full analysis, not newly recomputed here.',
        unused_counterfactuals='No nonexistent divergent macro budget/latch counterfactuals constructed.',
        new_worlds=0,new_policy_runs=0,new_quality_evaluations=0,new_tsdf_integrations=0)
    output.mkdir(parents=True)
    source=Path(__file__).resolve();read(source);(output/'source.py').write_bytes(source.read_bytes())
    (output/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+'\n')
    outfiles={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in output.iterdir()}
    (output/'pins.json').write_text(json.dumps(dict(schema='article.final_gate_pins.v1',inputs=PINS,outputs=outfiles),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(output),attempts=36,shared_action_pairs=480,verdict=verdict['gate'],bytes=sum(p.stat().st_size for p in output.iterdir()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
