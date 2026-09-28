#!/usr/bin/env python3
"""Read sealed old trajectories and static class/structure metadata only.

No World, rendering, mapping, policy branch or surface-quality evaluation.
Private category/structure metadata is used only for offline model explanation;
no observed instance ID is assumed to equal a physical GT object ID.
"""
from collections import Counter
from datetime import datetime,timezone
from itertools import product
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'audit_results/article_stage_20260928'
OUT=ROOT/'docs/research/ARTICLE_SEMANTIC_MODEL_SCOPE_20260928.analysis.json'


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def normalize(prior,evidence):
    x=np.asarray(prior)*np.exp(np.asarray(evidence)-max(evidence))
    return x/x.sum()
def tv(a,b):return float(.5*np.abs(np.asarray(a)-np.asarray(b)).sum())


def analyze():
    protocol_path=ROOT/'configs/virtual3d/article_development_v1_20260928.json'
    protocol=read(protocol_path);pins={str(protocol_path.relative_to(ROOT)):sha(protocol_path)}
    assets=ROOT/protocol['asset_root'];manifest=read(assets/'manifest.json')
    require(sha(assets/'manifest.json')==protocol['asset_manifest_sha256'],'asset manifest pin')
    pins[str((assets/'manifest.json').relative_to(ROOT))]=sha(assets/'manifest.json')
    table=[];priors=None;structures=None
    for split,family in product(('DEV','T0','T1'),('AISLE','CELL','LOOP')):
        scene='ART1_'+family+'_'+split
        for suffix in ('public_planner_spec.json','evaluation_private/instances.json'):
            relative=scene+'/'+suffix;path=assets/relative
            require(sha(path)==manifest['artifact_sha256'][relative],'static metadata pin: '+relative)
            pins[str(path.relative_to(ROOT))]=sha(path)
        spec=read(assets/scene/'public_planner_spec.json')
        if priors is None:
            priors=spec['structure_prior']['probability_by_category'];structures=spec['structure_prior']['abstract_structures']
        require(priors==spec['structure_prior']['probability_by_category'],'one common public prior table')
        metadata=read(assets/scene/'evaluation_private/instances.json')
        instances=[{k:row[k] for k in ('instance_id','category','structure','structure_quantile',
            'relationship_shift','recognition_corruption')} for row in metadata['private_instances']]
        table.append(dict(scene_id=scene,instances=instances,
            category_structure_multiset=sorted((r['category'],r['structure']) for r in instances)))
    algebra=[];g=np.full(4,.25)
    for label,values in sorted(priors.items()):
        q=np.array(values);b=(g+q)/2
        hs=[int(np.searchsorted(np.cumsum(q),u,side='right')) for u in (.25,.75)]
        rows=[]
        for target,peer in (hs,hs[::-1]):
            ratio=q[peer]/g[peer];rho=ratio/(1+ratio);active=(1-rho)*g+rho*q
            rows.append(dict(target_structure=structures[target],perfectly_known_peer_structure=structures[peer],
                rho=rho,B_active_true_structure_weight=b[target],S_active_true_structure_weight=active[target],
                S_minus_B_true_structure_weight=active[target]-b[target],active_prior_TV=tv(active,b),
                target_has_own_evidence=False,limit_is_not_an_executed_posterior=True))
        # Paired condition under ideal calibrated likelihoods, not actual CPU evidence.
        cond=(.5*np.outer(g,g)+.5*np.outer(q,q))
        expected_brier_gain=0.;covariance_error=0.
        for peer in range(4):
            conditional=cond[:,peer]/cond[:,peer].sum()
            covariance_error=max(covariance_error,float(np.max(np.abs(
                cond-np.outer(b,b)-.25*np.outer(q-g,q-g)))))
            expected_brier_gain+=float(b[peer]*np.sum((conditional-b)**2))
        rho_pair=float(np.prod(q[hs]/g[hs])/(1+np.prod(q[hs]/g[hs])))
        # Expected posterior prior-shift TV after one perfectly observed peer,
        # under the *assumed mixture model*, not fixed quantile scenes.
        mean_tv=sum(float(b[h])*tv((g[h]*g+q[h]*q)/(g[h]+q[h]),b) for h in range(4))
        algebra.append(dict(category=label,q=q.tolist(),g=g.tolist(),B_mixture=b.tolist(),
            deterministic_quantile_structures=[structures[h] for h in hs],perfect_peer_limits=rows,
            posterior_Z_after_both_true_structures=rho_pair,
            mixture_model_expected_squared_probability_improvement=expected_brier_gain,
            mixture_model_expected_active_prior_TV=mean_tv,
            covariance_identity_max_abs_error=covariance_error))
    phase=BASE/'development_v1';ledger_raw=(phase/'start_ledger.json').read_bytes();ledger=json.loads(ledger_raw)
    pins[str((phase/'start_ledger.json').relative_to(ROOT))]=sha_bytes(ledger_raw)
    runs={};actions={};packets={};same_path=[]
    for family,method in product(('AISLE','CELL','LOOP'),('G','B','S')):
        run=f'dev_{family}_{method}_b160_n92801';entry=next(r for r in ledger['entries'] if r['run_id']==run)
        if entry['status']!='controller_stop':
            runs[run]=dict(status=entry['status'],analyzed=False,reason='not a complete successfully sealed trajectory; no outcome imputation')
            continue
        episode=phase/'episodes'/run;manifest_path=episode/'artifact_manifest.json'
        require(sha(manifest_path)==entry['artifact_manifest_sha256'],'terminal artifact manifest pin')
        m=read(manifest_path);pins[str(manifest_path.relative_to(ROOT))]=sha(manifest_path)
        files=sorted(name for name in m['files'] if name.startswith('steps/') and name.endswith('.json.gz'))
        states=[];actions[run]=[];packets[run]=[];counts=Counter();return_step=None
        for name in files:
            raw=(episode/name).read_bytes();row=m['files'][name]
            require(len(raw)==row['bytes'] and sha_bytes(raw)==row['sha256'],'sealed step hash')
            step=json.loads(gzip.decompress(raw));d=step['decision'];paid=d['paid_step']
            actions[run].append(d['action']);packets[run].append(d['source_observation_sha256'])
            if d.get('routing',{}).get('reason')=='return' and return_step is None:return_step=paid
            selection=d.get('global_selection') or {}; selected=selection.get('selected')
            actual_replan=bool(d.get('global_replanned'))
            belief=step['controller_evidence']['structure_belief']
            for instance in belief['instances']:
                label=instance['observed_class'];geometry=instance['geometry_log_evidence']
                q=np.array(priors[label]) if label is not None else g
                prior_b=(g+q)/2
                p_b=normalize(prior_b,geometry);p_g=normalize(g,geometry)
                own=normalize(instance['active_structure_prior'],geometry)
                require(np.allclose(own,instance['structure_probabilities'],atol=1e-12,rtol=0.),'saved posterior reconstruction')
                count=len(instance['peer_instance_ids'])
                record=dict(paid_step=paid,instance_id=instance['instance_id'],label=label,
                    peer_count=count,rho=instance['rho'],active_prior_TV_vs_fixed_B=tv(instance['active_structure_prior'],prior_b),
                    posterior_TV_vs_fixed_B=tv(own,p_b),posterior_TV_vs_G=tv(own,p_g),
                    geometry_informative=float(np.ptp(geometry))>1e-12,actual_replan=actual_replan,
                    selected_kind=None if selected is None else selected['kind'])
                states.append(record)
                if label is not None:counts['qualified_class_instance_frames']+=1
                if count:counts['same_class_peer_instance_frames']+=1
                if record['posterior_TV_vs_fixed_B']>1e-12:counts['posterior_differs_from_fixed_B_instance_frames']+=1
        before=lambda r:return_step is None or r['paid_step']<return_step
        actionable=[r for r in states if before(r) and r['actual_replan'] and r['selected_kind'] in ('direct','diagnose_then_observe')]
        semantic=[r for r in states if r['label'] is not None]
        def greatest(rows,key):return max(rows,key=lambda r:r[key]) if rows else None
        runs[run]=dict(status=entry['status'],analyzed=True,frames=len(files),first_return_step=return_step,
            counts=dict(counts),maximum_same_class_peers=max((r['peer_count'] for r in states),default=0),
            qualified_rho_range=[min(r['rho'] for r in semantic),max(r['rho'] for r in semantic)] if semantic else None,
            max_active_prior_shift=greatest(states,'active_prior_TV_vs_fixed_B'),
            max_posterior_shift=greatest(states,'posterior_TV_vs_fixed_B'),
            max_pre_return_selected_replan_posterior_shift=greatest(actionable,'posterior_TV_vs_fixed_B'),
            first_nonzero_shift_step=min((r['paid_step'] for r in states if r['posterior_TV_vs_fixed_B']>1e-12),default=None),
            no_GT_instance_id_matching=True)
    for family in ('AISLE','CELL','LOOP'):
        for left,right in (('G','B'),('B','S')):
            a=f'dev_{family}_{left}_b160_n92801';b=f'dev_{family}_{right}_b160_n92801'
            same_path.append(dict(scene=family,left=left,right=right,available=a in actions and b in actions,
                actions_identical=actions[a]==actions[b] if a in actions and b in actions else None,
                paid_packet_hashes_identical=packets[a]==packets[b] if a in packets and b in packets else None))
    for path in (ROOT/'nso/semantic_reliability.py',ROOT/'nso/controller_semantic_mechanism.py',
                 ROOT/'nso/article_scene_assets_v1.py',Path(__file__),BASE/'diagnostics/shared_prior_capacity.json',
                 BASE/'diagnostics/direct_oracle_opportunity_v1/summary.json'):
        pins[str(path.relative_to(ROOT))]=sha(path)
    result=dict(schema='article.semantic_model_scope.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Sealed old ArticleV1 observations plus offline static category/structure metadata and algebra; no Ground outcome claim',
        source_sha256=pins,static_scene_table=table,perfect_peer_algebra=algebra,recorded_run_summaries=runs,
        actual_pair_path_identity=same_path,original_ledger_statuses={r['run_id']:r['status'] for r in ledger['entries']},
        geometry_and_test_image_assets_read=False,GT_only_for_offline_explanation=True,
        private_information_passed_to_controller=False,new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,
        new_surface_evaluations=0,actual_GT_instance_to_observed_identity_assumed=False,
        limits=['Perfect-peer calculations concern the transferred prior before target evidence, not TSDF performance.',
            'Mixture-model expectation assumes calibrated independent evidence and a genuinely shared latent class domain.',
            'Actual geometry scores are generalized evidence, not calibrated likelihoods.',
            'All arithmetic is descriptive; no new scene, tuning, main execution or positive-effect guarantee.'])
    with OUT.open('x') as stream:json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(OUT),scenes=len(table),analyzed_runs=sum(r['analyzed'] for r in runs.values()),
        perfect_peer_algebra=algebra,recorded_run_summaries=runs,actual_pair_path_identity=same_path),ensure_ascii=False,indent=2))


if __name__=='__main__':analyze()
