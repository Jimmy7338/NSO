#!/usr/bin/env python3
"""Predeclared fixed-paid-path evidence replay: no World, mapper or new quality.

--freeze seals sources/constants/one reviewed input before either real replay.
--control must reproduce the frozen ArticleV1 saved evidence before --new.
All outputs are exclusive. No policy choose(), sensor query or TSDF is invoked.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from env.development_sensor_v41 import runtime_counts_v41
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.observed_instances_v41 import support_digest
from nso.observed_instances_article_ground_v2 import ObservedInstancesArticleGroundV2,GROUND_POLICY
from nso.controller_article_v1 import ArticleObservedResidualV1,ArticleViewPredictorV1
from nso.semantic_reliability import SemanticReliabilityBelief

STAGE=ROOT/'audit_results/article_stage_20260928'
DEFAULT=STAGE/'ground_association_replay_v2_attempt03'
RUN_ID='dev_AISLE_S_b160_n92801'
EPISODE=STAGE/'development_v1/episodes'/RUN_ID
REVIEW=STAGE/'episode_reviews_v1'/RUN_ID
PROTOCOL=ROOT/'configs/virtual3d/article_development_v1_20260928.json'
NEW_SOURCES=('nso/observed_instances_article_ground_v2.py',
             'tests/virtual3d/test_observed_instances_article_ground_v2.py',
             'scripts/replay_article_ground_association_20260928.py')
ATOL=1e-12


def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def read(path):return json.loads(Path(path).read_text())
def canonical(data):return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(path,data):
    with Path(path).open('xb') as stream:stream.write(canonical(data))
def require(condition,message):
    if not condition:raise ValueError(message)


def runtime_metadata():
    stream=io.StringIO()
    with redirect_stdout(stream):np.show_config()
    return dict(python_executable=sys.executable,python_version=sys.version,
        numpy_version=np.__version__,numpy_path=np.__file__,numpy_build=stream.getvalue(),
        scipy_version=scipy.__version__,scipy_path=scipy.__file__,
        thread_environment={name:os.environ.get(name) for name in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})


def source_pins():
    protocol=read(PROTOCOL)
    pins=dict(protocol['source_sha256'])
    pins.update({name:sha(ROOT/name) for name in NEW_SOURCES})
    for name,pin in protocol['source_sha256'].items():require(sha(ROOT/name)==pin,'frozen source differs: '+name)
    return pins


def freeze(base):
    require(not base.exists(),'exclusive new predeclaration directory required')
    review=read(REVIEW/'review.json');rm=read(REVIEW/'manifest.json')
    require(review['all_checks_passed'] and review['qualified'] and review['status']=='reviewed','reviewed qualified saved input required')
    manifest=read(EPISODE/'artifact_manifest.json')
    require(sha(EPISODE/'artifact_manifest.json')==review['input_manifest_sha256']==rm['input_episode_manifest_sha256'],'input seal/review binding')
    protocol=read(PROTOCOL)
    require(sha(PROTOCOL)==manifest['protocol_sha256']==review['protocol_sha256'],'frozen protocol binding')
    # Constants are sealed before any paid input is decoded or used by V2.
    pins=source_pins()
    input_paths=[PROTOCOL,EPISODE/'artifact_manifest.json',REVIEW/'manifest.json',REVIEW/'review.json']
    declaration=dict(schema='article.ground_fixed_path.predeclaration.v2',created_utc=datetime.now(timezone.utc).isoformat(),
        run_id=RUN_ID,episode=str(EPISODE.relative_to(ROOT)),review=str(REVIEW.relative_to(ROOT)),
        original_episode_manifest_sha256=sha(EPISODE/'artifact_manifest.json'),source_sha256=pins,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in input_paths},ground_policy=dict(GROUND_POLICY),
        numerical_runtime=runtime_metadata(),
        prior_attempts='attempt01/02 retained: Ubuntu .venv numerical backend failed strict old control; attempt03 uses the original .venv-3d backend without changing ground constants or tolerance.',
        original_source_archive=protocol['source_archive'],original_source_archive_sha256=protocol['source_archive_sha256'],
        control=dict(float_absolute_tolerance=ATOL,float_relative_tolerance=0.,
                     compared=['association','observed_residual','geometry_feedback','view_evidence','structure_belief'],
                     exact=['dict_keys','list_lengths','strings_except_verified_support_hash_correspondence','booleans','integers','packet_sha256','mask_hashes'],
                     support_hash_rule='Both hashes must recompute from complete stored/replayed point arrays; every coordinate <= 1e-12 m, identical ordering/shape. Only corresponding support_sha256 fields may be translated for control comparison; no old message is reused.'),
        revision='Only measured ground mask before object depth-component proposals; all ArticleV1 guards inherited.',
        packets='All 161 original paid AISLE_S packets, original frame IDs and exact original packet hashes.',
        belief_methods=['G','B','S'],common_history='One common geometry/association/residual history; G ignores class, B no sharing, S sharing.',
        identity_mapping=dict(rule='observed marker anchors only; unique bidirectional match within 0.25 m; no class/asset IDs',maximum_distance_m=.25),
        reporting_boundary_paid_step=126,boundary_scope='observed v1 return begins at 126; fixed-path diagnostic threshold, not new method stop time',
        required_reports=['all ground accept/fallback reasons','full association rejection reasons','identity ambiguity',
                          'qualified same-class peers','informative peer messages','valid feedback before 126','S/B posterior differences'],
        maximum_output_bytes=10*1024**2,
        prohibited=['new World','new sensor render/action','policy choose','TSDF reintegration','new quality evaluation','reuse old residual or feedback in V2'],
        interpretation='Conditional fixed-path evidence mechanism only; no completed new trajectory or improved-quality claim.',
        stage_order=['freeze','old control with exact original packets','new evidence only after successful control'])
    base.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(base/'new_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in NEW_SOURCES:archive.writestr(name,(ROOT/name).read_bytes())
    declaration['new_source_archive_sha256']=sha(base/'new_sources.zip')
    write(base/'predeclaration.json',declaration)
    print(json.dumps(dict(status='frozen_before_real_replay',output=str(base),source_files=len(pins),predeclaration_sha256=sha(base/'predeclaration.json'))))


def verify(base):
    declaration=read(base/'predeclaration.json')
    require(declaration['ground_policy']==dict(GROUND_POLICY),'ground policy differs from predeclaration')
    require(declaration['numerical_runtime']==runtime_metadata(),'runtime differs from predeclaration')
    require(declaration['source_sha256']==source_pins(),'current source closure differs')
    for name,pin in declaration['input_sha256'].items():require(sha(ROOT/name)==pin,'sealed input differs: '+name)
    require(sha(base/'new_sources.zip')==declaration['new_source_archive_sha256'],'new source archive differs')
    require(sha(ROOT/declaration['original_source_archive'])==declaration['original_source_archive_sha256'],'original source archive differs')
    manifest=read(EPISODE/'artifact_manifest.json')
    return declaration,manifest


def payload(manifest,name):
    path=EPISODE/name;row=manifest['files'][name];data=path.read_bytes()
    require(len(data)==row['bytes'] and sha_bytes(data)==row['sha256'],'input artifact differs: '+name)
    return data


def load_observation(manifest,step):
    with np.load(io.BytesIO(payload(manifest,f'packets/{step:03d}_rgbd.npz')),allow_pickle=False) as data:
        fields={name:data[name].copy() for name in data.files}
    fields['frame_id']=fields['frame_id'].item();fields['paid_step']=int(fields['paid_step'].item())
    observation=PaidRGBDObservationV40.from_mapping(fields)
    receipt=json.loads(payload(manifest,f'packets/{step:03d}_receipt.json'))
    require(observation.sha256()==receipt['observation_sha256'],'original physical packet logical SHA differs')
    return observation


class Compare:
    def __init__(self):self.leaves=0;self.max_abs=0.;self.failures=[]
    def record(self,path,message):
        if len(self.failures)<20:self.failures.append(dict(path=path,message=message))
    def check(self,left,right,path):
        if isinstance(left,dict) and isinstance(right,dict):
            if set(left)!=set(right):self.record(path,'dictionary keys differ');return
            for key in left:self.check(left[key],right[key],path+'/'+str(key))
        elif isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
            if len(left)!=len(right):self.record(path,'sequence lengths differ');return
            for i,(a,b) in enumerate(zip(left,right)):self.check(a,b,path+'/'+str(i))
        else:
            self.leaves+=1
            if type(left) is float and type(right) is float:
                delta=abs(left-right);self.max_abs=max(self.max_abs,delta)
                if not np.isfinite(delta) or delta>ATOL:self.record(path,'float absolute difference '+str(delta))
            elif type(left)!=type(right) or left!=right:self.record(path,'exact scalar differs')


def bind_support_hashes(actual,saved,step,correspondence,proofs):
    for collection in ('accepted','instances'):
        left,right=actual['association'][collection],saved['association'][collection]
        require(len(left)==len(right),'support correspondence requires equal instance/accepted counts')
        for a,b in zip(left,right):
            require(a['instance_id']==b['instance_id'],'support correspondence identity/order differs')
            array_key='points_world_m' if collection=='accepted' else 'support_points_world_m'
            x,y=np.asarray(a[array_key],dtype=float),np.asarray(b[array_key],dtype=float)
            require(x.shape==y.shape,'support array shape differs')
            error=float(np.max(np.abs(x-y))) if x.size else 0.
            require(error<=ATOL,'support coordinate comparison exceeds frozen tolerance')
            require(support_digest(x)==a['support_sha256'] and support_digest(y)==b['support_sha256'],
                    'advertised support hash does not bind its complete coordinate array')
            old,new=b['support_sha256'],a['support_sha256']
            require(old not in correspondence or correspondence[old]==new,'inconsistent historical support correspondence')
            correspondence[old]=new
            if old!=new:proofs.append(dict(paid_step=step,collection=collection,instance_id=a['instance_id'],
                original_support_sha256=old,recomputed_support_sha256=new,points=len(x),maximum_coordinate_error_m=error))


def translated_control_hashes(value,correspondence):
    if isinstance(value,dict):
        return {k:(correspondence.get(v,v) if k=='support_sha256' and isinstance(v,str)
                   else translated_control_hashes(v,correspondence)) for k,v in value.items()}
    if isinstance(value,list):return [translated_control_hashes(v,correspondence) for v in value]
    return value


class EvidencePipeline:
    def __init__(self,configuration,*,ground=False):
        cls=ObservedInstancesArticleGroundV2 if ground else ObservedInstancesArticleV1
        self.ledger=cls(palette=configuration['palette'],structure_names=configuration['structure_names'],
            class_structure_prior=configuration['class_structure_prior'],geometry_prior=configuration['geometry_prior'],
            maximum_instances=configuration['maximum_instances'],mode='S')
        self.residual=ArticleObservedResidualV1(multi_view_planes=True)
        self.predictor=ArticleViewPredictorV1(residual=self.residual,maximum_instances=configuration['maximum_instances'])
        self.beliefs={method:SemanticReliabilityBelief(geometry_prior=self.ledger.geometry_prior,
            class_structure_priors=self.ledger.class_priors,share_across_instances=method=='S') for method in ('G','B','S')}
    def accept(self,observation):
        association=self.ledger.observe(observation)
        residual=self.residual.observe(observation,association['accepted'])
        feedback=[self.ledger.apply_geometry_feedback(row['instance_id'],frame_id=observation.frame_id,
            observation_sha256=observation.sha256(),log_likelihoods=row['log_likelihoods'])
            for row in residual['results'] if row['accepted']]
        views=self.predictor.observe(observation,association['accepted'])
        instances=self.ledger.snapshot()['instances']
        for method,belief in self.beliefs.items():
            for instance in instances:
                key=instance['instance_id'];label=instance['observed_class'] if (
                    method!='G' and instance['semantic_conditioning_used'] and not instance['association_uncertain']) else None
                if key not in belief.instance_ids:belief.register(key,label)
                else:belief.set_class(key,label)
                belief.replace_log_evidence(key,instance['geometry_log_scores'])
        return dict(association=association,observed_residual=residual,geometry_feedback=feedback,
                    view_evidence=views,structure_belief=self.beliefs['S'].snapshot(),
                    all_method_beliefs={k:v.snapshot() for k,v in self.beliefs.items()})


def summary_instances(rows):
    results=[]
    for row in rows:
        item={k:v for k,v in row.items() if k!='support_points_world_m'}
        points=np.asarray(row['support_points_world_m'],dtype=float)
        item.update(support_points=len(points),near_ground_support_points_z_below_015m=int((points[:,2]<.15).sum()),
                    support_bbox_m=np.stack((points.min(0),points.max(0))).tolist() if len(points) else None)
        results.append(item)
    return results


def compact_evidence(evidence):
    result=deepcopy(evidence)
    result['association']['instances']=summary_instances(result['association']['instances'])
    for row in result['association']['accepted']:
        points=np.asarray(row.pop('points_world_m'));pixels=np.asarray(row.pop('pixel_indices'),dtype=np.int64)
        row.update(associated_points=len(points),associated_pixel_indices_sha256=sha_bytes(pixels.tobytes()))
    for row in result['geometry_feedback']:
        row['instance']=summary_instances([row['instance']])[0]
    return result


def identity_mapping(new,old):
    distances={(a['instance_id'],b['instance_id']):float(np.linalg.norm(np.asarray(a['anchor_world_m'])-b['anchor_world_m']))
               for a in new for b in old}
    result=[]
    for a in new:
        candidates=[b['instance_id'] for b in old if distances[(a['instance_id'],b['instance_id'])]<=.25]
        unique=None
        if len(candidates)==1:
            b=candidates[0]
            if sum(distances[(x['instance_id'],b)]<=.25 for x in new)==1:unique=b
        result.append(dict(new_instance_id=a['instance_id'],matched_old_instance_id=unique,
            candidate_old_instance_ids=candidates,status='unique_observed_anchor' if unique else 'unmatched_or_ambiguous',
            distance_m=None if unique is None else distances[(a['instance_id'],unique)]))
    return result


def run(base,*,ground):
    declaration,manifest=verify(base)
    if ground:require(read(base/'control/summary.json')['status']=='passed','old control must pass before new evidence replay')
    out=base/('new_ground' if ground else 'control');out.mkdir(exist_ok=False)
    write(out/'started.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        predeclaration_sha256=sha(base/'predeclaration.json'),new_ground=ground))
    before=runtime_counts_v41();comparison=Compare();records=[];mappings=[];ground_masks=[]
    support_correspondence={};support_proofs=[]
    all_reasons=Counter();ground_reasons=Counter();counters=Counter();first={};per_instance={};status='running';error=None
    try:
        configuration=json.loads(payload(manifest,'controller_final.json'))['configuration']
        pipeline=EvidencePipeline(configuration,ground=ground)
        total=json.loads(payload(manifest,'result.json'))['acquired_and_saved_packets']
        for step in range(total):
            observation=load_observation(manifest,step);original_sha=observation.sha256()
            saved=json.loads(gzip.decompress(payload(manifest,f'steps/{step:03d}.json.gz')))['controller_evidence']
            evidence=pipeline.accept(observation)
            require(observation.sha256()==original_sha,'raw packet mutated')
            if not ground:
                bind_support_hashes(evidence,saved,step,support_correspondence,support_proofs)
                for key in ('association','observed_residual','geometry_feedback','view_evidence','structure_belief'):
                    comparison.check(evidence[key],translated_control_hashes(saved[key],support_correspondence),str(step)+'/'+key)
                if comparison.failures:raise ValueError('old control differs from saved evidence')
            else:
                ground_receipt=evidence['association']['article_ground_association']
                ground_reasons[ground_receipt['reason']]+=1
                counters['ground_accepted_frames']+=ground_receipt['accepted']
                counters['excluded_association_pixels']+=ground_receipt['excluded_pixels']
                counters['valid_depth_pixels']+=int(((observation.depth_m>=.1)&(observation.depth_m<=4.)).sum())
                ground_masks.append(pipeline.ledger._current_ground_mask.copy())
            rows=evidence['association']['instances'];all_reasons.update(r['reason'] for r in evidence['association']['rejected'])
            mapping=identity_mapping(rows,saved['association']['instances']);mappings.append(dict(paid_step=step,matches=mapping))
            counters['unmatched_or_ambiguous_instance_frames']+=sum(r['status']!='unique_observed_anchor' for r in mapping)
            counters['uncertain_instance_frames']+=sum(r['association_uncertain'] for r in rows)
            counters['class_conflict_instance_frames']+=sum(r['class_conflict'] for r in rows)
            for feedback in evidence['geometry_feedback']:
                if feedback['applied']:
                    counters['applied_feedback']+=1;first.setdefault('applied_feedback',step)
                    if step<126:counters['applied_feedback_before_v1_return']+=1
                    per_instance.setdefault(feedback['instance_id'],{}).setdefault('first_feedback',step)
            for row in rows:
                firsts=per_instance.setdefault(row['instance_id'],{})
                firsts.setdefault('first_observed',step)
                if row['semantic_conditioning_used']:firsts.setdefault('first_qualified_class',step)
            for key,plane in pipeline.predictor.observed_plane_receipts().items():
                if plane['plane_fit'] is not None and not plane['conflict']:
                    first.setdefault('reliable_plane',step);per_instance.setdefault(key,{}).setdefault('first_reliable_plane',step)
            sb={m:{r['instance_id']:r for r in evidence['all_method_beliefs'][m]['instances']} for m in ('G','B','S')}
            for key,row in sb['S'].items():
                if row['peer_instance_ids']:
                    first.setdefault('qualified_same_class_peer',step)
                    if any(np.ptp(sb['S'][peer]['geometry_log_evidence'])>ATOL for peer in row['peer_instance_ids']):
                        first.setdefault('informative_same_class_peer',step)
                delta=max(abs(x-y) for x,y in zip(row['structure_probabilities'],sb['B'][key]['structure_probabilities']))
                if delta>ATOL:
                    first.setdefault('S_B_structure_posterior_difference',step)
                    counters['S_B_changed_instance_frames']+=1
                    if step<126:counters['S_B_changed_instance_frames_before_v1_return']+=1
            records.append(dict(paid_step=step,observation_sha256=original_sha,**compact_evidence(evidence)))
            if step%20==0:print(json.dumps(dict(stage='new_ground' if ground else 'old_control',paid_step=step,first=first)),flush=True)
        status='passed'
    except Exception as exc:
        status='failed';error=dict(type=type(exc).__name__,message=str(exc))
    after=runtime_counts_v41()
    require(before==after,'offline replay created a sensor World or action')
    require(verify(base)[0]==declaration,'freeze changed during replay')
    summary=dict(schema='article.ground_fixed_path.result.v2',status=status,error=error,
        stage='ground_v2_fixed_path' if ground else 'article_v1_control',run_id=RUN_ID,processed_frames=len(records),
        predeclaration_sha256=sha(base/'predeclaration.json'),source_sha256=declaration['source_sha256'],
        comparison=dict(leaves=comparison.leaves,maximum_absolute_float_error=comparison.max_abs,failures=comparison.failures),
        verified_support_hash_correspondences=len(support_correspondence),
        support_coordinate_roundoff_records=len(support_proofs),
        first_steps=first,per_instance_first_steps=per_instance,counters=dict(counters),
        association_rejection_reasons=dict(all_reasons),ground_reasons=dict(ground_reasons),
        runtime_before=before,runtime_after=after,new_worlds=0,new_sensor_queries=0,new_policy_runs=0,
        new_tsdf_integrations=0,new_quality_evaluations=0,
        limits='Fixed original paid path. No candidate rescoring or hypothetical action selection yet; no new trajectory or reconstruction advantage claim.')
    with (out/'evidence.json.gz').open('xb') as stream:stream.write(gzip.compress(canonical(records),mtime=0))
    with (out/'observed_anchor_matches.json.gz').open('xb') as stream:stream.write(gzip.compress(canonical(mappings),mtime=0))
    with (out/'support_coordinate_correspondence.json.gz').open('xb') as stream:stream.write(gzip.compress(canonical(support_proofs),mtime=0))
    if ground_masks:
        with (out/'ground_masks.npz').open('xb') as stream:np.savez_compressed(stream,masks=np.packbits(np.stack(ground_masks),axis=-1),shape=np.array(np.stack(ground_masks).shape))
    write(out/'summary.json',summary)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
    write(out/'manifest.json',dict(schema='article.ground_fixed_path.output_manifest.v2',files=files,
        predeclaration_sha256=sha(base/'predeclaration.json'),source_episode_manifest_sha256=declaration['original_episode_manifest_sha256']))
    require(sum(p.stat().st_size for p in base.rglob('*') if p.is_file())<=declaration['maximum_output_bytes'],'output budget exceeded')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if status!='passed':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--control',action='store_true');group.add_argument('--new',action='store_true')
    args=parser.parse_args();base=args.output.resolve()
    require(base.is_relative_to(STAGE) and not base.is_relative_to(EPISODE),'external analysis output under article stage required')
    if args.freeze:freeze(base)
    else:run(base,ground=args.new)


if __name__=='__main__':main()
