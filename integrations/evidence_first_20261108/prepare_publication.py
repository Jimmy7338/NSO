#!/usr/bin/env python3
"""Stage evidence after terminal/freeze; no ROS, motion or remote writes."""
import argparse,gzip,hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--report',required=True,type=pathlib.Path);ap.add_argument('--output',required=True,type=pathlib.Path);a=ap.parse_args();R=a.report.resolve();O=a.output.resolve()
if O.exists():raise FileExistsError(O)
assert (R/'result.json').is_file() and (R/'summary.json').is_file(),'episode not terminal'
Q=P/'theory/Q2_execution';results=sorted(Q.glob('Q2_*_result.json'));assert len(results)==10,'Q2 needs ten terminal conditions'
O.mkdir(parents=True);files=[]
def stage(src,rel,compress=False):
 src=pathlib.Path(src);dst=O/rel;dst.parent.mkdir(parents=True,exist_ok=True);data=src.read_bytes()
 if compress:dst.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
 else:dst.write_bytes(data)
 b=dst.read_bytes();files.append(dict(local=str(dst),remote='integrations/evidence_first_20261108/'+str(dst.relative_to(O)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),original_sha256=hashlib.sha256(data).hexdigest(),git_sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),gzip=compress))
for n in ['theory_audit.json','controlled_plan.json','manifest.json','arithmetic_verification.json','verify_score_identity.py','execute_fixed_Q2.py','analyze_fixed_Q2.py']:
 f=P/'theory'/n
 if f.exists():stage(f,'theory/'+n)
for f in sorted(Q.iterdir()):
 if f.is_file() and (f.suffix in ('.json','.csv','.py')) and f.name!='heartbeat.json':
  compress=f.stat().st_size>250000;stage(f,'theory/Q2_execution/'+f.name+('.gz' if compress else ''),compress)
for n in ['interface_audit.md','fit_floor_profile.py','mapper_floor_counterfactual.py','trace_changed_cells.py','verification.json','frozen_profile.json','mapper_counterfactual.json','changed_cell_sources.json','changed_cell_summary.json','capture_ground_reference.py','current_capture/protocol.json','current_capture/receipt.json','current_capture/visual_ROI_review.json','roi_replay/frame_0.npz','roi_replay/frame_1.npz','roi_replay/frame_2.npz','roi_replay/manifest.json','roi_replay/verification.json']:
 f=P/'sensor'/n
 if f.exists():stage(f,'sensor/'+n)
for n in ['protocol.json','prepare_capture.py','run_with_fault_ring.py','pre_execution_comparison.json','fresh/configuration.json','fresh/source.json','run/observer_result.json']:
 f=P/'legacy_trial'/n
 if f.exists():stage(f,'legacy_trial/'+n)
for n in ['summary.json','joint_configuration.json','runtime_parameters.json','source_manifest.json','joint_navigation_contract.json','experiment_receipt.json','running.json']:
 if (R/n).exists():stage(R/n,'legacy_trial/report/'+n)
for f in P.glob('legacy_trial/*summary*.json'):stage(f,'legacy_trial/'+f.name)
# Preserve hashes of the full original logs/packets locally; no claim of full replay.
raw={str(f.relative_to(R)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in [R/'result.json',R/'joint/decisions.jsonl'] if f.exists()}
(O/'legacy_trial/source_hashes.json').write_text(json.dumps({'scope':'full originals retained locally; no full sensor/decision replay asserted','files':raw},indent=2)+'\n');f=O/'legacy_trial/source_hashes.json';b=f.read_bytes();files.append(dict(local=str(f),remote='integrations/evidence_first_20261108/legacy_trial/source_hashes.json',bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
manifest={'parent_expected':'329cfc74b151992d92d4555c67317aca5d8abd81','files':files,'total_bytes':sum(x['bytes'] for x in files),'limits':'Derived ROI is not complete sensor replay. Gzip originals preserve full choices/ledgers. State/doc snapshots added only after root freeze.'}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'files':len(files),'bytes':manifest['total_bytes']}))
