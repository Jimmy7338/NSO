"""Compatibility check on one already-completed B10 run; no new experiment."""
from pathlib import Path
import hashlib,json,time
from evaluate_full_run import Evaluator,resource,read,write,sha,audit_frames,audit_trace,first_correct_free_actions,HERE
BASE=Path('/home/ouc/nso_robot/analyses')
def main():
 started=time.monotonic();b10=BASE/'dual_template_gain_alignment_20261120';b=BASE/'dual_template_exploration_20261119'
 attempt=b10/'rollouts/native_novelty/layouts/shelf_left/runs/S_native_novelty/attempt_002'
 previous=b10/'policy_audit/completed_rollouts/results/shelf_left/S_native_novelty/attempt_002/result.json'
 expected=read(previous);e=Evaluator();reference=e.reference(b/'audit/references/shelf_left');native=e.native(resource(attempt/'final_tsdf_mesh.ply'),reference)
 assert native['metrics']==expected['metrics'],'Native metric implementation compatibility changed'
 r=read(attempt/'result.json');frames=read(attempt/'template_frame_receipts.json');final=read(attempt/'final_template_receipt.json');steps=read(attempt/'steps.json');calls=read(attempt/'template_audit.json')
 fa=audit_frames(frames,final,r['paid_actions'],len(calls));ka=audit_trace(steps,r['final_pose'],r)
 hits=first_correct_free_actions(read(attempt/'initial_evaluation.json')['free_correct_fraction'],steps)
 assert all(hits[k]==r['first_free_correct_action'][k]for k in hits)
 e.verify();write(HERE/'compatibility_receipt.json',dict(status='passed_existing_B10_native_metrics_exact_compatibility',old_result=resource(previous),old_attempt=str(attempt),all_nested_native_metrics_exact=True,frame_audit=fa,kinematic_audit=ka,A95_A100=hits,inputs=e.inputs,wall_s=time.monotonic()-started,new_actor_actions=0,new_B11_outcomes_evaluated=0))
 print(json.dumps(dict(status='passed',metrics_exact=True,frames=len(frames))))
if __name__=='__main__':main()
