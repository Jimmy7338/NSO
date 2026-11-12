#!/usr/bin/env python3
"""Explicit once-only post-terminal timing audit; no ROS, polling or acquisition."""
import argparse,json,pathlib,hashlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=pathlib.Path,required=True);p.add_argument('--can-ring',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
P=a.report
if not (P/'result.json').is_file():raise RuntimeError('Terminal root result required; no polling or partial-run interpretation')
load=lambda p:json.loads(p.read_text());result=load(P/'result.json');adapter=load(P/'adapter/result.json');side=P/'adapter/control_timing_fault.json';timing=load(side) if side.exists() else None
records=[json.loads(x) for x in (P/'adapter/events.jsonl').read_text().splitlines()];faultcommands=[x for x in records if x.get('event')=='command' and x.get('reason')==adapter.get('reason')];faultrows=[x for x in records if x.get('fault')];milestones=[load(f) for f in sorted((P/'cleanup_milestones').glob('*.json'))];stop=next((x for x in milestones if x['event']=='stop_request_begin'),None);signals=[x for x in milestones if x['event']=='signal_received'];rows=[] if timing is None else timing.get('rows',[]);reason=adapter.get('reason');idx=next((i for i,r in enumerate(rows) if r.get('outcome')==reason),None);current=None if idx is None else rows[idx];previous=None if idx is None or idx==0 else rows[idx-1]
def durations(r):
 if r is None:return None
 return {k[:-3]+'_ms':v/1e6 for k,v in r.items() if k.endswith('_ns') and k!='cycle_enter_ns' and isinstance(v,(int,float))}
first=faultcommands[0] if faultcommands else faultrows[0] if faultrows else None
firstunix=None if first is None else first.get('guard_stamp_ns',0)/1e9 if first.get('guard_stamp_ns') else first.get('kernel_unix_s')
can=None
if a.can_ring:
 ring=load(a.can_ring);samples=ring['samples'];t=firstunix
 if t is not None:
  pre=[x for x in samples if t-1<=x['kernel_stamp_ns']/1e9<=t];can=dict(samples_previous1s=len(pre),max_abs_v=None if not pre else max(abs(x['vx']) for x in pre),max_abs_w=None if not pre else max(abs(x['w']) for x in pre),max_source_interval_s=None if len(pre)<2 else max((y['kernel_stamp_ns']-x['kernel_stamp_ns'])/1e9 for x,y in zip(pre,pre[1:])))
known=[P/'result.json',P/'adapter/result.json',P/'adapter/events.jsonl']+([side] if side.exists() else [])+([a.can_ring] if a.can_ring else [])
out=dict(status='post_terminal_readonly_timing_audit',report=str(P),root_status=result.get('status'),root_reason=result.get('reason'),adapter_reason=reason,root_zero_after=result.get('feedback_zero_after_close'),timing_sidecar_present=timing is not None,fault_evidence=adapter.get('fault_evidence'),first_fault_record=first,first_fault_before_supervisor_stop_s=None if firstunix is None or stop is None else stop['unix_ns']/1e9-firstunix,current_fault_iteration=current,current_stage_ms=durations(current),previous_iteration=previous,previous_stage_ms=durations(previous),last_recorded_iteration=None if not rows else rows[-1],signal_receipts=signals,can_previous1s=can,interpretation=['Tick delta comes from first-fault evidence previous_tick/now; journal command rows are deduplicated and do not measure tick intervals.','Stage elapsed times describe where wall time was spent, not unique OS/GIL/lock root causes.','Current fault row and previous complete row are reported separately. Missing stages mean no measurement, not zero time.','adapter_shutdown/native_driver_fault8 may arise in ordinary shutdown; compare supervisor stop and received signals before attributing cause. A final sleep can include signal handling or shutdown interruption and must not be called a watchdog overrun unless first-fault evidence says so.','While-AST dry test validates row/sleep instrumentation and diagnostic I/O exception swallowing; production finally stop order was inspected in source, not executed by that test.'],input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in known})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['adapter_reason','timing_sidecar_present','fault_evidence','first_fault_before_supervisor_stop_s','current_stage_ms','previous_stage_ms']}))
