"""Validate earliest eligible selection and export immutable prefix overrides."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
scene=read(ROOT/'scene/SCENE_PROTOCOL.json');frozen=read(ROOT/'scene/freeze.json');overrides={};attempts=[];sources=[]
for layout in ['chain_a','chain_b']:
 path=HERE/layout/'result.json';r=read(path);sources.append(dict(path=str(path),sha256=sha(path)))
 rows=r['all_attempts'];assert [x['paid_actions']for x in rows]==list(range(r['search_range_inclusive'][0],r['search_range_inclusive'][0]+len(rows)))
 selected=r['selected'];assert not any(x['status']=='eligible_cost_pool'for x in rows[:-1])
 for row in rows:
  state=HERE/layout/f"state_{row['paid_actions']:03d}.json";assert read(state)==row
  attempts.append(dict(layout=layout,paid_actions=row['paid_actions'],status=row['status'],fit_object_count=row.get('fit_object_count'),fit_support_tracks=row.get('fit_support_tracks'),cost_band=row.get('cost_band'),record=dict(path=str(state),sha256=sha(state))))
 if selected:
  assert selected==rows[-1]and selected['status']=='eligible_cost_pool';full=scene['layouts'][layout]['sealed_prefixes'][1]['actions'];assert selected['actions']==full[:selected['paid_actions']];assert len(full)-selected['paid_actions']>=4
  overrides[layout]={'p1':dict(paid_actions=selected['paid_actions'],actions=selected['actions'],expected_position=selected['position'],expected_heading=selected['heading'],packet_sha256=selected['packet_sha256'],observed_input_sha256=selected['observed_input_sha256'],native_vertices_sha256=selected['native_vertices_sha256'],native_triangles_sha256=selected['native_triangles_sha256'],original_paid_actions=r['original_paid_actions'],selected_cost_band=selected['cost_band'],selection='earliest_eligible_in_predeclared_order',preparation_record=dict(path=str(path),sha256=sha(path)))}
for p,h in frozen['code_and_data_sha256'].items():assert sha(p)==h,p
result=dict(status='completed_cost_only_prefix_preparation_v2'if len(overrides)==2 else'cost_only_prefix_preparation_v2_incomplete',scene_freeze_sha256=sha(ROOT/'scene/freeze.json'),preparation_protocol_sha256=sha(ROOT/'PREFIX_PREPARATION_V2.json'),selector_freeze_sha256=sha(HERE/'selector_freeze.json'),finalizer_sha256=sha(Path(__file__)),overrides=overrides,all_attempts=attempts,selection_source_results=sources,retained_v1_preflight=dict(path=str(ROOT/'performance/observed_prefix_preflight.json'),sha256=sha(ROOT/'performance/observed_prefix_preflight.json')),retained_states=['chain_a/p2','chain_b/p2','cycle_a/p1','cycle_a/p2','cycle_b/p1','cycle_b/p2'],scene_geometry_modified=False,script_paths_modified=False,cost_band_width_modified=False,original_failures_preserved=True,visibility_or_acquisition_template_gain_evaluated=False,actual_branch_outcomes_evaluated=False,candidate_winners_selected=False,source_reference_surface_metrics_read=False,common_computation_note='The inherited native_candidates helper also computes original shared common proxies. Those values were neither retained nor used: v2 selection reads only the number and outbound action costs of native routes. Raw gain_called=False fields refer to B11 visibility/acquisition completion estimators, neither of which was invoked.')
(HERE/'prefix_override.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],selected_paid_actions={l:v['p1']['paid_actions']for l,v in overrides.items()},checked_states=len(attempts),sha256=sha(HERE/'prefix_override.json'))))
