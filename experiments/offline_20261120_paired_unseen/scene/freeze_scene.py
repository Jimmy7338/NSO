"""Seal all layout geometry/prefix/reference inputs before estimator scoring."""
from pathlib import Path
import json,time,hashlib
from unseen_world import SPECS,LAYOUTS,sha,OLD,old
HERE=Path(__file__).resolve().parent
(HERE/'SCENE_PROTOCOL.json').write_text(json.dumps(dict(schema='B11_four_new_layouts_before_outcomes_v1',layout_order=list(LAYOUTS),layouts=SPECS,geometry_revision='Four new rectangular architecture dimensions/partitions and two actual graph topologies; all furniture placements/yaws new; two original authored assets retain physical dimensions and original mesh geometry.',prefix_contract='Scripted legal acquired prefixes, all actions charged; p1 andp2 predetermined before candidate scores; not autonomous trajectory evidence. Insufficient original candidate cost band remains a protocol failure, not a reason to replace a state.',actor_truth_boundary='Simulator receives source geometry for rendering/collision; predictor receives acquired RGBD/scan and actual observed support only. Source reference directory is evaluator-only. This module establishes packet/API isolation, not an OS sandbox by itself.',front_end='One open asset31 and one closedasset118 per layout. Current first-hit category2/3 andinstance1/2 match the unchanged B9 frontend; no hidden geometry or global labels in sensor packets.',shared_sensor_motion='Inherited noisefree RGBD120x90/90deg, exact poses, planar scan,0.2m grid/circular footprint, original source triangle sweep collision andstep256 contract. No scene-specific policy thresholds.',scope='Four controlled layouts, two room graph topologies, same two authored objects; not four natural environments or unseen object assets.',selection='No new candidate score, predicted gain, branch outcome or policy return was computed to select any layout or prefix.'),indent=2)+'\n')
files=[HERE/f for f in ['unseen_world.py','freeze_scene.py','prepare.py','prepare_reference.py','test_scene.py','tests.log','REFERENCE_PROTOCOL.json','SCENE_PROTOCOL.json','preparation_failures.json']]
files+=[OLD,old.PREVIOUS/'scene_adapter/replicacad_world.py',old.PREVIOUS/'scene_adapter/textured_raycast.py',old.ASSETS/'assemble_apt0.py',old.ASSETS/'download_receipt.json']
# Actual renderer import may live under assets, resolve the imported module.
files=[p for p in files if p.exists()]
import inspect
files.append(Path(inspect.getfile(old.TexturedRayRenderer)))
files += [p for p in (HERE/'prepared').rglob('*')if p.is_file()]
files += [p for p in (HERE/'precomputed_assets').glob('*')if p.is_file()]
files += [p for p in (HERE/'evaluator_references').rglob('*')if p.is_file()]
parent_freeze=HERE.parents[1]/'dual_template_exploration_20261119/layouts/shelf_left/prepared/freeze.json'
parent=json.loads(parent_freeze.read_text())
for relative,digest in parent['source_asset_sha256'].items():
 p=old.ASSETS/relative
 assert sha(p)==digest,relative
 files.append(p)
files.append(parent_freeze)
receipt=dict(status='frozen_four_new_layouts_two_topologies_eight_scripted_states',frozen_unix_s=time.time(),layout_order=list(LAYOUTS),layouts=4,room_graph_topologies=2,scripted_states=8,scene_tests_passed=5,preparation_failures=1,rerolled_layouts=0,new_estimator_scores_computed=0,new_branches_executed=0,new_policy_runs=0,code_and_data_sha256={str(p):sha(p)for p in sorted(set(files))},api=dict(world_factory='unseen_world.make_world(layout_id, seed=20261120, config=None)',compatible_class='ControlledFunctionalWorld',prefix_action_list="unseen_world.sealed_prefix_actions(layout_id, 'p1' or 'p2')",metadata='SPECS[layout_id] (simulator/protocol only)',evaluator_reference='evaluator_references/<layout>/{fixed_reference.npz,source_mesh.ply,RECEIPT.json}',reference_keys=['all','open31','closed118','camera_poses']),shared_frontend_unchanged=True,limits=['Controlled architecture and ideal current first-hit categories/instances, original authored source assets repeated acrosslayouts.','Scene topology/motion legality is checked using simulator geometry; policy receives only acquired inputs.','Scripted prefixes are fixed acquisition conditions, not NSO autonomous completion evidence.','Do not change or replace any layout/state after observing candidate scores, absent candidates or acquired outcome.'])
(HERE/'freeze.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],sha256=sha(HERE/'freeze.json'),files=len(files))))
