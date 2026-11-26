"""B10 acquired-prefix reconstruction adapter; never executes a new branch.

The renderer reconstructs only the sealed historical action prefix. The return
value exposes acquired frames/mapper/routes, not the simulator or source scene.
Consumers must pass only mapper/acquired data to an actor-side gain revision.
"""
from pathlib import Path
from dataclasses import dataclass
import importlib.util,sys,json,tempfile,hashlib
BASE=Path('/home/ouc/nso_robot/analyses')
B8=BASE/'dual_template_uncertainty_20261120'
B9=BASE/'dual_template_followup_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}

@dataclass
class AcquiredPrefix:
    mapper: object
    routes: list
    common_rows: list
    acquired_frames: tuple
    acquired_scans: tuple
    receipt: dict

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def rebuild_observed_prefix(layout,paid):
    if layout not in STATES or paid not in STATES[layout]:raise ValueError('Not a sealed development prefix')
    parent=load('B10_frozen_B8_prefix',B8/'replay/run_particle_predictions.py')
    parent.verify();old,b,repair,parts,model=parent.dependencies()
    context=old.sealed_prefix(b,layout,paid)
    reference=json.loads((B9/'structure/replay_attempt_001'/f'{layout}_{paid}'/'result.json').read_text())
    with tempfile.TemporaryDirectory(prefix='nso-b10-prefix-') as directory:
        receipt=old.save_mesh(context['mapper'],Path(directory)/'native.ply')
    assert receipt['sha256']==reference['native_mesh_sha256']
    assert context['world'].step_count==paid
    assert len(context['acquired'])==paid+1
    originals=json.loads((B8/'replay'/f'{layout}_{paid}_attempt_001'/'fixed_candidates.json').read_text())
    assert context['routes']==originals
    kept={id(f) for f in context['mapper'].keyframes}
    frames=tuple(f for f,_ in context['acquired'])
    audit={**context['prefix_receipt'],'native_mesh_sha256':receipt['sha256'],
        'acquired_frame_count':len(frames),'history_keyframe_count':len(kept),
        'acquired_frames_omitted_from_history':sum(id(f)not in kept for f in frames),
        'new_branch_actions':0,'new_observations':0,'renderer_replays_historical_prefix_only':True,
        'returned_simulator_or_truth':False,'candidate_generator_or_common_score_changed':False,
        'warning':'Do not replace mapper.keyframes: it also controls likelihood/rays. A consumption-only comparison must modify only reward masks.'}
    return AcquiredPrefix(context['mapper'],context['routes'],context['common_rows'],frames,
                          tuple(s for _,s in context['acquired']),audit)
