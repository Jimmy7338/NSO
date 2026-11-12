"""Pure helpers for paired capture configuration; importing performs no I/O."""
import copy, hashlib, math
import numpy as np

def select_grid(pose, floor_z, prior_settings, make_grid):
    if prior_settings is None:
        grid = make_grid(pose, floor_odom_z_m=floor_z)
    else:
        grid = copy.deepcopy(prior_settings['calibration']['initial_grid'])
    anchor = grid['initial_actual_base_odom_pose']
    yaw = math.atan2(math.sin(pose[2]-anchor[2]), math.cos(pose[2]-anchor[2]))
    return grid, {'actual_pose':list(pose), 'shared_anchor_pose':list(anchor), 'xy_m':float(np.linalg.norm(np.asarray(pose[:2])-anchor[:2])), 'yaw_rad':float(yaw)}

def set_score(cfg, mode):
    if mode not in ('C','G','S'): raise ValueError('invalid score mode')
    out=copy.deepcopy(cfg);out['score_mode']=mode;return out

def packet_grid_evidence(path):
    with np.load(path, allow_pickle=False) as z:
        a=np.asarray(z['joint_belief_grid'])
        return {'field':'joint_belief_grid', 'shape':list(a.shape), 'dtype':str(a.dtype), 'array_sha256':hashlib.sha256(a.tobytes()).hexdigest(), 'counts':{str(int(v)):int(c) for v,c in zip(*np.unique(a,return_counts=True))}, 'quality_support':'not available as an initialized mapper field in this packet; must be recorded after actual worker observe', 'scope':'first captured navigation input only; not the fused mapper belief'}
