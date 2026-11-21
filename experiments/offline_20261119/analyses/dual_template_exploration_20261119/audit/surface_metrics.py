"""Reuse the independently checked metric implementation with new target names.

The old helper has a historical fixed target-key spelling. The adapter below
changes dictionary names only, never point sets, distances, thresholds or seeds.
No actor imports this evaluator module.
"""
import sys
from pathlib import Path

OLD_EVALUATION=Path('/home/ouc/nso_robot/analyses/semantic_scene_redesign_20261119/exploration_evaluation')
sys.path.insert(0,str(OLD_EVALUATION))
from surface_reference import (unique_sampling_faces,subset_mesh,fixed_camera_poses,
    visible_reference,surface_samples,ray_scene,evaluate_surface as original_evaluate_surface)

def evaluate_surface(predicted_mesh,source_mesh,reference):
    # The closed object is now full source instance118; the old 115 ROI is never
    # loaded or used. Only adapt the legacy function's positional target slot.
    compatible={'all':reference['all'],'open31':reference['open31'],
                'closed115_fronts':reference['closed118']}
    result=original_evaluate_surface(predicted_mesh,source_mesh,compatible)
    result['target_surface_metrics']['closed118']=result['target_surface_metrics'].pop('closed115_fronts')
    return result
