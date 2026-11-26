"""Sensor-only calibration inference API; scale selection belongs to evaluator."""
from pathlib import Path
import importlib.util

spec=importlib.util.spec_from_file_location('B8_infer_model',Path(__file__).with_name('model.py'))
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)

def infer_depth_distribution(mapper,neutral_model=None,frozen_scale_config=None):
    bank=model.build_particle_bank(mapper,neutral_model)
    sigma=None if frozen_scale_config is None else float(frozen_scale_config['sigma_m'])
    output=bank.distribution(sigma_m=sigma,semantic=False)
    output['scale_selection_performed_by_inference']=False
    output['no_policy_or_navigation_execution']=True
    output['sigma_candidates_m']=list(model.SIGMA_CANDIDATES_M)
    return output
