import importlib.util,json
from pathlib import Path
R=Path(__file__).parent
original=Path('/home/ouc/nso_robot/analyses/sensor_repair_20261108/nov06_timing_trial/laser_coverage_evaluator.py')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('old_laser',original);new=load('new_laser',R/'laser_coverage_v2.py')
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261112/resume_S96_20261112_01');cfg=P/'joint_configuration.json';a=old.evaluate(P,cfg);b=new.evaluate(P,cfg)
assert len(a['curve'])==len(b['curve'])
for key in ['initial','final']:
 for name,value in a[key].items():assert b[key][name]==value,(key,name)
checks=old.validate(P,cfg,a)
receipt=dict(status='passed_same_geometry_and_all_original_measurement_contracts',legacy_S96_observations=len(a['curve']),legacy_initial=a['initial'],legacy_final=a['final'],original_validation_checks=checks['checks'],v2_filter_scope='observe plus accepted nonduplicate observe_failed_action only; all raster/range/ROI/TF/timestamp/CANbracket rules unchanged')
(R/'laser_v2_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],legacy_count=len(a['curve']),initial=a['initial']['observed_union_m2'],final=a['final']['observed_union_m2'])))
