"""Call only immediately after real observe20 and before its subsequent plan."""
import importlib.util
import hashlib
import json
from pathlib import Path

def audit(session,output_path):
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('native_frozen_availability_prefix20',here/'audit_module.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=here.parent/'source'
    result=json.loads((report/'result.json').read_text())
    running=json.loads((report/'running.json').read_text())
    arrival=result['successful_arrivals'][19]
    first=result['successful_arrivals'][0]
    assert arrival['verified'] and arrival['can_end']['accounting_complete']
    cost=dict(paid_actions=20,CAN_distance_m=arrival['can_end']['distance_m']-first['can_start']['distance_m'],
        CAN_end_record=arrival['can_end'],CAN_initial_record=first['can_start'],
        elapsed_to_arrival_pose_source_s=arrival['after']['source_stamp_s']-running['started_unix_s'],
        arrival_pose_source_stamp_s=arrival['after']['source_stamp_s'],started_unix_s=running['started_unix_s'],
        time_definition='difference of recorded arrival20 after-pose source timestamp and running.started_unix_s; not exact monotonic elapsed_run_s or observation21 completion time',
        prefix_not_free=True,source='original result.successful_arrivals[19].can_end/after and running.json',
        source_sha256={f:hashlib.sha256((report/f).read_bytes()).hexdigest() for f in ['result.json','running.json']})
    return module.audit(session,output_path,expected_observations=21,expected_paid_actions=20,prefix_acquisition=cost)
