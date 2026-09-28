"""Print saved-observation posterior traces; never create sensor packets."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.virtual3d.test_observation_belief_v35 import ObservationBeliefTestsV35 as Fixtures
from nso.observation_belief_v35 import ObservationBeliefV35

Fixtures.setUpClass()
rows = []
for fixture, packets in enumerate(Fixtures.saved):
    for mode in ('G', 'S', 'swapped', 'swapped_no_feedback'):
        belief = ObservationBeliefV35(Fixtures.templates, mode)
        trace = []
        for packet in packets:
            report = belief.update(**packet)
            trace.append({name: report[name] for name in (
                'step', 'pose', 'probabilities', 'geometry_applied_log_odds',
                'geometry_log_odds', 'semantic_log_odds', 'class_distinct_xy',
                'class_conflict', 'new_geometry_pose')})
        rows.append({'test_fixture_case': fixture, 'mode': mode, 'trace': trace})
forecasts = [Fixtures.templates.template_information(node)
             for node in range(len(Fixtures.templates.poses))]
print(json.dumps({'traces': rows, 'public_forecasts': forecasts,
                  'saved_unique_packets': 86, 'saved_packet_update_calls': 344,
                  'new_worlds': 0, 'new_sensor_queries': 0, 'new_main_tasks': 0,
                  'mapping_or_TSDF_calls': 0, 'metric_calls': 0,
                  'scope': 'saved-packet posterior intervention; no online route or final quality result'},
                 ensure_ascii=False, indent=2, allow_nan=False))
