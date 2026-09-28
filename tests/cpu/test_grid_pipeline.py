"""Exercise the real CLI, saved observations and paired baseline protocol."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def test_cli_artifacts_reconstruct_metrics_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((ROOT / 'configs/cpu/smoke.json').read_text())
            config['environment']['max_steps'] = 12
            (root / 'config.json').write_text(json.dumps(config))
            command = [sys.executable, str(ROOT / 'scripts/eval_cpu.py'),
                       '--config', str(root / 'config.json'), '--output', str(root / 'run')]
            subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
            folder = root / 'run'
            records = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
            report = json.loads((folder / 'summary.json').read_text())
            self.assertTrue(report['run_complete'])
            self.assertEqual(report['completed_episodes'], 2)
            self.assertEqual(records[0]['initial_pose'], records[1]['initial_pose'])
            self.assertEqual(records[0]['ground_truth_sha256'], records[1]['ground_truth_sha256'])
            for record in records:
                with (folder / record['artifact_dir'] / 'steps.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 13)
                self.assertEqual(int(rows[-1]['steps']), record['steps'])
                with np.load(folder / record['artifact_dir'] / 'observations.npz') as data:
                    h, w = data['map_shape']
                    visible = np.unpackbits(data['visible_packed'], axis=1)[:, :h*w].reshape((-1, h, w))
                    observed = visible.any(0)
                    np.testing.assert_array_equal(observed, data['final_belief'] != -1)
                    count = np.count_nonzero(observed & data['reachable'])
                    self.assertAlmostEqual(record['coverage_ratio'], count / data['reachable'].sum())
                    self.assertAlmostEqual(record['explored_area_m2'], count * .01)
            self.assertTrue((folder / 'coverage_curves.png').is_file())
            previous = (folder / 'episodes.jsonl').read_bytes()
            attempt = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(attempt.returncode, 0)
            self.assertEqual(previous, (folder / 'episodes.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
