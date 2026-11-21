#!/usr/bin/env python3
"""Compare saved telemetry's turn reversals to the archived temporal diagnosis."""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT / path).read_text())

def main():
    summary = read('checkpoint_summary.json')
    diagnosis = read('temporal_turn_diagnosis.json')
    original = {p['probe']: p for p in diagnosis['probes']}
    results = []
    for item in summary['trials']:
        name = item['name']
        base = Path('probes') / name
        telemetry = read(base / 'telemetry.json')
        goal = read(base / 'goal.json')['target_odom']
        errors = [math.hypot(row['raw_odom']['pose'][0] - goal[0],
                             row['raw_odom']['pose'][1] - goal[1])
                  for row in telemetry if row.get('seconds_since_goal') is not None
                  and row.get('raw_odom')]
        result = {'probe': name, 'samples': len(telemetry),
                  'minimum_position_error_m': min(errors) if errors else None,
                  'samples_in_8cm': sum(e <= .08 for e in errors)}
        if name in original:
            ref = original[name]
            start = ref['start_unix_s']
            prior = None
            flips = []
            for row in telemetry:
                if row.get('seconds_since_goal') is None or not row.get('raw_odom'):
                    continue
                actuator = row.get('actuator', {})
                command, can = actuator.get('last_command'), actuator.get('can_velocity')
                if not command or not can or abs(command['w']) < .05:
                    continue
                current = (actuator['sampled_unix_s'], command['w'])
                if prior and current[1] * prior[1] < 0 and current[0] - prior[0] < 2:
                    flips.append((current[0] - start, prior[1], current[1]))
                prior = current
            expected = ref['command_sign_flips']
            assert len(flips) == len(expected), (name, len(flips), len(expected))
            for got, want in zip(flips, expected):
                assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
                           for a, b in zip(got, (want['at_s'], want['previous_command_w'], want['new_command_w']))), (name, got, want)
            result['command_sign_flips_verified'] = len(flips)
        results.append(result)
    print(json.dumps({'results': results, 'status': 'telemetry checks passed'}, indent=2))

if __name__ == '__main__':
    main()
