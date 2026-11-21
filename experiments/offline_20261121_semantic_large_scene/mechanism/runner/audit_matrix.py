"""Check all fixed attempts and the within-placement isolation receipts."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_arm(layout, arm):
    out = HERE / 'layouts' / layout / 'runs' / arm / 'attempt_001'
    assert tuple(out.parent.glob('attempt_*')) == (out,), out
    if (out / 'failure.json').is_file():
        return dict(status='failed', failure=read(out / 'failure.json'), output=str(out))
    manifest = read(out / 'artifact_manifest.json')
    for file, digest in manifest['files_sha256'].items():
        assert sha(out / file) == digest, (out, file)
    attempt = read(out / 'result.json')
    trace = read(out / 'steps.json')
    choices = read(out / 'choices.json')
    gate = read(out / 'intervention.json')['intervention']
    cap = read(HERE / 'protocol.json')['budget_including_return']
    expected = read(HERE / 'layouts' / layout / 'prepared/initial_state.json')
    assert len(trace) == attempt['paid_actions'] and len(trace) <= cap == attempt['paid_action_budget']
    assert attempt['initial_packet_sha256'] == expected['initial_packet_sha256']
    assert [x['step'] for x in trace] == list(range(1, len(trace) + 1))
    assert attempt['final_pose_equals_initial_anchor'] == (attempt['final_pose'] == expected['anchor'])
    assert attempt['collisions'] == sum(x['collision'] for x in trace)
    for choice in choices:
        c = choice['outputs']
        active = gate is not None and c['plan_number'] == gate['native_plan_number']
        assert c['mode'] == (arm if active else 'G'), (layout, arm, c['plan_number'])
    packets = [row['packet_sha256'] for row in trace]
    pretrigger_hash = hashlib.sha256(''.join(packets[:len(trace) if gate is None else gate['paid_before']]).encode()).hexdigest()
    mesh_hash = manifest['files_sha256'].get('final_tsdf_mesh.ply')
    return dict(output=str(out), status='completed_execution', initial_packet_sha256=attempt['initial_packet_sha256'],
                trigger=gate, pretrigger_packet_chain_sha256=pretrigger_hash,
                paid_actions=attempt['paid_actions'], returned=attempt['final_pose_equals_initial_anchor'],
                collisions=attempt['collisions'], trajectory_sha256=attempt['trajectory_sha256'], mesh_sha256=mesh_hash,
                manifest_sha256=sha(out / 'artifact_manifest.json'))


def audit():
    protocol = read(HERE / 'protocol.json')
    output = {}
    for layout in protocol['layouts']:
        arms = {arm: verify_arm(layout, arm) for arm in protocol['arms']}
        passed = [r for r in arms.values() if r['status'] == 'completed_execution']
        gids = [r['trigger'] for r in passed]
        checks = {}
        if passed:
            assert len({r['initial_packet_sha256'] for r in passed}) == 1
            assert len({r['pretrigger_packet_chain_sha256'] for r in passed}) == 1
            assert all(x is not None for x in gids) or all(x is None for x in gids), layout
            if gids[0] is None:
                assert len({r['trajectory_sha256'] for r in passed}) == 1
                assert len({r['mesh_sha256'] for r in passed}) == 1
            else:
                for field in ('paid_before', 'packet_sha256', 'belief_sha256', 'candidate_set_sha256', 'score_set_sha256'):
                    assert all(r[field] == gids[0][field] for r in gids), (layout, field)
            checks.update(cold_start_paired=True, pretrigger_trace_paired=True,
                          triggered=gids[0] is not None, valid_ends=len(passed), failure_count=len(arms)-len(passed))
        output[layout] = dict(arms=arms, pairing=checks)
    return dict(status='audited_terminal_complete_matrix', layouts=output, numerical_libraries_thread_limit=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'matrix_audit.json')
    args = parser.parse_args()
    report = audit()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({layout: row['pairing'] for layout, row in report['layouts'].items()}))
