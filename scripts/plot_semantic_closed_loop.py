#!/usr/bin/env python3
"""Plot the saved P02 diagnosis witness; no policy or sensor execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.diagnostic) != args.expected_sha256:
        raise ValueError('Explicit diagnostic hash mismatch')
    data = json.loads(args.diagnostic.read_text())
    witness = data['closed_loop_witness']
    if witness['run_id'] != 'core_P02_nom_S_b120_lexicographic':
        raise ValueError('This figure describes the explicitly selected P02 witness')
    episode = ROOT/'audit_results'/data['phase_id']/'episodes'/witness['run_id']
    pair = next(p for p in data['pairs'] if p['left'] == 'core_P02_nom_G_b120_lexicographic'
                and p['right'] == witness['run_id'])
    if (pair['first_different_decision_paid_step'] != 70
            or not pair['identical_rgbd_payloads_through_first_different_decision']):
        raise ValueError('Witness lacks the common observed-prefix comparison')
    rows = {r['paid_step']: r for r in witness['paid_steps']}
    prior = next(r for r in rows[70]['belief'] if r['instance_id'] == 'instance_0001')
    posterior = next(r for r in rows[71]['belief'] if r['instance_id'] == 'instance_0001')
    if prior['peer_instance_ids'] or prior['rho'] != .5:
        raise ValueError('This witness must not be attributed to cross-instance sharing')
    if not any(r['applied'] for r in rows[71]['feedback']):
        raise ValueError('Witness requires an actual paid-feedback update')
    if rows[72]['executed_sensor_action'] != 'observe' or not rows[72]['global_replanned']:
        raise ValueError('Actual macro completion and subsequent replanning required')
    manifest = json.loads((episode/'artifact_manifest.json').read_text())
    import numpy as np
    images = []
    inputs = {str(args.diagnostic.resolve()): args.expected_sha256}
    for step in (70, 71):
        file = episode/'packets'/f'{step:03d}_rgbd.npz'
        expected = manifest['files'][str(file.relative_to(episode))]['sha256']
        if sha(file) != expected:
            raise ValueError('Saved RGB-D packet has changed')
        inputs[str(file)] = expected
        with np.load(file, allow_pickle=False) as arrays:
            images.append(arrays['rgb'].copy())
    output = args.output.resolve()
    if output.exists() or 'episodes' in output.parts:
        raise ValueError('A new figure directory outside saved episodes is required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 240})
    fig = plt.figure(figsize=(11.4, 5.9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], width_ratios=[1, 1, 1.25])
    for index, step in enumerate((70, 71)):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(images[index], interpolation='nearest')
        ax.set_axis_off()
        ax.set_title(f'Paid RGB frame {step}' + (' · after forward' if step == 70 else ' · after right turn'), fontsize=9)
    ax = fig.add_subplot(grid[0, 2])
    margins = [pair['first_different_decision'][side]['diagnostic_minus_direct'] for side in ('left', 'right')]
    ax.barh([1, 0], np.array(margins)*1000, color=['#555555', '#009E73'], height=.45)
    ax.axvline(0, color='#666666', linewidth=.8)
    ax.set_yticks([1, 0], ['Geometry', 'Bayesian / shared'])
    ax.set_xlabel('Diagnostic − direct proxy score (×10⁻³)')
    ax.set_title('Decision after frame 70', fontsize=10)
    ax.grid(axis='x', color='#EEEEEE', linewidth=.6)
    ax = fig.add_subplot(grid[1, :2])
    x = np.arange(4)
    ax.bar(x-.17, prior['structure_probabilities'], width=.32, color='#B8B8B8', label='Before paid turn')
    ax.bar(x+.17, posterior['structure_probabilities'], width=.32, color='#009E73', label='After frame 71 feedback')
    ax.set_xticks(x, ['h₀', 'h₁', 'h₂', 'h₃'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Structure belief (uncalibrated)')
    ax.set_xlabel('Four public structural hypotheses')
    ax.set_title('Observed cabinet: actual belief update', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', color='#EEEEEE', linewidth=.6)
    ax = fig.add_subplot(grid[1, 2])
    ax.set_axis_off()
    labels = [('70', 'Global selection: diagnose'),
              ('71', 'Local turn → RGB-D → belief update'),
              ('72', 'Paid observe → macro complete'),
              ('72', 'Global replanning: direct observation')]
    for index, (step, text) in enumerate(labels):
        y = .92-index*.25
        ax.text(.02, y, step, fontsize=10, fontweight='bold', color='#0072B2', va='center')
        ax.text(.17, y, text, fontsize=8, va='center')
        if index < 3:
            ax.annotate('', xy=(.06, y-.18), xytext=(.06, y-.055),
                        arrowprops={'arrowstyle': '->', 'color': '#888888', 'linewidth': .8})
    fig.suptitle('P02: a paid diagnostic action closes the observation–belief–planning loop', fontsize=11)
    fig.text(.5, .018, 'Bayesian and shared-reliability policies execute the same trajectory. No same-class peer exists at the first divergence.\n'
             'Proxy-score and belief changes demonstrate execution; endpoint benefit is assessed separately.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .94), h_pad=2.3, w_pad=3)
    output.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'P02_paid_diagnosis.{suffix}', bbox_inches='tight')
    plt.close(fig)
    record = dict(schema='semantic.closed_loop_figure.v1', inputs=inputs,
                  source_sha256=sha(__file__), run_id=witness['run_id'], paid_frames=[70, 71, 72],
                  measured_endpoint_benefit_claim=False, cross_instance_increment_claim=False,
                  new_worlds=0, new_policy_runs=0, new_sensor_frames=0,
                  files={p.name: sha(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'output': str(output), 'new_worlds': 0, 'endpoint_benefit_claim': False}))


if __name__ == '__main__':
    main()
