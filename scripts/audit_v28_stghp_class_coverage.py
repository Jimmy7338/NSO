#!/usr/bin/env python3
"""Count frozen V27 semantic candidate/selection coverage by class."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_observed_autonomous_v26 import verify_inventory

INPUT = ROOT/'audit_results/observed_autonomous_v27_20260916'
DEFAULT = ROOT/'audit_results/v28_stghp_class_coverage_diagnosis_20260916'
CASES = (1, 3)
MINIMUM_COMPLEX_SELECTION_SHARE = .20


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def stable_counter(counter):
    return {str(key): value for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def audit_case(index):
    folder = INPUT/f'case_{index:02d}'
    result = read(folder/'result.json')
    plans = read_gzip(folder/result['plans_file'])
    candidate, selected, selected_after_both = Counter(), Counter(), Counter()
    selected_groups = Counter()
    candidate_gains, selected_gains = defaultdict(list), defaultdict(list)
    both_cue_plans = 0
    for plan in plans:
        for row in plan['candidates']:
            classes = {x['class_id'] for x in row['semantic_evidence'] if x['hypothesis_gain'] > 0}
            for class_id in classes:
                candidate[class_id] += 1
                candidate_gains[class_id].append(sum(x['hypothesis_gain'] for x in row['semantic_evidence']
                    if x['class_id'] == class_id and x['hypothesis_gain'] > 0))
        choice = plan['selected']
        classes = set() if choice is None else {
            x['class_id'] for x in choice['semantic_evidence'] if x['hypothesis_gain'] > 0}
        for class_id in classes:
            selected[class_id] += 1
            selected_groups[(class_id, choice['group'])] += 1
            selected_gains[class_id].append(sum(x['hypothesis_gain'] for x in choice['semantic_evidence']
                if x['class_id'] == class_id and x['hypothesis_gain'] > 0))
        action_id = plan['audit']['action_id']
        event = read_gzip(folder/'audit'/f'{action_id:04d}.json.gz')
        if {x['class_id'] for x in event['cues']} == {2, 3}:
            both_cue_plans += 1
            for class_id in classes:
                selected_after_both[class_id] += 1
    denominator = sum(selected_after_both.values())
    complex_share = selected_after_both[3]/denominator if denominator else 0.
    gains = {}
    for class_id in (2, 3):
        cg, sg = candidate_gains[class_id], selected_gains[class_id]
        gains[str(class_id)] = dict(candidate_mean=None if not cg else sum(cg)/len(cg),
            candidate_max=None if not cg else max(cg), selected_mean=None if not sg else sum(sg)/len(sg),
            selected_max=None if not sg else max(sg))
    return dict(case_index=index, assignment=result['assignment'], total_plans=len(plans),
        nonempty_selected_plans=sum(x['selected'] is not None for x in plans),
        both_cue_plans=both_cue_plans, candidate_with_class_evidence=stable_counter(candidate),
        selected_with_class_evidence=stable_counter(selected),
        selected_after_both_cues=stable_counter(selected_after_both),
        selected_group_by_class={f'{key[0]}:{key[1]}': value for key, value in sorted(selected_groups.items())},
        hypothesis_gain=gains, complex_selected_share_after_both_cues=complex_share,
        complex_selection_share_gate_passed=complex_share >= MINIMUM_COMPLEX_SELECTION_SHARE,
        evaluation_Q_read=False, planner_rerun=False, autonomous_actions=0)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    verify_inventory(INPUT)
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [audit_case(i) for i in CASES]
    args.output.mkdir(parents=True)
    manifest = dict(status='complete', input=str(INPUT.relative_to(ROOT)),
        input_inventory_sha256=sha(INPUT/'artifact_hashes.json'),
        script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(__file__),
        cases=list(CASES), minimum_complex_selection_share=MINIMUM_COMPLEX_SELECTION_SHARE,
        evaluation_Q_read=False, planner_rerun=False, autonomous_actions=0, new_main_tasks=0)
    result = dict(status='complete_frozen_planning_record_class_coverage_audit', cases=rows,
        both_arrangements_complex_selection_share_gate_passed=all(
            row['complex_selection_share_gate_passed'] for row in rows),
        total_both_cue_plans=sum(row['both_cue_plans'] for row in rows),
        total_class2_selected_after_both=sum(row['selected_after_both_cues'].get('2', 0) for row in rows),
        total_class3_selected_after_both=sum(row['selected_after_both_cues'].get('3', 0) for row in rows),
        evaluation_Q_read=False, planner_rerun=False, autonomous_actions=0, new_main_tasks=0)
    write(args.output/'manifest.json', manifest)
    write(args.output/'result.json', result)
    write(args.output/'artifact_hashes.json', {p.name: sha(p) for p in sorted(args.output.iterdir())
          if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
