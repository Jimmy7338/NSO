#!/usr/bin/env python3
"""Summarize schema-v2 episodes or audit legacy NSO logs without inventing units."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from datetime import datetime, timezone
from pathlib import Path


def _stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'count': 0}
    return {'mean': st.mean(values), 'std': st.pstdev(values),
            'median': st.median(values), 'min': min(values), 'max': max(values),
            'count': len(values)}


def _read_matrix(path):
    if not path.is_file():
        return []
    text = path.read_text()
    if '[' in text:
        blocks = re.findall(r'\[([^\[\]]*)\]', text, re.S)
        if text.count('[') != len(blocks) or text.count(']') != len(blocks):
            raise ValueError(f'Incomplete matrix record: {path}')
    else:
        blocks = text.splitlines()
    rows = [[float(x) for x in block.split()] for block in blocks if block.strip()]
    if rows and len({len(row) for row in rows}) != 1:
        raise ValueError(f'Inconsistent matrix widths: {path}')
    return rows


def _logged_config(text):
    result = {}
    keys = ('eval', 'num_processes', 'num_episodes', 'max_episode_length',
            'num_local_steps', 'train_goal_reachability', 'train_global',
            'train_local', 'train_slam', 'split', 'noisy_odometry')
    for key in keys:
        values = sorted(set(re.findall(r'\b' + key + r'=([^,\)\n]+)', text)))
        if len(values) == 1:
            value = values[0].strip()
            if value in ('True', 'False'):
                value = value == 'True'
            elif value.isdigit():
                value = int(value)
            else:
                value = value.strip("'\"")
            result[key] = value
        elif values:
            result[key] = {'conflicting_values': values}
    return result


def summarize(log_path, dump_dir, tag, meta=None):
    log_path, dump_dir = Path(log_path), Path(dump_dir)
    report = {'tag': tag, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'log_path': str(log_path), 'dump_dir': str(dump_dir),
              'user_meta': meta or {}}
    metadata_file = dump_dir / 'run_metadata.json'
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text())
        if metadata.get('metrics_schema_version') != 2:
            raise ValueError('Unsupported metrics schema')
        records = [json.loads(line) for line in (dump_dir / 'episodes.jsonl').read_text().splitlines()
                   if line.strip()]
        seen = set()
        for record in records:
            identity = (record.get('env_index'), record.get('episode_index'))
            if identity in seen:
                raise ValueError(f'Duplicate episode: {identity}')
            seen.add(identity)
            if record.get('run_id') != metadata['run_id'] or record.get('metrics_schema_version') != 2:
                raise ValueError('Mixed runs or metric schemas')
            if record.get('status') != 'complete':
                raise ValueError('Incomplete episode record')
            for key in ('coverage_ratio', 'explored_area_m2'):
                value = record.get(key)
                if value is not None and not math.isfinite(value):
                    raise ValueError(f'Non-finite {key}')
            coverage = record.get('coverage_ratio')
            if coverage is not None and not 0 <= coverage <= 1:
                raise ValueError('Coverage outside [0,1]')
        cfg = metadata['config']
        expected = cfg['num_processes'] * cfg['num_episodes']
        frozen = (metadata.get('eval_frozen_verified') is True
                  and bool(metadata.get('model_fingerprints_before'))
                  and metadata.get('model_fingerprints_before') == metadata.get('model_fingerprints_after'))
        valid = (metadata.get('status') == 'complete' and frozen and len(records) == expected
                 and all(r.get('step_count') == cfg['max_episode_length'] for r in records))
        report.update({'source_format': 'episodes_v2', 'run_metadata': metadata,
                       'evaluation_integrity_verified': valid,
                       'expected_episode_count': expected, 'episode_count': len(records),
                       'warnings': [] if valid else ['Incomplete run, unexpected lengths/count, or unverified frozen state.']})
        report['coverage_ratio_episode_final'] = _stats([r.get('coverage_ratio') for r in records])
        report['explored_area_m2_episode_final'] = _stats([r.get('explored_area_m2') for r in records])
        report['trajectory_drift_rmse_cm'] = _stats([r.get('trajectory_drift_rmse_cm') for r in records])
        report['unreachable_goal_count'] = _stats([r.get('unreachable_goal_count') for r in records])
        report['embodied_goal_success_rate'] = _stats([r.get('embodied_goal_success_rate') for r in records])
        report['raw_episodes'] = records
        return report

    text = log_path.read_text(errors='replace') if log_path.is_file() else ''
    ratios = _read_matrix(dump_dir / 'explored_ratio.txt')
    areas = _read_matrix(dump_dir / 'explored_area.txt')
    cfg = _logged_config(text)
    warnings = ['Legacy run: physical area cannot be recovered from scaled composite rewards.',
                'Legacy online coverage is an increment, not semantic or total coverage.',
                'Legacy drift is not trajectory ATE; logged zeros are retained only as raw observations.',
                'Module initialization does not establish participation in decisions.']
    if cfg.get('eval') == 1 and cfg.get('train_goal_reachability'):
        warnings.append('RPN training was enabled during evaluation; not a frozen-model benchmark.')
    expected = None
    if isinstance(cfg.get('num_processes'), int) and isinstance(cfg.get('num_episodes'), int):
        expected = cfg['num_processes'] * cfg['num_episodes']
        if expected != len(ratios):
            warnings.append('Matrix episode count differs from logged configuration.')
    if areas and len(areas) != len(ratios):
        warnings.append('Area and coverage matrix record counts differ.')
    matches = re.findall(r'Paper\[cov=([0-9.]+)% drift=([0-9.]+)cm unr=(\d+)\]', text)
    report.update({
        'source_format': 'legacy_audit', 'evaluation_integrity_verified': False,
        'logged_config': cfg, 'expected_episode_count': expected,
        'episode_count': len(ratios), 'warnings': warnings,
        'coverage_ratio_episode_final': _stats([r[-1] for r in ratios]),
        'coverage_ratio_episode_max': _stats([max(r) for r in ratios]),
        'explored_area_m2_episode_final': _stats([]),
        'trajectory_drift_rmse_cm': _stats([]),
        'legacy_unvalidated': {
            'scaled_composite_reward_episode_max': _stats([max(r) for r in areas]),
            'online_coverage_increment': _stats([float(m[0]) / 100 for m in matches]),
            'logged_drift_cm_not_ate': _stats([float(m[1]) for m in matches]),
            'repeated_unreachable_counter_snapshots': _stats([int(m[2]) for m in matches]),
        },
        'raw': {'coverage_rows': ratios, 'scaled_composite_reward_rows': areas},
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', default='', help='Legacy log; optional for schema v2')
    parser.add_argument('--dump', required=True)
    parser.add_argument('--tag', default='evaluation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--meta', default='{}')
    args = parser.parse_args()
    result = summarize(Path(args.log), Path(args.dump), args.tag, json.loads(args.meta))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing report: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f"[汇总] {output}: {result['episode_count']} records; "
          f"integrity_verified={result['evaluation_integrity_verified']}")


if __name__ == '__main__':
    main()
