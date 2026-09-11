#!/usr/bin/env python3
"""Guarded V7 T fitting, four-C prediction release, and C-only calibration.

No acquisition or reconstruction runs here. Missing parents, incomplete replay,
changed input seals, zero semantic support, and excessive selected-model df stop
the batch. ``predict`` releases all four C paired prefixes before C acquisition;
``calibrate`` requires that release and all four fully replayed C runs. Absolute
area calibration never changes raw ranking. Failure diagnostics are retained.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import traceback
import zipfile

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.conditional_response_v7 import (
    ALPHAS, AbsoluteAreaCalibrator, CapacityLimitError,
    HistoryFeatures, ResponseModelBank,
)
from scripts.compact_response_v7_meshes import validate_run_assets

PIPELINE = ROOT / 'configs/virtual3d/response_v7_pipeline.json'
VALIDATION = ROOT / 'configs/virtual3d/response_v7_validation.json'
MODEL_SOURCE = 'nso/conditional_response_v7.py'
FEATURE_NAMES = {'G', 'O', 'S', 'N', 'G_capacity', 'X', 'M'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def relative_file(folder, name):
    path = (folder / name).resolve()
    require(path.is_relative_to(folder.resolve()) and path != folder.resolve(), 'unsafe artifact path')
    return path


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.outcome_tables_opened = 0
        self.archived_meshes = {}
        self.compacted_runs = []

    def check(self, path, expected=None):
        path = Path(path).resolve()
        actual = file_hash(path)
        require(expected is None or actual == expected, f'hash mismatch: {path}')
        require(str(path) not in self.hashes or self.hashes[str(path)] == actual, f'input changed: {path}')
        self.hashes[str(path)] = actual
        return path

    def read(self, path):
        path = self.check(path)
        if path.name in ('outcomes.json', 'outcome.json'):
            raise ValueError('read family outcome tables only through the explicit outcome loader')
        return json.loads(path.read_text())

    def manifest(self, folder, allow_compacted=False):
        data = self.read(folder / 'artifact_hashes.json')
        require(isinstance(data, dict) and data, 'empty artifact inventory')
        removed = {}
        if allow_compacted:
            validate_run_assets(folder)
            if (folder / 'compaction_manifest.json').exists():
                compact = self.read(folder / 'compaction_manifest.json')
                removed = compact['meshes']
                self.compacted_runs.append(folder)
        for name, digest in data.items():
            path = relative_file(folder, name)
            if path.is_file():
                self.check(path, digest)
            else:
                require(name in removed and removed[name]['file_sha256'] == digest,
                        f'missing asset has no complete replayed-mesh certificate: {name}')
                self.archived_meshes[str(path)] = removed[name]
        return data

    def recheck(self):
        for folder in self.compacted_runs:
            validate_run_assets(folder)
        for name, digest in self.hashes.items():
            require(file_hash(Path(name)) == digest, f'input changed during operation: {name}')


def contracts(inputs):
    pipeline = inputs.read(PIPELINE)
    validation = inputs.read(VALIDATION)
    manifest = inputs.read(ROOT / pipeline['context_manifest'])
    require(pipeline['schema_version'] == 'response_v7_pipeline/1', 'unknown pipeline schema')
    require(validation['schema_version'] == 'response_v7_validation/1', 'unknown validation schema')
    require(pipeline['train_contexts'] == [f'T{i}' for i in range(8)], 'frozen eight T parents required')
    require(pipeline['calibration_contexts'] == [f'C{i}' for i in range(4)], 'frozen four C parents required')
    require(tuple(pipeline['fitting']['alpha_candidates']) == ALPHAS, 'alpha grid differs from frozen model')
    require(pipeline['fitting']['history_centering'] is True, 'this protocol ranks centered responses')
    require(pipeline['feature_contract']['columns'] == 12 and pipeline['candidate_limit'] == 6,
            'frozen six-candidate, twelve-column contract required')
    require(validation['ood']['range_margin_fraction'] == .25 and validation['ood']['minimum_margin'] == 1e-6,
            'OOD rule differs from fixed validation contract')
    require(manifest['family_order'] == ['storage_shelves', 'ventilation_baffles'], 'unexpected paired families')
    return {'pipeline': pipeline, 'validation': validation, 'manifest': manifest,
            'contexts': {x['context_id']: x for x in manifest['contexts']}, 'families': manifest['family_order']}


def archive_check(inputs, folder, source_hashes):
    path = inputs.check(folder / 'sources.zip')
    with zipfile.ZipFile(path) as archive:
        for name, digest in source_hashes.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, f'source archive mismatch: {name}')
    require(source_hashes[MODEL_SOURCE] == file_hash(ROOT / MODEL_SOURCE), 'frozen response implementation changed')


def context_inventory(inputs, paths, role, contract, prepared=False):
    """All parent completeness checks precede even a family outcome parse."""
    expected = contract['pipeline']['train_contexts' if role == 'train' else 'calibration_contexts']
    require(len(paths) == len(expected), f'require exactly all {len(expected)} {role} parent directories')
    inventory = {}
    for raw in paths:
        path = Path(raw).resolve()
        meta = inputs.read(path / 'metadata.json')
        require(meta['status'] == 'complete', f'incomplete parent: {path}')
        context = meta['context']; cid = context['context_id']
        require(cid in expected and cid not in inventory, f'wrong or duplicate parent: {cid}')
        require(context == contract['contexts'][cid] and context['role'] == role, f'parent contract mismatch: {cid}')
        require(context['outer_seed'] in (751, 752), 'unopened evaluation context is prohibited')
        if prepared:
            require(meta['candidate_structure_gate_passed'] is True, f'preparation structure failed: {cid}')
        else:
            require(meta['physical_branches'] == 12, f'incomplete family acquisition: {cid}')
            v = inputs.read(path / 'verification.json')
            required = {'status': 'passed_full', 'passed_full': True, 'partial': False, 'max_branches': None,
                        'branches_checked': 12, 'branches_total': 12, 'run': str(path), 'context_id': cid,
                        'role': role, 'raw_hashes_rechecked_after_replay': True,
                        'paired_geometric_prefix_exact': True, 'paired_route_pools_exact': True,
                        'paired_nonsemantic_features_exact': True}
            for key, value in required.items():
                require(key in v and v[key] == value, f'incomplete or mismatched replay {cid}: {key}')
            require(v['artifact_manifest_sha256'] == file_hash(path / 'artifact_hashes.json'), 'replay/manifest binding changed')
            require(v['source_archive_sha256'] == file_hash(path / 'sources.zip'), 'replay/archive binding changed')
            require(v['source_sha256'] == meta['source_sha256'], 'replay/source inventory differs')
            require(len(v['verifier_sha256']) == 64 and len(v['base_verifier_sha256']) == 64, 'missing independent verifier identity')
            require({f['family'] for f in v['families']} == set(contract['families']), 'replay paired family inventory differs')
            require(all(f['branches_total'] == 6 and len(f['checked_branches']) == 6 for f in v['families']), 'partial family replay')
        inventory[cid] = {'path': path, 'meta': meta}
    require(set(inventory) == set(expected), 'missing predeclared parent')
    return {cid: inventory[cid] for cid in expected}


def verified_packets(inputs, inventory, contract, prepared=False, bank_sha=None):
    packets = []
    common_sources = None
    for cid, record in inventory.items():
        folder, meta = record['path'], record['meta']
        artifact = inputs.manifest(folder, allow_compacted=not prepared)
        archive_check(inputs, folder, meta['source_sha256'])
        if common_sources is None:
            common_sources = meta['source_sha256']
        require(meta['source_sha256'] == common_sources, 'acquisition/preparation source versions differ across parents')
        require({p.parent.name for p in folder.glob('*/fixture.json')} == set(contract['families']), 'extra or missing physical family')
        if prepared:
            seal = artifact
            original = folder
            original_meta = meta
        else:
            require(inputs.read(folder / 'config.json') == contract['pipeline'], 'run pipeline differs from frozen config')
            seal = inputs.read(folder / 'pre_outcome_seal.json')
            original = Path(meta['prepared']).resolve()
            inputs.check(original / 'artifact_hashes.json', meta['prepared_manifest_sha256'])
            original_artifact = inputs.manifest(original)
            original_meta = inputs.read(original / 'metadata.json')
            require(original_meta['status'] == 'complete' and original_meta['candidate_structure_gate_passed'], 'original preparation failed')
            for name, digest in seal.items():
                require(artifact.get(name) == digest and original_artifact.get(name) == digest, 'decision input differs from pre-outcome seal')
                inputs.check(relative_file(folder, name), digest)
        if bank_sha is not None:
            require(original_meta['bank_sha256'] == bank_sha, 'C predictions were prepared with another bank')
        local = []
        for family in contract['families']:
            base = folder / family
            required = ['fixture.json', 'features.npz', 'feature_audit.json', 'candidates.json',
                        'candidate_audit.json', 'preparation.json']
            if bank_sha is not None:
                required += ['predictions.json', 'choices.json']
            for name in required:
                require(f'{family}/{name}' in seal, f'input was not sealed before outcomes: {family}/{name}')
            fixture = inputs.read(base / 'fixture.json')
            require(fixture['context'] == contract['contexts'][cid] and fixture['family'] == family, 'family fixture mismatch')
            routes = inputs.read(base / 'candidates.json')
            require([r['candidate_id'] for r in routes] == list(range(6)), 'feature row/candidate alignment is not frozen 0..5')
            require(all(r['cost'] == len(r['actions']) == len(r['states']) - 1 and 0 < r['cost'] <= 48 for r in routes), 'invalid planned paid cost')
            with np.load(inputs.check(base / 'features.npz'), allow_pickle=False) as arrays:
                features = {k: arrays[k].copy() for k in arrays.files}
            require(set(features) == FEATURE_NAMES, 'unexpected feature channels')
            require(all(x.shape == (6, 12) and np.isfinite(x).all() for x in features.values()), 'invalid physical feature matrices')
            for name, x in features.items():
                np.testing.assert_array_equal(x[:, :4], features['G'][:, :4], err_msg=f'{name} changed common geometry')
            np.testing.assert_array_equal(features['M'], features['G'])
            np.testing.assert_array_equal(features['S'][:, :8], features['O'][:, :8])
            np.testing.assert_array_equal(features['X'][:, :8], features['S'][:, :8])
            np.testing.assert_array_equal(features['X'][:, 8:], -features['S'][:, 8:])
            np.testing.assert_array_equal(features['G_capacity'][:, :8], features['G'][:, :8])
            require(not features['N'][:, 4:].any() and not features['G'][:, 8:].any() and not features['O'][:, 8:].any(), 'control contains forbidden feature columns')
            packet = {'history': HistoryFeatures(f'{cid}/{family}', cid, features), 'folder': base,
                'routes': routes, 'feature_audit': inputs.read(base / 'feature_audit.json'),
                'candidate_audit': inputs.read(base / 'candidate_audit.json'),
                'preparation': inputs.read(base / 'preparation.json'),
                'original_prepared': str(original), 'family': family}
            packets.append(packet); local.append(packet)
        require(local[0]['routes'] == local[1]['routes'], 'paired candidate pool differs')
        for name in ('G', 'O', 'N', 'M', 'G_capacity'):
            np.testing.assert_array_equal(local[0]['history'].features[name], local[1]['history'].features[name])
        np.testing.assert_array_equal(local[0]['history'].features['S'][:, 8:], -local[1]['history'].features['S'][:, 8:])
    return packets, common_sources


def structure_diagnostics(packets):
    records, errors = [], []
    class_support = {-1: {'histories': [], 'costs': set(), 'roles': set(), 'front': False, 'side': False},
                      1: {'histories': [], 'costs': set(), 'roles': set(), 'front': False, 'side': False}}
    for packet in packets:
        h = packet['history']; patches = packet['feature_audit']['patches']
        roles = packet['candidate_audit']['available_roles']
        marked_roles = sorted({r['role'] for r in roles if r.get('patch') is not None and patches[r['patch']]['marked_points'] > 0})
        signs = sorted({int(np.sign(p['class_vote'])) for p in patches if p['marked_points'] > 0 and p['class_vote'] != 0})
        if len(marked_roles) < 2 or not signs:
            errors.append(f'{h.history_id}: missing visible class or two marked directional roles')
        for sign in signs:
            entry = class_support[sign]; entry['histories'].append(h.history_id)
            marked = [j for j, p in enumerate(patches) if p['marked_points'] > 0 and np.sign(p['class_vote']) == sign]
            descriptors = np.array([r['unscaled_patch_descriptors'] for r in packet['feature_audit']['routes']])
            supported = np.any(descriptors[:, marked, :] > 0, axis=(1, 2))
            entry['costs'].update(r['cost'] for r, yes in zip(packet['routes'], supported) if yes)
            entry['roles'].update(r['group'] for r, yes in zip(packet['routes'], supported) if yes)
            entry['front'] |= bool(np.any(descriptors[:, marked, 0] > 0))
            entry['side'] |= bool(np.any(descriptors[:, marked, 1] > 0))
        records.append({'history': h.history_id, 'visible_vote_signs': signs, 'marked_direction_roles': marked_roles,
            'available_roles': roles, 'missing_roles': packet['candidate_audit'].get('missing_roles', []),
            'candidate_roles': [r['group'] for r in packet['routes']], 'costs': [r['cost'] for r in packet['routes']]})
    for sign, item in class_support.items():
        if not item['histories'] or not item['front'] or not item['side'] or len(item['costs']) < 2:
            errors.append(f'visible sign {sign}: zero class/front/side support or only one route cost')
    design = np.concatenate([(p['history'].features['S'] - p['history'].features['S'].mean(axis=0)) / np.sqrt(len(p['routes'])) for p in packets])
    common, conditional = design[:, :8], design[:, 8:]
    residual = conditional - common @ np.linalg.lstsq(common, conditional, rcond=None)[0]
    common_rank, full_rank = int(np.linalg.matrix_rank(common)), int(np.linalg.matrix_rank(design))
    if full_rank <= common_rank:
        errors.append('conditional columns have no numerically identifiable direction beyond common geometry')
    report = {'passed': not errors, 'errors': errors, 'histories': records,
        'class_support': {str(k): {key: sorted(value) if isinstance(value, set) else value for key, value in v.items()} for k, v in class_support.items()},
        'common_design_rank': common_rank, 'full_S_design_rank': full_rank,
        'conditional_norm': float(np.linalg.norm(conditional)), 'conditional_projection_residual_norm': float(np.linalg.norm(residual)),
        'conditional_projection_residual_fraction': float(np.linalg.norm(residual) / np.linalg.norm(conditional)) if np.linalg.norm(conditional) else None,
        'direction_support_source': 'each actually marked patch descriptor, never aggregated unmarked background support',
        'predeclared_roles': ['coverage_anchor', 'old_surface_rotation', 'left', 'right', 'back', 'front'],
        'observed_role_counts': dict(Counter(r['group'] for p in packets for r in p['routes'])),
        'interpretation': 'finite T support and linear identifiability checks, not a generalization guarantee'}
    return report


def make_guard(packets, validation):
    x = np.concatenate([p['history'].features['G'][:, :8] for p in packets])
    lower, upper = x.min(axis=0), x.max(axis=0)
    margin = np.maximum(validation['ood']['range_margin_fraction'] * (upper - lower), validation['ood']['minimum_margin'])
    return {'schema_version': 'response_v7_guard/1', 'columns': list(range(8)), 'channel': 'G',
        'training_min': lower.tolist(), 'training_max': upper.tolist(), 'margin': margin.tolist(),
        'lower': (lower - margin).tolist(), 'upper': (upper + margin).tolist(),
        'history_trigger': 'any candidate outside any bound', 'fallback': 'N',
        'whole_history_fallback': True, 'raw_predictions_retained': True,
        'probabilistic_coverage_guarantee': False, 'changes_absolute_calibration': False}


def guarded_predictions(bank, history, guard):
    raw = bank.predict(history)
    x = history.features['G'][:, guard['columns']]
    bad = (x < np.array(guard['lower'])) | (x > np.array(guard['upper']))
    fallback = bool(bad.any())
    guarded = {name: raw['N'].copy() if fallback else value.copy() for name, value in raw.items()}
    return raw, guarded, {'fallback': fallback, 'violating_candidate_columns': np.argwhere(bad).tolist()}


def select(predictions, routes):
    return {name: min(range(len(routes)), key=lambda i: (-values[i, 0], routes[i]['cost'], i))
            for name, values in predictions.items()}


def load_outcomes(inputs, packets):
    """This is the only entry that parses any family outcome values."""
    targets = {}
    for packet in packets:
        path = inputs.check(packet['folder'] / 'outcomes.json')
        rows = json.loads(path.read_text()); inputs.outcome_tables_opened += 1
        require(len(rows) == len(packet['routes']), 'outcome count differs from sealed candidates')
        mapping = {r['candidate_id']: r for r in rows}
        require(len(mapping) == len(rows) and set(mapping) == {r['candidate_id'] for r in packet['routes']}, 'duplicate or missing candidate outcomes')
        ordered = [mapping[r['candidate_id']] for r in packet['routes']]
        for row, route in zip(ordered, packet['routes']):
            require(0 < row['paid_actions'] <= row['planned_actions'] == route['cost'], 'paid/planned action mismatch')
            require(row['failure'] is not None or row['paid_actions'] == route['cost'], 'unrecorded truncated execution')
            require(np.isfinite([row['new_area_m2'], row['area_per_action'], row['f1_gain_05cm']]).all() and row['new_area_m2'] >= 0, 'invalid response label')
            np.testing.assert_allclose(row['area_per_action'], row['new_area_m2'] / row['paid_actions'], rtol=0, atol=1e-12)
            row['returned_to_origin'] = row['failure'] is None
            if row['failure'] is not None:
                actions = inputs.read(packet['folder'] / f"candidate_{row['candidate_id']:03d}" / 'actions.json')
                last = actions[-1]
                row['returned_to_origin'] = list(last['position']) + [last['heading']] == route['states'][0]
                row['failure_stage'] = last['stage']
        packet['outcomes'] = ordered
        targets[packet['history'].history_id] = np.array([[r['area_per_action'], r['f1_gain_05cm']] for r in ordered])
    return targets


def snapshot(output, inputs, contract):
    names = [str(Path(__file__).resolve().relative_to(ROOT)), MODEL_SOURCE, 'nso/__init__.py',
             str(PIPELINE.relative_to(ROOT)), str(VALIDATION.relative_to(ROOT)),
             contract['pipeline']['context_manifest'], 'requirements-3d.lock.txt', 'scripts/compact_response_v7_meshes.py']
    hashes = {name: file_hash(inputs.check(ROOT / name)) for name in names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    write_json(output / 'pipeline.json', contract['pipeline'])
    write_json(output / 'validation.json', contract['validation'])
    return hashes


def load_bank(inputs, path, contract):
    path = Path(path).resolve()
    manifest = inputs.manifest(path.parent)
    meta = inputs.read(path.parent / 'metadata.json')
    require(meta['status'] == 'complete' and meta['stage'] == 'fit', 'bank is not a completed accepted T fit')
    require(path.name == 'model_bank.json' and manifest.get(path.name) == file_hash(path) == meta['model_bank_sha256'],
            'bank bytes are not the completed fit artifact announced in metadata')
    data = inputs.read(path)
    provenance = data['_training_contract']
    require(provenance['pipeline_sha256'] == file_hash(PIPELINE) and provenance['validation_sha256'] == file_hash(VALIDATION), 'bank validation contract changed')
    guard = inputs.read(path.parent / 'guard.json')
    require(provenance['guard_sha256'] == file_hash(path.parent / 'guard.json'), 'guard differs from bank-bound pre-C rule')
    require(provenance['train_contexts'] == contract['pipeline']['train_contexts'], 'bank trained on another parent set')
    bank = ResponseModelBank.from_dict(data)
    require(set(bank.models) == set(contract['pipeline']['fitting']['fit_methods']), 'bank model inventory differs')
    require(all(m.capacity_ok and m.diagnostics['parent_context_count'] == 8 for m in bank.models.values()), 'bank failed capacity gate')
    require(bank.common_columns == 4 and bank.object_columns == 4 and
            all(m.center_history is True and m.coefficient.shape == (12, 2) and
                m.diagnostics['parent_contexts'] == contract['pipeline']['train_contexts'] for m in bank.models.values()),
            'bank feature dimensions, centering, or parent identities differ')
    return bank, guard, provenance


def fit_stage(args, output, inputs, contract, meta):
    inventory = context_inventory(inputs, args.runs, 'train', contract)
    packets, acquisition_sources = verified_packets(inputs, inventory, contract)
    report = structure_diagnostics(packets)
    write_json(output / 'training_structure.json', report)
    require(report['passed'], 'training structure gate failed: ' + '; '.join(report['errors']))
    guard = make_guard(packets, contract['validation'])
    write_json(output / 'guard.json', guard)
    # Target arrays are assembled only after all gates. Replay status documents
    # can contain response fields; those fields are never inputs to prediction.
    targets = load_outcomes(inputs, packets)
    write_json(output / 'execution_failures.json', failure_records(packets))
    fitting = contract['pipeline']['fitting']
    try:
        bank = ResponseModelBank.fit([p['history'] for p in packets], targets,
            fit_names=tuple(fitting['fit_methods']), common_columns=4, object_columns=4,
            center_history=fitting['history_centering'], enforce_capacity=True)
    except CapacityLimitError as exc:
        failed_bank = ResponseModelBank(exc.models, 4, 4)
        write_json(output / 'failed_model_bank.json', failed_bank.to_dict())
        write_json(output / 'failed_model_training_choices.json', training_choice_diagnostics(failed_bank, packets))
        write_json(output / 'paired_information_oracle.json', paired_information_oracle(packets))
        meta['failure_kind'] = 'selected_model_capacity_gate'
        raise
    provenance = {'train_contexts': list(inventory), 'histories': [p['history'].history_id for p in packets],
        'pipeline_sha256': file_hash(PIPELINE), 'validation_sha256': file_hash(VALIDATION),
        'guard_sha256': file_hash(output / 'guard.json'), 'acquisition_source_sha256': acquisition_sources,
        'outcomes_used': 'T only; all 96 fixed branches, including every execution failure',
        'independent_confirmation': False}
    write_json(output / 'model_bank.json', {**bank.to_dict(), '_training_contract': provenance})
    write_json(output / 'paired_information_oracle.json', paired_information_oracle(packets))
    training_choices = training_choice_diagnostics(bank, packets)
    write_json(output / 'training_choices.json', training_choices)
    write_json(output / 'fit_summary.json', {'parents': 8, 'histories': len(packets),
        'candidates': sum(len(p['routes']) for p in packets),
        'execution_failures_retained': sum(r['failure'] is not None for p in packets for r in p['outcomes']),
        'selected_training_failures_by_scorer': training_choices['selected_failures_by_scorer'],
        'training_choices_are_in_sample_diagnostics': True,
        'models': {name: {'alpha': m.alpha, **m.diagnostics} for name, m in bank.models.items()},
        'training_structure_passed': True, 'capacity_passed': True, 'C_or_E_outcomes_used': False,
        'C_must_still_pass': True, 'full_system_advantage_proven': False})
    meta.update(model_bank_sha256=file_hash(output / 'model_bank.json'), next_stage='prepare all four C prefixes, then predict release')


def predict_stage(args, output, inputs, contract, meta):
    bank, guard, provenance = load_bank(inputs, args.bank, contract)
    bank_sha = file_hash(args.bank)
    inventory = context_inventory(inputs, args.prepared, 'calibration', contract, prepared=True)
    packets, prepared_sources = verified_packets(inputs, inventory, contract, prepared=True, bank_sha=bank_sha)
    require(all(provenance['acquisition_source_sha256'].get(k) == v for k, v in prepared_sources.items()),
            'C preparation differs from frozen T acquisition dependencies')
    release = {'bank_sha256': bank_sha, 'guard_sha256': provenance['guard_sha256'],
               'prepared': {cid: {'path': str(row['path']), 'artifact_manifest_sha256': file_hash(row['path'] / 'artifact_hashes.json')} for cid, row in inventory.items()},
               'histories': [], 'outcomes_opened': False, 'must_precede_every_C_physical_branch': True}
    for packet in packets:
        h = packet['history']; raw, guarded, decision = guarded_predictions(bank, h, guard)
        original = inputs.read(packet['folder'] / 'predictions.json')
        require(set(original) == set(raw), 'prepared prediction channel inventory differs')
        for name in raw:
            np.testing.assert_allclose(raw[name], original[name], rtol=0, atol=1e-12)
        require(select(raw, packet['routes']) == inputs.read(packet['folder'] / 'choices.json'), 'prepared raw choices differ')
        target = output / h.history_id
        write_json(target / 'raw_predictions.json', {k: v.tolist() for k, v in raw.items()})
        write_json(target / 'guarded_predictions.json', {k: v.tolist() for k, v in guarded.items()})
        write_json(target / 'choices.json', {'raw': select(raw, packet['routes']), 'guarded': select(guarded, packet['routes']), 'ood': decision})
        release['histories'].append(h.history_id)
    write_json(output / 'release.json', release)
    meta.update(bank_sha256=bank_sha, next_stage='execute and independently replay all four C parents; never fit on partial C')


def ranking_metrics(prediction, actual):
    pairs = [(i, j) for i in range(len(actual)) for j in range(i + 1, len(actual)) if actual[i] != actual[j]]
    correctness = [1. if (prediction[i] - prediction[j]) * (actual[i] - actual[j]) > 0
                   else .5 if prediction[i] == prediction[j] else 0. for i, j in pairs]
    correlation = float(spearmanr(prediction, actual).statistic) if np.ptp(prediction) and np.ptp(actual) else None
    return {'relative_area_rate_rmse': float(np.sqrt(np.mean((prediction - (actual - actual.mean())) ** 2))),
            'pairwise_accuracy': float(np.mean(correctness)) if correctness else None, 'spearman': correlation}


def paired_information_oracle(packets):
    """GT posthoc value upper bound; never a scorer, gate, or sampling rule."""
    by_context = {}
    for p in packets:
        by_context.setdefault(p['history'].context_id, {})[p['family']] = p
    rows = []
    for context, pair in sorted(by_context.items()):
        require(set(pair) == {'storage_shelves', 'ventilation_baffles'}, 'oracle needs the complete physical pair')
        first, second = (pair[k] for k in ('storage_shelves', 'ventilation_baffles'))
        require(first['routes'] == second['routes'], 'oracle requires an identical action pool')
        a, b = (np.array([r['area_per_action'] for r in p['outcomes']]) for p in (first, second))
        marginal = .5 * (a + b)
        value = float(.5 * (a.max() + b.max()) - marginal.max())
        tolerance = 1e-12 * max(1., float(abs(a).max()), float(abs(b).max()))
        require(value >= -tolerance, 'paired information oracle violated nonnegativity')
        optimum_a, optimum_b = set(np.flatnonzero(a == a.max())), set(np.flatnonzero(b == b.max()))
        rows.append({'context_id': context, 'storage_area_rates': a.tolist(), 'ventilation_area_rates': b.tolist(),
            'conditional_family_oracle': float(.5 * (a.max() + b.max())),
            'common_geometry_marginal_oracle': float(marginal.max()), 'value_of_family_information_upper_bound': value,
            'storage_maximizing_candidate_ids': sorted(map(int, optimum_a)),
            'ventilation_maximizing_candidate_ids': sorted(map(int, optimum_b)),
            'maximizer_sets_intersect_exactly': bool(optimum_a & optimum_b),
            'value_zero_within_roundoff': abs(value) <= tolerance, 'roundoff_tolerance': tolerance})
    return {'scope': 'posthoc GT family-aware versus shared-action oracle; not learned S performance',
            'formula': '0.5*(max(u_storage)+max(u_ventilation))-max(0.5*(u_storage+u_ventilation))',
            'used_for_gate_or_context_selection': False, 'contexts': rows}


def failure_records(packets):
    return {'policy': 'all failed/short/nonreturning branches retained; area/actual-paid is not a full successful route',
            'rows': [{'history': p['history'].history_id, **{k: r.get(k) for k in ('candidate_id', 'failure',
                'failure_stage', 'paid_actions', 'planned_actions', 'returned_to_origin', 'new_area_m2',
                'area_per_action', 'f1_gain_05cm')}} for p in packets for r in p['outcomes'] if r['failure'] is not None]}


def training_choice_diagnostics(bank, packets):
    rows = []
    for p in packets:
        for name, chosen in bank.select(p['history'], [r['cost'] for r in p['routes']]).items():
            result = p['outcomes'][chosen]
            rows.append({'history': p['history'].history_id, 'scorer': name,
                **{key: result.get(key) for key in ('candidate_id', 'failure', 'failure_stage',
                    'paid_actions', 'planned_actions', 'returned_to_origin', 'area_per_action', 'f1_gain_05cm')}})
    return {'scope': 'T in-sample post-fit selection diagnostics, not held-out method efficacy',
            'rows': rows, 'selected_failures_by_scorer': dict(Counter(r['scorer'] for r in rows if r['failure'] is not None)),
            'failed_branches_were_not_deleted_or_zero_filled': True}


def verify_execution_receipt(inputs, run, release_dir, bank_sha):
    receipt = inputs.read(run / 'analysis/execution_receipt.json')
    require(receipt['status'] == 'execution_complete' and receipt['run'] == str(run), 'C run lacks successful execution guardian receipt')
    armed_path = Path(receipt['armed_path'])
    inputs.manifest(armed_path.parent)
    guard_meta = inputs.read(armed_path.parent / 'metadata.json')
    require(guard_meta['status'] == 'complete' and guard_meta['stage'] == 'execute-calibration', 'execution guardian did not complete')
    require(inputs.read(armed_path.parent / 'execution_receipt.json') == receipt, 'run/guardian receipt copies differ')
    inputs.check(armed_path, receipt['armed_sha256'])
    armed = inputs.read(armed_path)
    require(armed['status'] == 'armed' and armed['target_was_absent'] is True and armed['run'] == str(run), 'invalid pre-execution receipt')
    require(armed['release_manifest_sha256'] == file_hash(release_dir / 'artifact_hashes.json') and
            armed['release'] == str(release_dir) and armed['bank_sha256'] == bank_sha, 'C execution was not gated by this release')
    require(receipt['artifact_manifest_sha256'] == file_hash(run / 'artifact_hashes.json'), 'execution receipt/run manifest binding differs')


def execute_calibration(args):
    """Pre-execution guardian around the unchanged acquisition runner."""
    run_path = args.output.resolve()
    require(not run_path.exists(), 'refuse to attach an execution receipt to a preexisting C run')
    output = run_path.with_name(run_path.name + '.execution_guard')
    output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs(); meta = {'stage': 'execute-calibration', 'status': 'running'}
    code = 2
    try:
        contract = contracts(inputs)
        source_hashes = snapshot(output, inputs, contract)
        bank, _, provenance = load_bank(inputs, args.bank, contract)
        release_dir = args.release.resolve(); inputs.manifest(release_dir)
        release_meta = inputs.read(release_dir / 'metadata.json')
        release = inputs.read(release_dir / 'release.json')
        require(release_meta['stage'] == 'predict' and release_meta['status'] == 'complete', 'C release is incomplete')
        require(set(release['prepared']) == set(contract['pipeline']['calibration_contexts']), 'all four C prefixes must be released together')
        require(args.context in release['prepared'], 'context not in the frozen C release')
        require(release['bank_sha256'] == file_hash(args.bank) and release['guard_sha256'] == provenance['guard_sha256'], 'release bank or guard differs')
        for record in release['prepared'].values():
            prepared = Path(record['path'])
            inputs.check(prepared / 'artifact_hashes.json', record['artifact_manifest_sha256'])
            inputs.manifest(prepared)
            require(inputs.read(prepared / 'metadata.json')['bank_sha256'] == release['bank_sha256'], 'prepared bank changed')
        for name, digest in provenance['acquisition_source_sha256'].items():
            inputs.check(ROOT / name, digest)
        prepared = Path(release['prepared'][args.context]['path'])
        armed = {'schema_version': 'response_v7_C_execution_guard/1', 'status': 'armed',
            'context_id': args.context, 'run': str(run_path), 'target_was_absent': not run_path.exists(),
            'prepared': str(prepared), 'prepared_manifest_sha256': file_hash(prepared / 'artifact_hashes.json'),
            'release': str(release_dir), 'release_manifest_sha256': file_hash(release_dir / 'artifact_hashes.json'),
            'bank_sha256': file_hash(args.bank), 'source_sha256': source_hashes,
            'created_utc': datetime.now(timezone.utc).isoformat()}
        require(armed['target_was_absent'], 'C output appeared before the execution guard armed')
        write_json(output / 'armed.json', armed)
        inputs.recheck()
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/eval_response_v7.py'),
                                    '--prepared', str(prepared), '--output', str(run_path)], check=False)
        require(completed.returncode == 0, f'guarded C acquisition exited {completed.returncode}')
        require(inputs.read(run_path / 'metadata.json')['status'] == 'complete', 'guarded C acquisition incomplete')
        inputs.recheck()
        receipt = {'schema_version': 'response_v7_C_execution_receipt/1', 'status': 'execution_complete',
            'run': str(run_path), 'armed_path': str(output / 'armed.json'), 'armed_sha256': file_hash(output / 'armed.json'),
            'artifact_manifest_sha256': file_hash(run_path / 'artifact_hashes.json'),
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'independent_replay_still_required': True}
        write_json(output / 'execution_receipt.json', receipt)
        write_json(run_path / 'analysis/execution_receipt.json', receipt)
        meta['status'] = 'complete'; code = 0
    except Exception as error:
        meta.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'failure.json', {'error': meta['error'], 'traceback': traceback.format_exc()})
    finally:
        write_json(output / 'metadata.json', meta); write_json(output / 'input_hashes.json', inputs.hashes)
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
            for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps({'status': meta['status'], 'guard': str(output), 'run': str(run_path), 'error': meta.get('error')}, ensure_ascii=False))
    return code


def calibrate_stage(args, output, inputs, contract, meta):
    bank, guard, provenance = load_bank(inputs, args.bank, contract)
    bank_sha = file_hash(args.bank); release_dir = Path(args.release).resolve()
    inputs.manifest(release_dir)
    release_meta = inputs.read(release_dir / 'metadata.json')
    require(release_meta['status'] == 'complete' and release_meta['stage'] == 'predict', 'four-C prediction release is incomplete')
    release = inputs.read(release_dir / 'release.json')
    require(release['bank_sha256'] == bank_sha and release['guard_sha256'] == provenance['guard_sha256'], 'release uses another bank or guard')
    require(set(release['prepared']) == set(contract['pipeline']['calibration_contexts']), 'release omitted a C parent')
    inventory = context_inventory(inputs, args.runs, 'calibration', contract)
    for cid, record in inventory.items():
        verify_execution_receipt(inputs, record['path'], release_dir, bank_sha)
        require(str(Path(record['meta']['prepared']).resolve()) == release['prepared'][cid]['path'], 'C run differs from released preparation')
        require(record['meta']['prepared_manifest_sha256'] == release['prepared'][cid]['artifact_manifest_sha256'], 'released C preparation changed')
    packets, acquisition_sources = verified_packets(inputs, inventory, contract, bank_sha=bank_sha)
    require(acquisition_sources == provenance['acquisition_source_sha256'], 'T/C acquisition source versions differ')
    all_predictions = {}
    for packet in packets:
        h = packet['history']; raw, guarded, decision = guarded_predictions(bank, h, guard)
        for mode, values in (('raw', raw), ('guarded', guarded)):
            saved = inputs.read(release_dir / h.history_id / f'{mode}_predictions.json')
            require(set(saved) == set(values), 'released prediction inventory differs')
            for name in values:
                np.testing.assert_allclose(values[name], saved[name], rtol=0, atol=1e-12)
        require(inputs.read(release_dir / h.history_id / 'choices.json') ==
                {'raw': select(raw, packet['routes']), 'guarded': select(guarded, packet['routes']), 'ood': decision}, 'released guarded/raw choices changed')
        actual_sealed = inputs.read(packet['folder'] / 'predictions.json')
        for name in raw:
            np.testing.assert_allclose(raw[name], actual_sealed[name], rtol=0, atol=1e-12)
        all_predictions[h.history_id] = {'raw': raw, 'guarded': guarded, 'ood': decision}
    write_json(output / 'evaluation_prediction_seal.json', {'release_manifest_sha256': file_hash(release_dir / 'artifact_hashes.json'),
        'bank_sha256': bank_sha, 'guard_sha256': provenance['guard_sha256'], 'all_C_replays_passed': True,
        'family_outcome_tables_parsed_at_seal': inputs.outcome_tables_opened,
        'replay_status_reports_may_contain_response_fields_unused_for_prediction': True})
    targets = load_outcomes(inputs, packets)
    write_json(output / 'execution_failures.json', failure_records(packets))
    write_json(output / 'paired_information_oracle.json', paired_information_oracle(packets))
    choices = []
    for packet in packets:
        h = packet['history']; prediction = all_predictions[h.history_id]
        for mode in ('raw', 'guarded'):
            for name, values in prediction[mode].items():
                selected = select({name: values}, packet['routes'])[name]
                row = packet['outcomes'][selected]
                choices.append({'history': h.history_id, 'context_id': h.context_id, 'family': packet['family'],
                    'mode': mode, 'scorer': name, 'candidate_id': selected, 'ood_fallback': prediction['ood']['fallback'],
                    **{key: row[key] for key in ('new_area_m2', 'area_per_action', 'f1_gain_05cm', 'f1_gain_per_action',
                        'coverage_gain_m2', 'paid_actions', 'planned_actions', 'failure', 'returned_to_origin', 'branch_joint_auc_05cm')},
                    **ranking_metrics(values[:, 0], targets[h.history_id][:, 0])})
    aggregate = {}
    for mode in ('raw', 'guarded'):
        aggregate[mode] = {}
        for name in all_predictions[packets[0]['history'].history_id][mode]:
            rows = [r for r in choices if r['mode'] == mode and r['scorer'] == name]
            aggregate[mode][name] = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                if any(r[key] is not None for r in rows) else None for key in ('area_per_action', 'new_area_m2', 'f1_gain_05cm',
                    'f1_gain_per_action', 'coverage_gain_m2', 'branch_joint_auc_05cm', 'pairwise_accuracy', 'spearman', 'relative_area_rate_rmse')}
    write_json(output / 'choices.json', choices)
    write_json(output / 'ranking_report.json', {'raw_mechanism': aggregate['raw'], 'guarded_system_suggestion': aggregate['guarded'],
        'ood_histories': sum(v['ood']['fallback'] for v in all_predictions.values()), 'histories': len(packets),
        'selected_failures_by_mode_scorer': {mode: dict(Counter(r['scorer'] for r in choices if r['mode'] == mode and r['failure'] is not None)) for mode in ('raw', 'guarded')},
        'all_samples_retained': True, 'independent_confirmation': False, 'automatic_E_release': False})
    calibration_models, area_rows = {}, []
    costs = np.concatenate([[r['cost'] for r in p['routes']] for p in packets])
    areas = np.concatenate([[r['new_area_m2'] for r in p['outcomes']] for p in packets])
    ids = np.concatenate([[p['history'].history_id] * len(p['routes']) for p in packets])
    c = contract['pipeline']['calibration']
    for name in contract['pipeline']['fitting']['fit_methods']:
        relative = np.concatenate([all_predictions[p['history'].history_id]['raw'][name][:, 0] for p in packets])
        model = AbsoluteAreaCalibrator.fit(relative, costs, areas, ids, slope_max=c['slope_max'],
                                           intercept_bounds=c['intercept_bounds_m2_per_action'])
        calibrated = model.predict_area(relative, costs); original = model.uncalibrated_area(relative, costs)
        baseline = model.diagnostics['uncalibrated_history_equal_area_mae']; fitted = model.diagnostics['history_equal_area_mae']
        ratio = float(calibrated.sum() / areas.sum()) if areas.sum() else None
        calibration_models[name] = {**model.to_dict(), 'C_in_sample_MAE_reduction_fraction': (baseline-fitted)/baseline if baseline else None,
            'C_total_prediction_to_actual_ratio': ratio, 'zero_actual_positive_prediction_fraction_all_candidates': float(np.mean((areas == 0) & (calibrated > 0))),
            'positive_prediction_fraction_among_zero_actual': float(np.mean(calibrated[areas == 0] > 0)) if np.any(areas == 0) else None,
            'C_fit_diagnostic_pass_not_E_confirmation': bool(baseline > 0 and fitted <= .9 * baseline and ratio is not None and .5 <= ratio <= 2),
            'cost_input': 'sealed planned cost, never future actual paid cost', 'calibration_used_for_ranking': False}
        at = 0
        for p in packets:
            for route, result in zip(p['routes'], p['outcomes']):
                area_rows.append({'scorer': name, 'history': p['history'].history_id, 'candidate_id': route['candidate_id'],
                    'role': route['group'], 'planned_cost': route['cost'], 'actual_paid_cost': result['paid_actions'],
                    'failure': result['failure'], 'relative_rate': float(relative[at]), 'uncalibrated_area': float(original[at]),
                    'calibrated_area': float(calibrated[at]), 'actual_area': float(areas[at])})
                at += 1
    write_json(output / 'area_calibrators.json', calibration_models)
    write_json(output / 'area_predictions.json', area_rows)
    meta.update(bank_sha256=bank_sha, fits_ranking_model=False, fitted_calibrators=len(calibration_models),
                independent_confirmation=False, automatic_E_release=False)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    meta = {'schema_version': 'response_v7_fit_pipeline/1', 'status': 'running', 'stage': args.command,
            'started_utc': datetime.now(timezone.utc).isoformat(), 'new_physical_branches': 0,
            'independent_confirmation': False, 'changes_to_frozen_acquisition': False}
    write_json(output / 'metadata.json', meta)
    code = 0
    try:
        contract = contracts(inputs)
        meta['source_sha256'] = snapshot(output, inputs, contract)
        {'fit': fit_stage, 'predict': predict_stage, 'calibrate': calibrate_stage}[args.command](args, output, inputs, contract, meta)
        inputs.recheck()
        meta['status'] = 'complete'
    except Exception as error:
        meta.update(status='failed', error=f'{type(error).__name__}: {error}', permitted_to_advance=False)
        write_json(output / 'failure.json', {'error': meta['error'], 'traceback': traceback.format_exc(),
            'outcome_tables_opened': inputs.outcome_tables_opened,
            'parameters_or_contexts_automatically_changed': False})
        code = 2
    finally:
        meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), outcome_tables_opened=inputs.outcome_tables_opened,
                    input_hashes_checked=len(inputs.hashes))
        write_json(output / 'input_hashes.json', inputs.hashes)
        write_json(output / 'archived_mesh_inputs.json', inputs.archived_meshes)
        write_json(output / 'metadata.json', meta)
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
            for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps({'status': meta['status'], 'stage': args.command, 'output': str(output),
                      'outcome_tables_opened': inputs.outcome_tables_opened, 'error': meta.get('error')}, ensure_ascii=False))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='all 8 verified T parents; one fixed grouped CV fit')
    fit.add_argument('--runs', type=Path, nargs='+', required=True)
    predict = commands.add_parser('predict', help='release all 4 prepared C prefixes before any C acquisition')
    predict.add_argument('--prepared', type=Path, nargs='+', required=True)
    predict.add_argument('--bank', type=Path, required=True)
    calibrate = commands.add_parser('calibrate', help='all 4 verified C runs and original prediction release')
    calibrate.add_argument('--runs', type=Path, nargs='+', required=True)
    calibrate.add_argument('--bank', type=Path, required=True)
    calibrate.add_argument('--release', type=Path, required=True)
    execute = commands.add_parser('execute-calibration', help='require the four-C release before calling the unchanged acquisition runner')
    execute.add_argument('--release', type=Path, required=True)
    execute.add_argument('--bank', type=Path, required=True)
    execute.add_argument('--context', required=True)
    for command in (fit, predict, calibrate, execute):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return execute_calibration(args) if args.command == 'execute-calibration' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
