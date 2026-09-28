#!/usr/bin/env python3
"""Comparator-only continuation; preserve the original completed TARE main.

The original failed replay compared int-key in-memory cue dictionaries to JSON
string-key dictionaries. Canonical JSON comparison fixes representation only.
No planner, sensor, mapping, evaluation or movement parameter is changed.
"""
import argparse
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_external_routes_v39 import ExperimentV39,prior
from scripts.run_tare_routes_v39 import TareExperimentV39,PHYSICS_TRACE_FIELDS,BRIDGE,PROTOCOL,OVERRIDES

OLD=ROOT/'audit_results/v39_tare_adapter_20260920'
OUTPUT=ROOT/'audit_results/v39_tare_adapter_r1_20260920'
AMENDMENT=ROOT/'docs/research/V39_TARE_REPLAY_COMPARATOR_AMENDMENT_20260920.md'


class TareContinuationV39( TareExperimentV39):
    def __init__(self,output=OUTPUT):
        super().__init__(output)

    def prepare(self):
        # Validate the existing experiment and completed main, including failure.
        original=TareExperimentV39(OLD)
        _,old_config=original.frozen()
        prior.verify_seal(OLD/'case00','main_seal.json','replay')
        prior.verify_seal(OLD/'case00/replay','seal.json')
        assert (OLD/'case00/result.json').is_file()
        assert (OLD/'case00/replay/failure.json').is_file()
        saved=prior.read(OLD/'case00/trace.json')
        failed=prior.read(OLD/'case00/replay/partial_trace.json')
        fields=lambda rows:[{k:r[k] for k in PHYSICS_TRACE_FIELDS} for r in rows]
        assert prior.digest(fields(saved))==prior.digest(fields(failed))
        additions={k:v for k,v in old_config.items() if k.startswith('native_') or k in (
            'camera_visibility_model_adapted','adapter_maximum_paid_right_turns_per_wait',
            'ros_master_port','xy_projection_maximum_error_m','early_stop_at_anchor_allowed',
            'not_fair_narrow_FOV_ranking')}
        additions.update(comparator_only_amendment=str(AMENDMENT.relative_to(ROOT)),
            completed_main_imported_indices=[0],new_main_starts_in_continuation=3,
            combined_original_and_continuation_main_starts=4,
            combined_maximum_replay_attempts=5,original_failed_replay_retained=True,
            original_completed_main_rerun=False,algorithm_parameters_unchanged=True)
        sources=[Path(__file__),PROTOCOL,OVERRIDES,BRIDGE,AMENDMENT,
            ROOT/'scripts/run_tare_routes_v39.py',ROOT/'tests/test_tare_adapter_v39.py',
            ROOT/'audit_results/v39_tare_prefix_smoke_20260920/result.json',
            ROOT/'docs/research/V39_TARE_RUNTIME_PREFLIGHT_20260920.md',
            ROOT/'third_party/official_baselines/tare_official/src/tare_planner/config/garage.yaml',
            *[OLD/name for name in ('config.json','manifest.json','prepare_seal.json')],
            *[OLD/'case00'/name for name in ('main_seal.json','result.json')],
            *[OLD/'case00/replay'/name for name in ('seal.json','failure.json','partial_trace.json')],
            *[ROOT/'audit_results/v39_tare_runtime_preflight_20260920'/name for name in
                ('installed_packages.json','environment.json','verification.json')]]
        ExperimentV39.prepare(self,methods=('TARE',),extra_sources=sources,
            scope='L2-TARE-transfer',config_overrides=additions)
        assert prior.read(self.output/'config.json')['physical_cases']==old_config['physical_cases']
        destination=self.output/'case00';destination.mkdir()
        copied={}
        for p in prior.files_under(OLD/'case00'):
            relative=p.relative_to(OLD/'case00')
            if relative.parts[0]=='replay':continue
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            # Byte copy avoids later writes through shared hardlink inodes.
            self.write_bytes(target,p.read_bytes())
            assert prior.sha(target)==prior.sha(p)
            copied[str(relative)]=prior.sha(target)
        prior.verify_seal(destination,'main_seal.json','replay')
        self.write(self.output/'imported_main_receipt.json',dict(
            source='audit_results/v39_tare_adapter_20260920/case00',
            new_worlds=0,new_sensor_queries=0,new_main_starts=0,
            same_main_trajectory_counted_once=True,copied_sha256=copied,
            original_failed_replay_scope='39 fresh packets/maps/meshes matched; comparator failed before metric evaluation'))

    def verify_replay_policy(self,trace,evidence,folder):
        saved=prior.read(folder/'trace.json')
        fields=lambda rows:[{k:r[k] for k in PHYSICS_TRACE_FIELDS} for r in rows]
        if prior.digest(fields(trace))!=prior.digest(fields(saved)):
            raise AssertionError('canonical fixed-action fresh sensor/map trace differs')
        if prior.digest(evidence['actions'])!=prior.digest(prior.read(folder/'controller.json')['actions']):
            raise AssertionError('recorded actions differ')
        self.write(folder/'replay/policy_scope.json',dict(policy_reexecuted=False,
            compared_trace_fields=list(PHYSICS_TRACE_FIELDS),fresh_native_ROS_publications=0,
            all_saved_actions_executed=True,canonical_comparison='JSON key normalization, exact values, no tolerance',
            claim='sensor/fusion/measurement repeatability only'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','main','replay','analyze'));p.add_argument('--index',type=int)
    a=p.parse_args();experiment=TareContinuationV39()
    if a.command=='prepare':experiment.prepare()
    elif a.command=='analyze':experiment.analyze()
    else:
        if a.index is None:p.error('--index required')
        if a.command=='main' and a.index==0:p.error('completed original main must not be rerun')
        experiment.run_case(a.index,replay=a.command=='replay')


if __name__=='__main__':main()
