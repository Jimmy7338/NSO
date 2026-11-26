"""Capture stopped actual inputs, then declare no-deadline/no-return v4.

No navigation/map/base restart, goal or velocity command is issued here.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/ouc/nso_robot')
WORK = ROOT / 'workspaces/nso_hybrid_20261029'
CAPTURE = ROOT / 'analyses/sensor_repair_20261108/nov06_restore_trial/prepare_capture.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map-source-receipt', type=Path,
                        default=ROOT/'analyses/resume_20261115/processes.json')
    parser.add_argument('--retained-navigation-map', action='store_true',
                        help='Current Nov15 map inherited after earlier physical trials; research ledger remains fresh')
    parser.add_argument('--reuse-capture', type=Path,
                        help='Explicit existing fresh capture directory, only when root confirms its stopped pose/boot still apply')
    args = parser.parse_args()
    if os.environ.get('ROS_MASTER_URI') != 'http://127.0.0.1:11382':
        parser.error('Use existing isolated master http://127.0.0.1:11382')
    if args.reuse_capture is not None:
        parser.error('Nov15 requires a fresh stopped capture; do not reuse prior captures')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    map_receipt = json.loads(args.map_source_receipt.read_text())
    if map_receipt.get('boot_id') != boot:
        parser.error('Map source receipt must identify the current boot')
    assert map_receipt.get('depth_projection_profile') == 'source_tf_v1'
    assert map_receipt.get('display_map_source') == 'navpair'
    assert map_receipt.get('navigation_map_fresh') is True
    args.output.mkdir(parents=True, exist_ok=False)
    fresh = args.reuse_capture.resolve() if args.reuse_capture else args.output / 'fresh'
    # Existing capture's 600 s limit only serializes calibration metadata; it does
    # not arm or execute a 600 s mission. Preserve its original output verbatim.
    if args.reuse_capture is None:
        subprocess.run(['/usr/bin/python3', '-s', '-B', str(CAPTURE), '--output', str(fresh),
                        '--episode', args.episode, '--budget', '40', '--duration', '600',
                        '--map-source-receipt', str(args.map_source_receipt)], check=True)
    settings = json.loads((fresh/'configuration.json').read_text())
    source = json.loads((fresh/'source.json').read_text())
    original_transform = copy.deepcopy(settings['configuration']['planning_from_odom'])
    original_anchor = copy.deepcopy(settings['configuration']['return_anchor'])
    config = settings['configuration']
    config.update(policy_variant='ros_viewpoint_exploration_v4_unbounded', total_budget=None,
                  max_viewpoint_attempts=None, nominal_planning_horizon=None, score_mode='S',
                  execution_limits_profile='exploration_until_stopped_v1',
                  return_policy='no_automatic_return',
                  episode_id=args.episode, scene_id='exhibition_unknown_full_surroundings_20261115')
    calibration = settings['calibration']
    assert (calibration['depth_confidence'], calibration['depth_texture_confidence']) == (50,100)
    calibration.update(policy_profile='ros_viewpoint_exploration_v4_unbounded',
        initialization_scope='Fresh actual stopped RGBD/scan/odom at current boot and physical pose; fresh Nov15 navigation map; no old research observation ledger imported',
        history_acquisition='Fresh actual observation ledger; navigation map freshly initialized in Nov15 restore epoch; no prior map or research history imported',
        localization_epoch_boot_id=boot, old_pose_or_planning_transform_reused=False,
        odometry_execution_profile='source_owned_diagnostic_v1',
        depth_projection_profile='source_tf_v1', display_map_source='navpair', navigation_global_window_m=40.0,
        restore_epoch_id=map_receipt['epoch_id'],
        navigation_map_initialization='fresh Nov15 /nso_pair/map from restore_data.py; no prior map history imported',
        external_deadline_s=None, duration_profile='exploration_until_stopped_v1',
        accounting_unit='actual ROS waypoint attempt; not nominal grid primitive',
        max_viewpoint_attempts=None, nominal_planning_horizon=None,
        surrounding_resource_profile='v4_surroundings_until_stopped_v1',
        return_policy='no_automatic_return',
        research_policy_changed=True, original_method_equivalence=False,
        input_confidence_profile='SDK_native_50_100_v1',
        native_execution_profile='native_snapshot_handoff_v1',
        execution_clipping_profile='nso_dwa_linear_consistency_005_v1',
        feedback_speed_profile='measured_020_v1',
        sensor_loss_profile='recoverable_sensor_pause_v1',
        ordinary_reverse_profile='ordinary_ros_reverse_005_v1', ordinary_reverse_user_authorized=True,
        ROS_unknown_profile='ros_live_local_v3_retained_own_geometry',
        reverse_recovery_profile='v3 existing ROS recovery only; no primitive-coordinator backoff claimed',
        correction_reverse_profile='ordinary ROS reverse +/-0.05m/s; no primitive endpoint correction scope',
        rtabmap_shadow_initially_running=False, rtabmap_navigation_authority=False,
        evidence_scope='Declared unbounded v4 full-viewpoint exploration, no overall/per-goal deadline, waypoint cap, roundtrip budget or automatic return. One existing scene with a fresh Nov15 navigation map; complete real outcomes/costs required. No original-method equivalence, matched benefit or independently verified coverage accuracy.')
    assert config['planning_from_odom'] == original_transform
    assert config['return_anchor'] == original_anchor == source['grid']['return_anchor']
    assert config['planning_from_odom'] == source['grid']['planning_from_odom']
    if args.retained_navigation_map:
        calibration.update(initialization_scope='Fresh current stopped RGBD/scan/odom research initialization; current Nov15 navigation map inherited after NSO01 and independent ROS escape; no old research ledger imported',
            history_acquisition='Fresh research observation ledger; navigation map retains NSO01 and independent escape history in same Nov15 epoch',
            navigation_map_initialization='Retained current Nov15 /nso_pair/map after NSO01 and independent ROS_escape_20261115_01; no map reset',
            evidence_scope='Unbounded v4 integration trial with inherited same-scene navigation map and fresh research ledger; not an independent matched repeat, original-method equivalence or coverage accuracy proof.')
    save(args.output/'configuration.json', settings)
    protocol = dict(schema='nso.full_surroundings_waypoint_protocol.v1', frozen_unix_s=time.time(),
        episode=args.episode, boot_id=boot, research_policy='ros_viewpoint_exploration_v4_unbounded',
        execution_controller='ros_viewpoint', score_mode='S', max_actual_waypoint_attempts=None,
        nominal_planning_horizon=None, external_duration_s=None,
        duration_profile='exploration_until_stopped_v1',
        surrounding_resource_profile='v4_surroundings_until_stopped_v1',
        return_policy='no_automatic_return', roundtrip_budget=None, per_goal_timeout_s=None,
        resource_change='User explicitly removed overall/per-goal time limits, actual waypoint count, nominal roundtrip budget and automatic return. Hardware/source freshness protections remain separate and unchanged.',
        purpose='Actively explore surrounding unknown space and measured surfaces with real full ROS viewpoint goals',
        fresh_source=str(fresh/'source.json'), fresh_source_sha256=digest(fresh/'source.json'),
        configuration=str(args.output/'configuration.json'), configuration_sha256=digest(args.output/'configuration.json'),
        initial_actual_base_odom_pose=calibration['initial_actual_base_odom_pose'],
        old_coordinates_reused=False, retained_navigation_map=args.retained_navigation_map, navigation_map_epoch=map_receipt['epoch_id'],
        navigation_map_reset_for_this_trial=not args.retained_navigation_map, independent_matched_repeat=False, prior_research_ledger_imported=False,
        explicit_fresh_capture_reused=args.reuse_capture is not None,
        capture_reuse_contract='When reused, root explicitly confirms captured stopped pose/current boot still apply; no new pose fabricated or old observation ledger imported',
        inherited_navigation_history=['NSO_surroundings_20261115_01','ROS_escape_20261115_01'] if args.retained_navigation_map else [],
        shared_on_ABORT_backoff_enabled=False, backoff_candidate_status='withheld_not_enabled',
        geometry='Restored own ROS footprint/default; no new geometry override',
        calibration='Confirmed Nov06 composed actual TF and existing frozen ground scalar; no refit',
        SDK=[50,100], feedback_stop_profile='measured_020_v1', ordinary_reverse='ordinary_ros_reverse_005_v1',
        sensor_loss='recoverable_sensor_pause_v1', native='native_snapshot_handoff_v1',
        RTAB_initially_off=True, automatic_return_override_added=False,
        candidate_filters_added=False, scorer_terms_added=False, stopping_rules_added=False,
        attempt_accounting='Each actual complete waypoint success/failure followed by a fresh real observation charges one waypoint attempt; bootstrap uncharged; nominal grid costs are prediction units. Physical CAN distance/time include all ROS interventions; unobserved attempts reported separately, never synthesized.',
        success_claims='No full-coverage percentage without denominator; report actual saved-scan union, terminal return, actual goal outcomes and own-support quality separately',
        source_bindings={str(p):digest(p) for p in [CAPTURE, Path(__file__).resolve(),
            WORK/'config/joint_navigation_params.yaml', WORK/'runtime/mission.py',
            WORK/'runtime/ros_viewpoint_coordinator.py',
            WORK/'nso_bridge/nso_bridge/ros_viewpoint_unbounded_v4.py']})
    save(args.output/'protocol.json', protocol)
    print(json.dumps(dict(configuration=protocol['configuration'], protocol=str(args.output/'protocol.json'),
                         actual_pose=protocol['initial_actual_base_odom_pose'], no_motion=True)))


if __name__ == '__main__':
    main()
