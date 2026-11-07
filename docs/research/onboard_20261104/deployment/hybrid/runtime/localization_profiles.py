"""Explicit producer contracts only; selecting a profile never starts a source."""
DEFAULT_LOCALIZATION_SOURCE = 'lidar_rf2o'
LOCALIZATION_SOURCES = ('lidar_rf2o', 'lidar_csm')


def control_velocity_receipt(profile, reported_source=None, pose_source=None):
    """Describe the active control input; this metadata does not admit motion."""
    if profile['name'] != 'lidar_csm':
        return dict(control_velocity_source=profile['velocity_source'],
                    control_velocity_model=profile['control_velocity_model'])
    source = reported_source if reported_source is not None else 'csm_se2'
    models = dict(csm_se2=profile['control_velocity_model'],
                  can_feedback='CAN_measured_vx_w_paired_at_or_before_scan_raw_CSM_pose_and_TF_retained')
    result = dict(control_velocity_source=source,
                  control_velocity_model=models.get(source, 'unrecognized_reported_control_source'))
    if pose_source:
        result['pose_source'] = pose_source
        if pose_source.get('fixed_coordinate_transform_applied'):
            result['control_velocity_model'] = ('CAN_measured_vx_w_paired_at_or_before_scan_fixed_SE2_aligned_CSM_pose'
                if source == 'can_feedback' else 'CSM_SE2_body_twist_fixed_coordinate_alignment')
    return result


def localization_profile(name=DEFAULT_LOCALIZATION_SOURCE):
    if name == 'lidar_rf2o':
        return dict(name=name, odom_topic='/odom_lidar',
            expected_publisher='/rf2o_laser_odometry', tf_owner='/rf2o_laser_odometry',
            producer_label='RF2O', waiting_reason='waiting_for_RF2O',
            timestamp_semantics='RF2O_producer_timestamp_not_scan_acquisition',
            velocity_source='RF2O_original_twist',
            control_odom_topic='/nso_visual/odom', control_velocity_model='unprojected_source_twist',
            required_nodes=('/lslidar_driver_node', '/rf2o_laser_odometry', '/slam_gmapping'))
    if name == 'lidar_csm':
        return dict(name=name, odom_topic='/nso_csm/odom',
            expected_publisher='/nso_csm_odometry_bridge', tf_owner='/nso_csm_odometry_bridge',
            matcher_node='/nso_csm/matcher', producer_label='CSM', waiting_reason='waiting_for_CSM',
            timestamp_semantics='scan_acquisition_timestamp_preserved',
            velocity_source='CSM_derived_SE2_base_twist',
            control_odom_topic='/nso_csm/control_odom',
            control_velocity_model='nonholonomic_vy_zero_projection_raw_odom_retained',
            required_nodes=('/lslidar_driver_node', '/nso_csm/matcher',
                            '/nso_csm_odometry_bridge', '/slam_gmapping'))
    raise ValueError('unsupported localization source: '+str(name))

