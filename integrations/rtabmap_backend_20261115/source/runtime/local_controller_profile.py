"""Pure validation of an explicitly selected existing ROS local backend."""
import math
import ast
TEB='teb_local_planner/TebLocalPlannerROS'
PROFILE='teb_slow_polygon_v1'
FOOTPRINT=[[-.392,-.2925],[.6,-.2925],[.6,.2925],[-.392,.2925]]
ONSITE_PROFILE='teb_onsite_062x059_v1'
ONSITE_FOOTPRINT=[[-.31,-.295],[.31,-.295],[.31,.295],[-.31,.295]]
PROFILE_FOOTPRINTS={PROFILE:FOOTPRINT,ONSITE_PROFILE:ONSITE_FOOTPRINT}
def checked_footprint(value, *, profile=PROFILE):
    if profile not in PROFILE_FOOTPRINTS:raise ValueError('unsupported explicit TEB geometry profile')
    """Normalize only representation; never change the declared physical polygon."""
    if isinstance(value,str):
        if len(value)>4096:raise ValueError('footprint representation too long')
        try:value=ast.literal_eval(value)
        except (ValueError,SyntaxError,TypeError,RecursionError):
            raise ValueError('invalid footprint literal') from None
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError('footprint requires exactly4vertices')
    normalized=[]
    for row,expected in zip(value,PROFILE_FOOTPRINTS[profile]):
        if not isinstance(row,(list,tuple)) or len(row)!=2:
            raise ValueError('footprint requires exactly2coordinates pervertex')
        for actual,want in zip(row,expected):
            if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual) or abs(actual-want)>1e-9:
                raise ValueError('footprint differs from actualrectangle')
        normalized.append([float(x) for x in row])
    return normalized

def validate_teb_backend(plugin, params, footprint, *, xy_tolerance=.15, yaw_tolerance=.35, profile=PROFILE):
    if plugin != TEB:raise ValueError('actual local planner is not declared TEB')
    if isinstance(yaw_tolerance,bool) or not isinstance(yaw_tolerance,(int,float)) or not math.isfinite(yaw_tolerance) or not .08<=yaw_tolerance<=.35:
        raise ValueError('expected TEB yaw tolerance must be finite in [.08,.35]')
    required={'max_vel_x':.05,'max_vel_x_backwards':.05,'max_vel_y':0.,'max_vel_theta':.2,
        'acc_lim_x':.25,'acc_lim_theta':.8,'min_turning_radius':0.,'xy_goal_tolerance':xy_tolerance,
        'yaw_goal_tolerance':yaw_tolerance,'max_global_plan_lookahead_dist':1.5,'dt_ref':1.,
        'max_samples':50,'feasibility_check_no_poses':50,'min_resolution_collision_check_angular':.05,
        'costmap_obstacles_behind_robot_dist':4.,'min_obstacle_dist':.05}
    if profile == ONSITE_PROFILE:required['penalty_epsilon']=.005
    for key,value in required.items():
        actual=params.get(key)
        if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual) or abs(actual-value)>1e-9:
            raise ValueError('TEB readback differs: '+key)
    for key in ('cmd_angle_instead_rotvel','enable_homotopy_class_planning','enable_multithreading'):
        if params.get(key) is not False:raise ValueError('TEB unsupported bool: '+key)
    if params.get('include_costmap_obstacles') is not True or params.get('odom_topic')!='/nso_csm/control_odom':
        raise ValueError('TEB obstacle/odom source mismatch')
    model=params.get('footprint_model',{})
    if model.get('type')!='polygon':
        raise ValueError('TEB optimization footprint must be polygon')
    optimizer_footprint=checked_footprint(model.get('vertices'),profile=profile)
    costmap_footprint=checked_footprint(footprint,profile=profile)
    return dict(profile=profile,actual_plugin=plugin,controller_namespace='/nso_visual/move_base/TebLocalPlannerROS',
        actual_parameters=params,footprint=costmap_footprint,optimizer_footprint=optimizer_footprint,
        original_costmap_footprint_readback=footprint,collision_equivalence_with_DWA=False,
        native_collision_semantics='lethal-1 rejected; unknown-2/outside-3 are not equivalent to DWA rejection',
        geometry_basis=('user-confirmed approximate62x59cm/nooverhang; declaredbase_linkcentred assumption, not independent laser metrology' if profile==ONSITE_PROFILE else 'legacy historical declared rectangle, not independently measured'),
        source_velocity_caps=dict(v=.05,reverse=.05,w=.2),full_discrete_check='max_samples50/feasibility50 plus native interpolated checks')
