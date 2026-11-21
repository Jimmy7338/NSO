"""Independent calibration geometry specification; generator/evaluator only."""
import hashlib
import math

DEPTHS = ((.28, .40, .34, .37), (.57, .69, .63, .66), (.94, 1.12, 1.03, 1.08))
WIDTHS = (.72, 1.08, 1.44, 1.80)
HEIGHTS = (.75, 1.10, 1.55, 2.05)
THICKNESS = (.025, .035, .045, .055)
YAWS_DEG = (-20., 0., 20., 40.)
SPLITS = ('train', 'train', 'validation', 'heldout')
CONDITIONS = {
    'front': (0.,),
    'front_side': (0., 35., 65.),
    'around': (0., 35., 65., 120., 180.),
    'occluded': (0., 35., 65.),
}
SENSOR_CONFIG = dict(resolution_m=.2, robot_radius_m=.2, camera_height_m=1.2,
    width_px=120, height_px=90, fov_deg=90., max_depth_m=8., depth_sigma_m=0.,
    dropout=0., action_duration_s=1., voxel_m=.06, max_steps=256,
    laser_height_m=.25, laser_rays=180, truncation_m=.12)
SCENE = dict(shape=[50,50], front_anchor_world=[5.,5.,0.], camera_radius_m=2.8,
    look_at_height_m=1.0, floor=[0.,0.,-.02,10.,10.,.02],
    background_walls=[[0.,9.8,0.,10.,.2,2.8],[0.,0.,0.,.2,10.,2.8]],
    occluder=[5.15,4.05,0.,.55,.10,1.65])
DEPTH_PRIOR = [round(.2+.1*i, 10) for i in range(13)]
SIGMA_GRID = [.01,.015,.025,.04,.06,.09,.14,.22]


def opaque(*parts):
    return hashlib.sha256(('|'.join(map(str,parts))+'|NSO-procedural-calibration-v1').encode()).hexdigest()[:16]


def objects():
    result=[]
    for family_index,family in enumerate(('closed','open')):
        for stratum,depths in enumerate(DEPTHS):
            for index,depth in enumerate(depths):
                wi=(index+stratum+family_index)%4
                hi=(index+2*stratum+family_index)%4
                thickness=THICKNESS[(index+stratum+2*family_index)%4]
                n_levels=3+((index+stratum+family_index)%2)
                # These nonuniform absolute-part fractions differ from the predictor.
                levels=([.07,.43,.91] if n_levels==3 else [.06,.29,.63,.93])
                result.append(dict(object_id=opaque(family,stratum,index),split=SPLITS[index],
                    family=family,depth_stratum=stratum,width_m=WIDTHS[wi],depth_m=depth,
                    height_m=HEIGHTS[hi],panel_thickness_m=thickness,
                    post_width_m=.035+.01*((index+family_index)%3),
                    shelf_level_bottom_fractions=levels,
                    rear_brace=bool((index+stratum)%2),
                    yaw_deg=YAWS_DEG[(index+stratum+family_index)%4],
                    door_recess_m=.012+.004*((index+stratum)%3)))
    return result


def specification():
    return dict(schema='B8_independent_procedural_calibration_v1',objects=objects(),
        sensor_config=SENSOR_CONFIG,scene=SCENE,conditions=CONDITIONS,
        depth_prior=dict(public_grid_nodes_m=DEPTH_PRIOR,density='fixed continuous uniform before observed-span conditioning',
            quadrature_mass='Defined/exported by the frozen method, including endpoint cell widths and any measured-span boundary; not learned from these objects.'),sigma_grid_m=SIGMA_GRID,
        calibration='Train-only equal-object, then equal-condition CRPS; ties use larger sigma. Validation and held-out never retune.',
        heldout_scope='Unseen object combinations and interpolated dimensions within the synthetic family; not natural or OOD furniture.',
        expected_objects=24,expected_cases=96,expected_frames=288,
        geometry_family='Independently assembled thin-panel enclosure and nonuniform open shelves; no ReplicaCAD mesh or predictor primitive constructor.',
        sensor_limitations='Opaque noiseless first-hit depth, exact poses, ideal visible instance mask; no natural detector or real sensor calibration claim.',
        counterfactual=dict(status='unit check, not an independent calibration object',depths_m=[.30,1.10],
            width_m=1.4,height_m=2.1,camera_height_m=1.2,look_at_height_m=1.2,yaw_deg=0.))
