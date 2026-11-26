"""B9 IID object generator specification, private to generation/evaluation."""
import hashlib
import numpy as np

SEED=2026112009
N_OBJECTS=48
CONDITIONS={'front':(0.,),'front_side':(0.,35.,65.),
            'around':(0.,35.,65.,120.,180.),'occluded':(0.,35.,65.)}
SENSOR_CONFIG=dict(resolution_m=.2,robot_radius_m=.2,camera_height_m=1.2,
    width_px=120,height_px=90,fov_deg=90.,max_depth_m=8.,depth_sigma_m=0.,
    dropout=0.,action_duration_s=1.,voxel_m=.06,max_steps=256,
    laser_height_m=.25,laser_rays=180,truncation_m=.12)
SCENE=dict(shape=[50,50],front_anchor_world=[5.,5.,0.],camera_radius_m=2.8,
    look_at_height_m=1.0,floor=[0.,0.,-.02,10.,10.,.02],
    background_walls=[[0.,9.8,0.,10.,.2,2.8],[0.,0.,0.,.2,10.,2.8]],
    occluder=[5.15,4.05,0.,.55,.10,1.65])
DEPTH_STRATA=((.24,.48),(.50,.86),(.90,1.30))

def opaque(*parts):
    return hashlib.sha256(('|'.join(map(str,parts))+'|NSO-independent-interval-followup-v1').encode()).hexdigest()[:16]

def objects():
    records=[]
    for index in range(N_OBJECTS):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,index]))
        family=('closed','open')[int(rng.integers(2))]
        stratum=int(rng.integers(3));mismatch=bool(rng.integers(2))
        levels=int(rng.integers(3,6))
        fractions=np.linspace(.06,.90,levels)
        fractions[1:-1]+=rng.uniform(-.035,.035,levels-2)
        record=dict(object_id=opaque(SEED,index),object_draw_index=index,
            split='calibration' if index<24 else 'test',family=family,
            depth_stratum=stratum,shape_regime='part_mismatch' if mismatch else 'regular_panels',
            width_m=float(rng.uniform(.70,1.90)),depth_m=float(rng.uniform(*DEPTH_STRATA[stratum])),
            height_m=float(rng.uniform(.78,2.02)),panel_thickness_m=float(rng.uniform(.022,.055)),
            post_width_m=float(rng.uniform(.035,.065)),shelf_level_bottom_fractions=fractions.tolist(),
            rear_brace=bool(rng.integers(2)),yaw_deg=float(rng.uniform(-40.,40.)),
            door_recess_m=float(rng.uniform(.008,.04)),
            # Part-mismatch latent variables are sampled for every draw, so no
            # result-dependent redraw or branch-specific RNG consumption occurs.
            door_segment_depth_fractions=rng.uniform(.03,.26,3).tolist(),
            shelf_depth_fractions=rng.uniform(.55,.95,levels).tolist(),
            shelf_setback_allocations=rng.uniform(0.,1.,levels).tolist(),
            partial_back_width_fraction=float(rng.uniform(.3,.65)),
            partial_back_height_fraction=float(rng.uniform(.25,.55)),
            partial_back_lower_fraction=float(rng.uniform(.12,.35)))
        records.append(record)
    return records

def specification():
    return dict(schema='B9_independent_interval_followup_v1',seed=SEED,objects=objects(),
        object_sampling='Independent draws from one frozen mixture; no quota balancing, rejection, result-dependent redraw or split reassignment.',
        public_mixture=dict(family_probability=[.5,.5],depth_stratum_probability=[1/3]*3,
            mismatch_probability=.5,depth_ranges_m=DEPTH_STRATA,width_range_m=[.70,1.90],
            height_range_m=[.78,2.02],yaw_range_deg=[-40.,40.]),
        split_rule='Draw indices 0..23 calibration, 24..47 test; identical generating distribution.',
        sensor_config=SENSOR_CONFIG,scene=SCENE,conditions=CONDITIONS,
        expected_objects=48,expected_cases=192,expected_frames=576,
        geometry_mismatch='Same envelope: closed bodies use three staggered recessed front segments; open shelves use unequal depth/setback layers and a partial back panel. Regular family remains an independent thin-panel assembly.',
        calibration_rule='Only G-neutral B8 posterior, frozen sigma0.01. s0 from B8 training12; object-level interval correction q from new calibration24; test24 evaluated once. No model/posterior/policy retuning.',
        metric_scope='Separate unchanged posterior CRPS/median error from interval coverage, width and interval score. Four conditions form one joint object-level target; paired views are not independent units.',
        limitations='Opaque clean depth, exact poses and ideal first-hit instance/category masks; synthetic mixture coverage is not real-world coverage or a planning benefit.',
        counterfactual=dict(status='unit check, not an independent calibration/test object',
            depths_m=[.30,1.10],width_m=1.4,height_m=2.1,yaw_deg=0.,
            panel_thickness_m=.025,door_recess_m=.012))
