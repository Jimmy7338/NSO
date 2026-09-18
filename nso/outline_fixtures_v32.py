"""Finite synthetic noise/gap fixtures, never observations or metric queries."""
from dataclasses import dataclass
import numpy as np

from nso.box_union_geometry_v30 import BoxV30, RectFaceV30, union_exterior_faces_v30
from nso.outline_fixtures_v31 import (MeshFixtureV31, BOX_BOUNDS, L_BOUNDS,
    THIN_WALL_BOUNDS, mesh_from_faces_v31, mesh_from_boxes_v31, translated_mesh_v31)


CONTRACT = 'v32-analytic-noise-gap-fixtures-r0'
PROTOCOL_SHA256 = '33f5aa878474def8d24c071f7c4aa5a0694b4aa1539f3f0579d6c80f81e5961b'
GAP_MM = (0, 1, 5, 10, 20, 50, 100, 300)
TRANSLATION_MM = (1, 5, 10, 20, 49, 50, 51)
ROTATIONS = (('0p5deg', .5), ('1deg', 1.), ('2deg', 2.))
NEIGHBOR_GAP_MM = (10, 50, 200)
JITTER_MM = (1, 5, 10)
JITTER_SEED = 320917
OPEN_SHELL_BOUNDS = ((0., 2., 0., .002, 0., 1.6),
                     (0., 2., 2.998, 3., 0., 1.6),
                     (1.998, 2., 0., 3., 0., 1.6),
                     (0., .002, .010, 3., 0., 1.6))


@dataclass(frozen=True)
class NoiseFixtureV32:
    name: str
    references: tuple
    predictions: tuple
    labels: dict
    facts: dict


def noise_fixture_names_v32():
    return tuple([f'corner_gap_{mm:03d}mm' for mm in GAP_MM]
        + [f'translate_x_{mm:03d}mm' for mm in TRANSLATION_MM]
        + ['rotate_z_'+name for name, _ in ROTATIONS]
        + ['one_vertical_side_missing']
        + [f'neighbors_gap_{mm:03d}mm'+suffix for mm in NEIGHBOR_GAP_MM for suffix in ('', '_wrong_bridge')]
        + ['concave_l_vertical']
        + [f'independent_face_jitter_{mm:03d}mm' for mm in JITTER_MM]
        + ['ambiguity_closed_solid_missing_corner', 'ambiguity_true_open_shell_same_observation',
           'thin_walls_continuous_002mm', 'thin_walls_001mm_opening'])


def _corner_gap(faces, gap_m):
    changed = []
    for face in faces:
        if face.axis == 0 and face.sign == -1:
            bounds = list(face.bounds)
            bounds[2] = float(gap_m)
            face = RectFaceV30(tuple(bounds), face.axis, face.sign, face.owner)
        changed.append(face)
    return mesh_from_faces_v31(changed)


def noise_fixtures_v32():
    """Fresh finite catalogue with no Q thresholds or category input.

    References, declared transforms and `facts` are evaluator-side test truth.
    Only prediction meshes may enter a prediction representation function.
    `reference_seed_xyz` is synthetic association metadata, not paid evidence.
    """
    solid = [BoxV30(BOX_BOUNDS, 0)]
    box = mesh_from_boxes_v31(solid)
    faces = union_exterior_faces_v30(solid, vertical_only=True)
    vertical = mesh_from_faces_v31(faces)
    cases = {}

    def add(name, predictions, *, references=(box,), family, facts=None,
            seeds=((1., 0., .8),), **labels):
        information = dict(reference_count=len(references), prediction_slot_count=len(predictions),
            reference_seed_xyz=[list(point) for point in seeds],
            seed_provenance='analytic_fixture_not_sensor_observation',
            references_and_facts_for_evaluation_only=True)
        information.update(facts or {})
        cases[name] = NoiseFixtureV32(name, tuple(references), tuple(predictions),
            dict(family=family, synthetic_only=True, q_gate_declared=False, **labels), information)

    for mm in GAP_MM:
        gap = mm/1000.
        add(f'corner_gap_{mm:03d}mm', (_corner_gap(faces, gap),), family='missing_corner_support',
            facts=dict(gap_m=gap, removed_surface_area_m2=1.6*gap,
                predicted_surface_area_m2=16.-1.6*gap, exact_xy_loop_closed=(mm == 0)))
    for mm in TRANSLATION_MM:
        offset = (mm/1000., 0., 0.)
        add(f'translate_x_{mm:03d}mm', (translated_mesh_v31(vertical, offset),),
            family='rigid_translation', facts=dict(translation_xyz_m=list(offset),
                center_error_inf_m=mm/1000., physical_dimensions_m=[2., 3., 1.6],
                surface_area_m2=16., explicitly_samples_5cm_boundary=mm in (49, 50, 51)))
    center = np.array([1., 1.5, .8])
    for suffix, degrees in ROTATIONS:
        radians = np.deg2rad(degrees)
        rotation = np.array([[np.cos(radians), -np.sin(radians), 0.],
                             [np.sin(radians), np.cos(radians), 0.], [0., 0., 1.]])
        prediction = MeshFixtureV31((vertical.vertices-center)@rotation.T+center, vertical.triangles)
        add('rotate_z_'+suffix, (prediction,), family='rigid_rotation', facts=dict(
            angle_degrees=degrees, axis=[0., 0., 1.], center_xyz_m=center.tolist(),
            rotation_matrix=rotation.tolist(), surface_area_m2=16.,
            metric_must_not_inverse_align=True))
    missing = mesh_from_faces_v31([f for f in faces if not (f.axis == 0 and f.sign == 1)])
    add('one_vertical_side_missing', (missing,), family='whole_side_missing',
        facts=dict(removed_axis=0, removed_coordinate_m=2., removed_surface_area_m2=4.8,
                   predicted_surface_area_m2=11.2))
    for mm in NEIGHBOR_GAP_MM:
        gap = mm/1000.
        second = translated_mesh_v31(box, (2.+gap, 0., 0.))
        refs, seeds = (box, second), ((1., 0., .8), (3.+gap, 0., .8))
        add(f'neighbors_gap_{mm:03d}mm', refs, references=refs, seeds=seeds,
            family='two_independent_instances', facts=dict(minimum_instance_gap_m=gap,
                exact_corresponding_slot_geometry=True, cross_instance_union_allowed=False))
        bridge = BoxV30((2., 2.+gap, 1.35, 1.65, .65, .95), 0)
        first_with_bridge = mesh_from_boxes_v31([*solid, bridge])
        add(f'neighbors_gap_{mm:03d}mm_wrong_bridge', (first_with_bridge, second),
            references=refs, seeds=seeds, family='two_instances_with_false_connection',
            facts=dict(minimum_reference_gap_m=gap, bridge_bounds=list(bridge.bounds),
                false_added_volume_m3=.09*gap, modified_prediction_slot=0,
                physical_bridge_contacts_both=True, cross_instance_union_allowed=False))
    l_boxes = [BoxV30(bounds, 0) for bounds in L_BOUNDS]
    add('concave_l_vertical', (mesh_from_boxes_v31(l_boxes, vertical_only=True),),
        references=(mesh_from_boxes_v31(l_boxes),), family='true_concavity',
        facts=dict(reference_footprint_area_m2=4., reference_volume_m3=6.4,
                   predicted_surface_area_m2=16., concavity_preserved=True))
    base_offsets = np.random.default_rng(JITTER_SEED).uniform(-1., 1., size=(len(faces), 3))
    for mm in JITTER_MM:
        amplitude = mm/1000.
        offsets = base_offsets*amplitude
        displaced = []
        for face, offset in zip(faces, offsets):
            shifted = tuple(value+offset[i//2] for i, value in enumerate(face.bounds))
            displaced.append(RectFaceV30(shifted, face.axis, face.sign, face.owner))
        add(f'independent_face_jitter_{mm:03d}mm', (mesh_from_faces_v31(displaced),),
            family='independent_planar_face_translation', facts=dict(seed=JITTER_SEED,
                coordinate_bound_m=amplitude, euclidean_bound_m=float(np.sqrt(3)*amplitude),
                face_order_axis_sign=[[f.axis, f.sign] for f in faces], face_offsets_xyz_m=offsets.tolist(),
                predicted_surface_area_m2=16., face_count=4, faces_remain_rectangles=True,
                shared_random_realization_across_amplitudes=True, independent_replication_count=1))
    observation = cases['corner_gap_010mm'].predictions[0]
    ambiguity = dict(input_equivalence_group='missing_corner_or_true_open_shell',
        observed_corner_gap_m=.010, identical_input_mesh=True,
        information_scope='prediction_mesh_only_not_full_depth_or_free_ray_history',
        no_connectivity_certification_possible_from_this_pair=True,
        reference_material_boundary_watertight=True)
    add('ambiguity_closed_solid_missing_corner', (observation,), family='mesh_only_ambiguity',
        facts=dict(ambiguity, true_opening_m=0., enclosure_open=False,
                   reference_kind='closed_solid_box'))
    open_shell = mesh_from_boxes_v31([BoxV30(bounds, 0) for bounds in OPEN_SHELL_BOUNDS])
    add('ambiguity_true_open_shell_same_observation', (observation,), references=(open_shell,),
        family='mesh_only_ambiguity', facts=dict(ambiguity, true_opening_m=.008, enclosure_open=True,
            material_thickness_m=.002, reference_kind='open_enclosure_with_closed_material_boundary'))
    continuous = mesh_from_boxes_v31([BoxV30(bounds, 0) for bounds in THIN_WALL_BOUNDS])
    opened_bounds = [list(bounds) for bounds in THIN_WALL_BOUNDS]
    opened_bounds[0][2] = .002
    opened = mesh_from_boxes_v31([BoxV30(tuple(bounds), 0) for bounds in opened_bounds])
    add('thin_walls_continuous_002mm', (continuous,), family='thin_material_representation',
        facts=dict(thickness_m=.002, actual_opening_m=0., enclosure_open=False,
            material_boundary_watertight=True, material_volume_m3=.032,
            reference_is_closed_box=True, physical_solid_equivalence_claimed=False))
    add('thin_walls_001mm_opening', (opened,), family='thin_material_representation',
        facts=dict(thickness_m=.002, actual_opening_m=.001, enclosure_open=True,
            material_boundary_watertight=True, removed_material_volume_m3=.002*.001*1.6,
            reference_is_closed_box=True, physical_solid_equivalence_claimed=False))
    if tuple(cases) != noise_fixture_names_v32() or len(cases) != 33:
        raise AssertionError('frozen fixture catalogue differs')
    return cases
