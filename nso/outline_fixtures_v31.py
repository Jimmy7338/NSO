"""Frozen analytic geometry fixtures, with no scoring or project simulation.

Coordinates, triangles and metadata are synthetic test inputs, never sensor
evidence. See V31_ANALYTIC_FIXTURES_PROTOCOL_20260917.md for geometric facts.
"""
from dataclasses import dataclass
import numpy as np

from nso.box_union_geometry_v30 import BoxV30, RectFaceV30, union_exterior_faces_v30


CONTRACT = 'v31-analytic-outline-fixtures-r0'
PROTOCOL_SHA256 = '67595ba303b82d820ae669c90bae5c6fa22d3dac1be4e42953e344f67245a219'
BOX_BOUNDS = (0., 2., 0., 3., 0., 1.6)
L_BOUNDS = ((0., 2., 0., 1., 0., 1.6), (0., 1., 1., 3., 0., 1.6))
THIN_WALL_BOUNDS = ((-.001, .001, -.001, 3.001, 0., 1.6),
                    (1.999, 2.001, -.001, 3.001, 0., 1.6),
                    (-.001, 2.001, -.001, .001, 0., 1.6),
                    (-.001, 2.001, 2.999, 3.001, 0., 1.6))
REFINEMENT = ((.5, 1., 1.5), (.75, 1.5, 2.25), (.4, .8, 1.2))


@dataclass(frozen=True)
class MeshFixtureV31:
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        original = np.asarray(self.triangles)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('finite [N,3] vertices required')
        if original.ndim != 2 or original.shape[1] != 3 or original.dtype.kind not in 'iu':
            raise ValueError('integer [M,3] triangle indices required')
        triangles = np.array(original, dtype=np.int64, copy=True)
        if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError('triangle index outside vertices')
        vertices.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, 'vertices', vertices)
        object.__setattr__(self, 'triangles', triangles)


@dataclass(frozen=True)
class OutlineFixtureV31:
    name: str
    reference_name: str
    prediction: MeshFixtureV31
    expected: dict


def empty_mesh_v31():
    return MeshFixtureV31(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))


def mesh_from_faces_v31(faces, *, alternate_diagonal=False):
    """Weld exact coordinates, triangulate each rectangle without closing gaps."""
    vertices, triangles, index = [], [], {}
    for face in faces:
        ids = []
        for point in face.vertices():
            if point not in index:
                index[point] = len(vertices)
                vertices.append(point)
            ids.append(index[point])
        local = ((0, 1, 3), (1, 2, 3)) if alternate_diagonal else ((0, 1, 2), (0, 2, 3))
        triangles.extend(tuple(ids[k] for k in row) for row in local)
    return MeshFixtureV31(np.asarray(vertices, dtype=float).reshape(-1, 3),
                          np.asarray(triangles, dtype=np.int64).reshape(-1, 3))


def mesh_from_boxes_v31(boxes, *, vertical_only=False, split_planes=None, alternate_diagonal=False):
    return mesh_from_faces_v31(union_exterior_faces_v30(boxes, vertical_only=vertical_only,
        split_planes=split_planes), alternate_diagonal=alternate_diagonal)


def translated_mesh_v31(mesh, offset):
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('finite xyz translation required')
    return MeshFixtureV31(mesh.vertices + offset, mesh.triangles)


def combine_meshes_v31(*meshes):
    if not meshes:
        return empty_mesh_v31()
    vertices, triangles, offset = [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles + offset)
        offset += len(mesh.vertices)
    return MeshFixtureV31(np.concatenate(vertices), np.concatenate(triangles))


def _duplicate(mesh):
    return MeshFixtureV31(mesh.vertices, np.concatenate((mesh.triangles, mesh.triangles)))


def _convex_shortcut():
    # Explicit negative fixture, not a hull/completion operation on predictions.
    polygon = ((0., 0.), (2., 0.), (2., 1.), (1., 3.), (0., 3.))
    n = len(polygon)
    vertices = [(x, y, z) for z in (0., 1.6) for x, y in polygon]
    triangles = []
    for i in range(1, n-1):
        triangles += [(0, i+1, i), (n, n+i, n+i+1)]
    for i in range(n):
        j = (i+1) % n
        triangles += [(i, j, n+j), (i, n+j, n+i)]
    return MeshFixtureV31(np.asarray(vertices), np.asarray(triangles, dtype=np.int64))


def reference_meshes_v31():
    """Fresh closed references; category and observation count are absent."""
    return {'box': mesh_from_boxes_v31([BoxV30(BOX_BOUNDS, 0)]),
            'l_shape': mesh_from_boxes_v31([BoxV30(b, 0) for b in L_BOUNDS])}


def geometry_fixtures_v31():
    """Name -> fixture. `expected` states geometry, never a Q target."""
    refs = reference_meshes_v31()
    box, l_shape = refs['box'], refs['l_shape']
    box_solids = [BoxV30(BOX_BOUNDS, 0)]
    box_vertical = mesh_from_boxes_v31(box_solids, vertical_only=True)
    l_vertical = mesh_from_boxes_v31([BoxV30(b, 0) for b in L_BOUNDS], vertical_only=True)
    faces = union_exterior_faces_v30(box_solids, vertical_only=True)
    missing_side = mesh_from_faces_v31([f for f in faces if not (f.axis == 0 and f.sign == 1)])
    strip = mesh_from_faces_v31([RectFaceV30((2.4, 3.2, 1., 1., .4, 1.2), 1, 1, 0)])
    fixtures = {}

    def add(name, prediction, reference='box', **expected):
        fixtures[name] = OutlineFixtureV31(name, reference, prediction, expected)

    add('box_closed', box, closed=True, volume_m3=9.6, surface_area_m2=28.,
        footprint_area_m2=6., same_geometry_as_reference=True)
    add('box_vertical', box_vertical, closed=False, surface_area_m2=16.,
        complete_vertical_boundary=True, xy_triangle_area_m2=0., xy_linework_closed=True)
    add('box_retriangulated', mesh_from_boxes_v31(box_solids, split_planes=REFINEMENT,
        alternate_diagonal=True), same_geometry_as='box_closed', surface_area_m2=28.)
    add('box_vertical_retriangulated', mesh_from_boxes_v31(box_solids, vertical_only=True,
        split_planes=REFINEMENT, alternate_diagonal=True), same_geometry_as='box_vertical', surface_area_m2=16.)
    add('box_duplicate', _duplicate(box), same_geometry_as='box_closed',
        raw_triangle_area_m2=56., duplicate_triangle_multiplier=2)
    add('box_vertical_duplicate', _duplicate(box_vertical), same_geometry_as='box_vertical',
        raw_triangle_area_m2=32., duplicate_triangle_multiplier=2)
    add('box_four_thin_walls', mesh_from_boxes_v31([BoxV30(b, 0) for b in THIN_WALL_BOUNDS]),
        closed=True, volume_m3=.032, surface_area_m2=32.04, thickness_m=.002,
        outer_dimensions_m=[2.002, 3.002, 1.6], xy_material_area_m2=.02,
        xy_outer_boundary_enclosed_area_m2=6.010004, solid_reference_equivalence_claimed=False)
    add('l_closed', l_shape, 'l_shape', closed=True, volume_m3=6.4,
        surface_area_m2=24., footprint_area_m2=4., concavity_preserved=True)
    add('l_vertical', l_vertical, 'l_shape', closed=False, surface_area_m2=16.,
        complete_vertical_boundary=True, xy_triangle_area_m2=0., xy_linework_closed=True)
    add('box_vertical_missing_side', missing_side, closed=False, surface_area_m2=11.2,
        removed_axis=0, removed_coordinate_m=2., xy_linework_closed=False)
    add('box_expanded_030', mesh_from_boxes_v31([BoxV30((-.3, 2.3, -.3, 3.3, -.3, 1.9), 0)]),
        dimensions_m=[2.6, 3.6, 2.2], outward_expansion_each_side_m=.3)
    add('box_shifted_030', translated_mesh_v31(box, (.3, 0., 0.)),
        dimensions_m=[2., 3., 1.6], translation_m=[.3, 0., 0.], center_error_inf_m=.3)
    add('l_notch_filled', box, 'l_shape', footprint_area_m2=6.,
        false_added_footprint_area_m2=2., concavity_preserved=False)
    add('l_convex_shortcut', _convex_shortcut(), 'l_shape', closed=True, volume_m3=8.,
        footprint_area_m2=5., false_added_footprint_area_m2=1., concavity_preserved=False)
    add('box_extra_detached_box', combine_meshes_v31(box,
        mesh_from_boxes_v31([BoxV30((3., 3.4, 1., 1.4, .4, .8), 0)])),
        surface_components=2, minimum_component_gap_m=1., false_added_volume_m3=.064)
    add('box_extra_open_strip', combine_meshes_v31(box, strip),
        surface_components=2, closed=False, false_added_surface_area_m2=.64,
        xy_detached_line_length_m=.8)
    add('empty', empty_mesh_v31(), empty=True, fixed_reference_retained=True)
    return fixtures


def association_fixtures_v31():
    """Evaluation-only seeds/windows, all explicitly synthetic test metadata.

    Each result has two fixed references. Never treat these seed coordinates as
    acquired observations, or use them for prediction repair/instance discovery.
    """
    box = reference_meshes_v31()['box']
    other = translated_mesh_v31(box, (5., 0., 0.))
    empty = empty_mesh_v31()

    def case(name, meshes, points, tracks, **expected):
        assets = [dict(id=0, mesh=box, bounds=[[-.1, -.1, -.1], [2.1, 3.1, 1.7]]),
                  dict(id=1, mesh=other, bounds=[[4.9, -.1, -.1], [7.1, 3.1, 1.7]])]
        return dict(name=name, assets=assets, observed_meshes=tuple(meshes),
            seeds=tuple(None if p is None else dict(observed_seed_xyz=list(p),
                provenance='analytic_fixture_not_sensor_observation') for p in points),
            observed_track_count=tracks, expected=dict(fixed_reference_count=2, **expected))

    first, second = (1., 0., .8), (6., 0., .8)
    return {
        'both_present': case('both_present', [box, other], [first, second], 2,
                             seed_matches=[[0], [1]], missing_reference_ids=[]),
        'one_instance_missing': case('one_instance_missing', [box, empty], [first, None], 1,
                                     seed_matches=[[0], []], missing_reference_ids=[1]),
        'all_instances_missing': case('all_instances_missing', [empty, empty], [None, None], 0,
                                      seed_matches=[[], []], missing_reference_ids=[0, 1]),
        'duplicate_seed': case('duplicate_seed', [box, other], [first, first], 2,
                               seed_matches=[[0], [0]], duplicate_reference_ids=[0],
                               missing_unique_associations=[0, 1])}
