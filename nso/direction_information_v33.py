"""Finite geometry tables for V33. No simulator, learned score, or reconstruction.

Every legal integer-grid pose is retained. Rewards integrate actual union
exterior patches; signatures contain physical geometry, never class/owner/area.
The ground-center mask is an analytic visibility proxy, not measured map C.
"""
from functools import lru_cache
import hashlib
import json
import math
import numpy as np
from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30
from scripts.preflight_scene_information_v29 import swept_clear, occludes, bfs, route

EPS = 1e-7
CAMERA_HEIGHT = .9
CAMERA_RANGE = 4.
HORIZONTAL_FOV = 90.
VERTICAL_FOV = math.degrees(2*math.atan(.75))
HORIZONTAL_SPACING = .5
VERTICAL_SPACING = .5
RADAR_HEIGHT = .25
RADAR_RANGE = 8.
RADAR_RAYS = 180
ROBOT_RADIUS = .2
HEADINGS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def boxes_for(parent, hypothesis):
    boxes = [BoxV30(tuple(bounds), None) for bounds in parent['background_boxes']]
    for asset in parent['hypotheses'][hypothesis]['assets']:
        boxes.extend(BoxV30(tuple(b), int(asset['id'])) for b in asset['boxes'])
    return tuple(boxes)


def sample_faces(faces):
    samples = []
    for face in faces:
        tangents = [i for i in range(3) if i != face.axis]
        lengths = [face.bounds[2*a+1]-face.bounds[2*a] for a in tangents]
        counts = [math.ceil(length/(VERTICAL_SPACING if axis == 2 else HORIZONTAL_SPACING))
                  for axis, length in zip(tangents, lengths)]
        for i in range(counts[0]):
            for j in range(counts[1]):
                point = [0., 0., 0.]
                point[face.axis] = face.bounds[2*face.axis]
                for axis, length, index, count in zip(tangents, lengths, (i, j), counts):
                    point[axis] = face.bounds[2*axis]+length*(index+.5)/count
                point = tuple(round(x, 10) for x in point)
                samples.append(dict(point=point, normal=face.normal, owner=face.owner,
                    area=face.area/(counts[0]*counts[1]), vertical=face.axis != 2,
                    physical_key=point+face.normal))
    return samples


def camera_visible(origin, heading, patch, bounds):
    delta = tuple(patch['point'][i]-origin[i] for i in range(3))
    front = HEADINGS[heading]; right = (front[1], -front[0])
    axial = sum(delta[i]*front[i] for i in (0, 1))
    lateral = sum(delta[i]*right[i] for i in (0, 1))
    if (axial <= EPS or abs(lateral) > axial+EPS or abs(delta[2]) > .75*axial+EPS
        or sum(d*d for d in delta) > CAMERA_RANGE**2+EPS
        or sum(-delta[i]*patch['normal'][i] for i in range(3)) <= EPS):
        return False
    return not any(occludes(origin, patch['point'], b) for b in bounds)


def radar_ranges(cell, bounds):
    hits = []
    x, y = cell
    planar = [b for b in bounds if b[4]-EPS <= RADAR_HEIGHT <= b[5]+EPS]
    for index in range(RADAR_RAYS):
        angle = 2*math.pi*index/RADAR_RAYS
        direction = (math.cos(angle), math.sin(angle)); nearest = RADAR_RANGE
        for b in planar:
            near, far = 0., RADAR_RANGE
            for axis, origin in enumerate((x, y)):
                step = direction[axis]
                if abs(step) < EPS:
                    if not b[2*axis]-EPS <= origin <= b[2*axis+1]+EPS:
                        far = -1.; break
                else:
                    a, c = sorted(((b[2*axis]-origin)/step, (b[2*axis+1]-origin)/step))
                    near, far = max(near, a), min(far, c)
            if 0. <= near <= far+EPS and far >= 0.:
                nearest = min(nearest, near)
        hits.append(round(nearest, 10))
    return tuple(hits)


class DirectionGeometryV33:
    def __init__(self, parent, hypothesis):
        self.parent, self.hypothesis = parent, hypothesis
        boxes = boxes_for(parent, hypothesis)
        bounds = tuple(box.bounds for box in boxes)
        self.blocks = bounds
        all_boxes = [b for h in (0, 1) for b in boxes_for(parent, h)]
        shared_splits = tuple(sorted({v for b in all_boxes for v in b.bounds[2*a:2*a+2]}) for a in range(3))
        faces = union_exterior_faces_v30(boxes, split_planes=shared_splits)
        self.surfaces = sample_faces(faces)
        self.targets = [p for p in self.surfaces if p['owner'] is not None and p['vertical']]
        if not self.targets:
            raise ValueError('nonempty actual exterior surface target required')
        self.owners = tuple(sorted({p['owner'] for p in self.targets}))
        self.areas = {owner:sum(p['area'] for p in self.targets if p['owner'] == owner) for owner in self.owners}
        self.target_index = {p['physical_key']:i for i, p in enumerate(self.targets)}
        if len(self.target_index) != len(self.targets):
            raise ValueError('duplicate physical quadrature point; shared partition invalid')
        cells = tuple(sorted(tuple(p) for p in parent['nav_cells']))
        if len(set(cells)) != len(cells) or any(type(c) is not int for cell in cells for c in cell):
            raise ValueError('unique integer grid coordinates required')
        # Independently reject a hand-selected centerline that omits legal
        # integer positions inside the declared rectangular bounds.
        complete = tuple((x, y) for x in range(min(c[0] for c in cells), max(c[0] for c in cells)+1)
            for y in range(min(c[1] for c in cells), max(c[1] for c in cells)+1)
            if swept_clear((x,y), (x,y), bounds, ROBOT_RADIUS))
        if cells != complete:
            raise ValueError('navigation list omits/adds safe positions in its declared grid rectangle')
        self.cells = cells; cell_set = set(cells)
        self.poses = tuple((x, y, heading) for x, y in cells for heading in range(4))
        self.pose_index = {p:i for i, p in enumerate(self.poses)}
        self.anchor = self.pose_index[tuple(parent['anchor'])]
        edges = []
        for x, y, heading in self.poses:
            dx, dy = HEADINGS[heading]; other = (x+dx, y+dy)
            links = []
            if other in cell_set and swept_clear((x,y), other, bounds, ROBOT_RADIUS):
                links.append(('forward', self.pose_index[(*other, heading)]))
            links.extend((('left', self.pose_index[(x,y,(heading-1)%4)]),
                          ('right', self.pose_index[(x,y,(heading+1)%4)])))
            edges.append(tuple(links))
        self.edges = tuple(edges)
        reachable, _ = bfs(self.edges, self.anchor)
        self.all_safe_connected = len(reachable) == len(self.poses)
        # Coverage denominator is every reachable center, never selected
        # route points; all-safe-connected is a separate prerequisite.
        self.floor_cells = tuple(c for c in cells if self.pose_index[(*c, 0)] in reachable)
        self.target_bits = len(self.targets)
        self.target_mask = (1 << self.target_bits)-1
        self.floor_visibility = {}; self.radar_signatures = {}
        for cell in cells:
            origin = (*cell, RADAR_HEIGHT); seen = []
            for i, other in enumerate(self.floor_cells):
                endpoint = (*other, RADAR_HEIGHT)
                if sum((cell[a]-other[a])**2 for a in (0,1)) <= RADAR_RANGE**2+EPS and not any(
                        occludes(origin, endpoint, b) for b in bounds):
                    seen.append(i)
            self.floor_visibility[cell] = tuple(seen)
            self.radar_signatures[cell] = radar_ranges(cell, bounds)
        self.signatures, self.signature_counts, self.observed_masks = [], [], []
        self.camera_target_masks = []
        for x, y, heading in self.poses:
            visible = [p for p in self.surfaces if camera_visible((x,y,CAMERA_HEIGHT), heading, p, bounds)]
            camera_signature = sorted(p['physical_key'] for p in visible)
            target = 0
            for p in visible:
                index = self.target_index.get(p['physical_key'])
                if index is not None: target |= 1 << index
            floor = self.floor_visibility[(x,y)]
            mask = target | sum(1 << (self.target_bits+i) for i in floor)
            signature = dict(camera_physical_points_normals=camera_signature,
                radar_ranges_world_angles=self.radar_signatures[(x,y)],
                visible_floor_coordinates=[self.floor_cells[i] for i in floor])
            self.signatures.append(digest(signature))
            self.signature_counts.append(dict(camera_points=len(visible), target_points=target.bit_count(),
                                              visible_floor_centers=len(floor)))
            self.observed_masks.append(mask); self.camera_target_masks.append(target)
        self.signatures = tuple(self.signatures); self.observed_masks = tuple(self.observed_masks)
        self.terminal = lru_cache(None)(self._terminal)

    def _terminal(self, mask):
        seen_area = {owner:0. for owner in self.owners}
        target = mask & self.target_mask
        while target:
            bit = target & -target; p = self.targets[bit.bit_length()-1]
            seen_area[p['owner']] += p['area']; target ^= bit
        per_asset = {owner:min(1.,max(0.,seen_area[owner]/self.areas[owner])) for owner in self.owners}
        coverage = (mask >> self.target_bits).bit_count()/len(self.floor_cells)
        surface = sum(per_asset.values())/len(per_asset)
        return dict(coverage=coverage, surface=surface, joint=coverage*surface,
            feasible=coverage >= .8-1e-12, per_asset_surface=per_asset,
            scope='finite ground-center visibility times potential exterior area; not measured C or Q')

    def prefix(self):
        node = self.anchor; nodes = [node]; observed = self.observed_masks[node]
        for action in self.parent['prefix_actions']:
            if action not in dict(self.edges[node]): raise ValueError('illegal shared prefix')
            node = dict(self.edges[node])[action]; nodes.append(node); observed |= self.observed_masks[node]
        if node != self.anchor: raise ValueError('prefix does not return to exact pose and heading')
        return nodes, observed

    def tables(self):
        nodes, mask = self.prefix()
        distances, _ = bfs(self.edges, self.anchor)
        visible = 0
        for node in distances: visible |= self.camera_target_masks[node]
        missing = [dict(index=i, **p) for i,p in enumerate(self.targets) if not visible & (1 << i)]
        return dict(hypothesis=self.hypothesis, poses=self.poses, edges=self.edges, anchor=self.anchor,
            floor_cells=self.floor_cells, target_bits=self.target_bits, targets=self.targets,
            exact_reference_vertical_areas=self.areas,
            observed_masks_hex=[hex(m) for m in self.observed_masks],
            signature_sha256=self.signatures, signature_counts=self.signature_counts,
            all_safe_connected=self.all_safe_connected, reachable_poses=len(distances),
            all_target_representatives_observable=not missing, unobservable_targets=missing,
            prefix_nodes=nodes, prefix_signatures=[self.signatures[n] for n in nodes],
            prefix_mask_hex=hex(mask), prefix_terminal=self.terminal(mask),
            semantic_cue_measured=False, geometric_pose_error=False)


def pair_geometry(models):
    a, b = models; shared = a.poses == b.poses and a.edges == b.edges
    prefixes = [m.prefix() for m in models]
    paired = shared and all(a.signatures[x] == b.signatures[y]
        for x,y in zip(prefixes[0][0], prefixes[1][0]))
    distances, previous = bfs(a.edges, a.anchor)
    differing = [node for node in distances if a.signatures[node] != b.signatures[node]] if shared else []
    first = min((distances[n] for n in differing), default=None)
    witnesses = [dict(node=n, pose=a.poses[n], actions=route(previous,a.anchor,n),
        signature_sha256=[m.signatures[n] for m in models])
        for n in differing if distances[n] == first]
    return dict(identical_legal_graph=shared, identical_prefix_geometry=paired,
        all_safe_connected=all(m.all_safe_connected for m in models),
        first_information_action_layer=first, first_information_witnesses=witnesses,
        signature_includes=['camera_visible_physical_points_and_normals_including_background_and_horizontal_surfaces',
                            '180_single_plane_range_samples', 'visible_ground_center_coordinates'],
        excluded_from_signature=['class','owner','hidden_area','reference_normalization','ground_truth_quality'],
        coverage_scope='declared reachable integer floor centers; not continuous ground area or measured C80',
        optimum_scope='all legal actions in shared finite grid; not a bound for arbitrary continuous motion')
