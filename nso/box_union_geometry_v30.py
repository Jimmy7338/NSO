"""Exact rectilinear box-union boundary; metadata owner is not a semantic class."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class BoxV30:
    bounds: tuple
    owner: int | None = None

    def __post_init__(self):
        bounds = tuple(float(x) for x in self.bounds)
        if len(bounds) != 6 or not all(math.isfinite(x) for x in bounds):
            raise ValueError('six finite box endpoints required')
        if any(bounds[2*a] >= bounds[2*a+1] for a in range(3)):
            raise ValueError('box extent must be positive on every axis')
        if self.owner is not None and (isinstance(self.owner, bool) or not isinstance(self.owner, Integral)):
            raise ValueError('owner must be an integer facility identifier or None, not a class label')
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'owner', None if self.owner is None else int(self.owner))


@dataclass(frozen=True)
class RectFaceV30:
    bounds: tuple
    axis: int
    sign: int
    owner: int | None

    @property
    def area(self):
        return math.prod(self.bounds[2*a+1]-self.bounds[2*a] for a in range(3) if a != self.axis)

    @property
    def normal(self):
        return tuple(float(self.sign) if a == self.axis else 0. for a in range(3))

    def vertices(self):
        a, b = (self.axis+1)%3, (self.axis+2)%3
        out = []
        for u,v in ((0,0),(1,0),(1,1),(0,1)):
            p = [0.,0.,0.]; p[self.axis] = self.bounds[2*self.axis]
            p[a] = self.bounds[2*a+u]; p[b] = self.bounds[2*b+v]; out.append(tuple(p))
        return tuple(out if self.sign > 0 else reversed(out))


def _grid(boxes, split_planes=None):
    boxes = tuple(boxes)
    if any(not isinstance(b, BoxV30) for b in boxes):
        raise TypeError('BoxV30 inputs required')
    if not boxes:
        return (), np.zeros((0,0,0),bool), np.empty((0,0,0),object)
    extra = ((),(),()) if split_planes is None else tuple(split_planes)
    if len(extra) != 3:
        raise ValueError('split_planes must provide three coordinate collections')
    coordinates = []
    for axis in range(3):
        points = {value for box in boxes for value in box.bounds[2*axis:2*axis+2]}
        lower,upper = min(points),max(points)
        for x in extra[axis]:
            x=float(x)
            if not math.isfinite(x): raise ValueError('split planes must be finite')
            if lower < x < upper: points.add(x)
        coordinates.append(tuple(sorted(points)))
    occupied = np.zeros(tuple(len(c)-1 for c in coordinates),bool)
    owners = np.empty(occupied.shape,object); owners.fill(None)
    indices = [{v:i for i,v in enumerate(c)} for c in coordinates]
    for box in boxes:
        region=tuple(slice(indices[a][box.bounds[2*a]],indices[a][box.bounds[2*a+1]]) for a in range(3))
        taken=occupied[region]; old=owners[region]
        if any(owner != box.owner for owner in old[taken]):
            raise ValueError('positive-volume overlap has conflicting owner metadata')
        occupied[region]=True; owners[region]=box.owner
    return tuple(coordinates),occupied,owners


def union_exterior_faces_v30(boxes, *, vertical_only=False, split_planes=None):
    coordinates,occupied,owners=_grid(boxes,split_planes)
    if not coordinates: return ()
    faces=[]
    for raw in np.argwhere(occupied):
        index=tuple(map(int,raw))
        for axis in ((0,1) if vertical_only else (0,1,2)):
            for sign in (-1,1):
                adjacent=list(index); adjacent[axis]+=sign
                if 0 <= adjacent[axis] < occupied.shape[axis] and occupied[tuple(adjacent)]:
                    continue
                bounds=[v for a in range(3) for v in (coordinates[a][index[a]],coordinates[a][index[a]+1])]
                fixed=coordinates[axis][index[axis]+(sign>0)]
                bounds[2*axis]=bounds[2*axis+1]=fixed
                faces.append(RectFaceV30(tuple(bounds),axis,sign,owners[index]))
    return tuple(faces)


def union_volume_v30(boxes):
    coordinates,occupied,_=_grid(boxes)
    if not coordinates: return 0.
    sizes=[np.diff(c) for c in coordinates]
    volume=sizes[0][:,None,None]*sizes[1][None,:,None]*sizes[2][None,None,:]
    return float(volume[occupied].sum())


def surface_audit_v30(faces):
    """Closed oriented two-manifold check, including vertex links; no mesh library."""
    edges=Counter(); balance=Counter(); links=defaultdict(list); volume=0.; triangles=0
    face_count=0; area=0.
    for face in faces:
        face_count+=1; area+=face.area; v=face.vertices()
        for tri in ((v[0],v[1],v[2]),(v[0],v[2],v[3])):
            triangles+=1
            volume+=float(np.dot(tri[0],np.cross(tri[1],tri[2])))/6.
            for i in range(3):
                a,b=tri[i],tri[(i+1)%3]; key=tuple(sorted((a,b)))
                edges[key]+=1; balance[key]+=1 if a<b else -1
                links[a].append((tri[(i+1)%3],tri[(i+2)%3]))
    bad_vertices=0
    for pairs in links.values():
        graph=defaultdict(list)
        for a,b in pairs: graph[a].append(b); graph[b].append(a)
        seen=set(); todo=[next(iter(graph))]
        while todo:
            vertex=todo.pop()
            if vertex in seen: continue
            seen.add(vertex); todo.extend(graph[vertex])
        if any(len(rows)!=2 for rows in graph.values()) or len(seen)!=len(graph): bad_vertices+=1
    edge_bad=sum(n!=2 for n in edges.values()); orientation_bad=sum(n!=0 for n in balance.values())
    return dict(face_count=face_count,triangle_count=triangles,vertex_count=len(links),edge_count=len(edges),
        area=area,signed_volume=volume,non_two_sided_edges=edge_bad,inconsistent_oriented_edges=orientation_bad,
        nonmanifold_vertices=bad_vertices,closed_oriented_two_manifold=bool(face_count) and not(edge_bad or orientation_bad or bad_vertices),
        empty=not face_count)
