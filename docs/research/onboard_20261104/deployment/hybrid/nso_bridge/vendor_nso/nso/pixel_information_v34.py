"""V33 saved reward tables with measured V34 image/scan information only.

No world construction or visibility calculation here. This changes the
observation partition, not the declared potential-area task function.
"""
from functools import lru_cache
import hashlib
import numpy as np


def arrays_sha256(**arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        if value.dtype.hasobject: raise ValueError('object array is not sensor evidence')
        for part in (name.encode(), value.dtype.str.encode(), str(value.shape).encode(), value.tobytes()):
            digest.update(len(part).to_bytes(8,'big')); digest.update(part)
    return digest.hexdigest()


class SavedPotentialModelV34:
    def __init__(self, table, signatures):
        self.poses = tuple(map(tuple,table['poses']))
        self.edges = tuple(tuple((name,int(node)) for name,node in links) for links in table['edges'])
        self.anchor = table['anchor']
        self.signatures = tuple(signatures)
        if len(self.signatures)!=len(self.poses): raise ValueError('one signature per complete pose required')
        self.observed_masks = tuple(int(value,16) for value in table['observed_masks_hex'])
        self.target_bits = table['target_bits']; self.targets = tuple(table['targets'])
        self.area = {int(k):float(v) for k,v in table['exact_reference_vertical_areas'].items()}
        self.floor_cells = tuple(map(tuple,table['floor_cells']))
        self.initial_mask = int(table['prefix_mask_hex'],16)
        self.prefix_nodes = tuple(table['prefix_nodes'])
        self.terminal = lru_cache(None)(self._terminal)
        if len(self.targets)!=self.target_bits or not self.area: raise ValueError('invalid saved targets')
        mask = 0
        for node in self.prefix_nodes: mask |= self.observed_masks[node]
        if mask != self.initial_mask: raise ValueError('saved prefix mask differs')

    def _terminal(self, mask):
        if mask<0 or mask.bit_length()>self.target_bits+len(self.floor_cells):
            raise ValueError('observations outside declared bitset')
        areas = {key:0. for key in self.area}
        observed = mask & ((1<<self.target_bits)-1)
        while observed:
            bit = observed & -observed
            patch = self.targets[bit.bit_length()-1]
            areas[int(patch['owner'])] += patch['area']; observed ^= bit
        per_asset = {key:min(1.,max(0.,areas[key]/total)) for key,total in self.area.items()}
        coverage = (mask>>self.target_bits).bit_count()/len(self.floor_cells)
        surface = sum(per_asset.values())/len(per_asset)
        return dict(coverage=coverage,surface=surface,joint=coverage*surface,
            feasible=coverage>=.8-1e-12,per_asset_surface=per_asset,
            scope='saved V33 ideal potential reward under measured clean pixel information; not C_map or reconstruction Q')


def information_pair_v34(models):
    from collections import deque
    a,b=models
    if (a.poses,a.edges,a.anchor)!=(b.poses,b.edges,b.anchor): raise ValueError('different legal graph')
    if a.prefix_nodes!=b.prefix_nodes: raise ValueError('different forced prefix')
    different=[n for n in range(len(a.poses)) if a.signatures[n]!=b.signatures[n]]
    d={a.anchor:0}; queue=deque([a.anchor]); previous={}
    while queue:
        node=queue.popleft()
        for action,nxt in a.edges[node]:
            if nxt not in d:
                d[nxt]=d[node]+1; previous[nxt]=(node,action); queue.append(nxt)
    shortest=min((d[n] for n in different if n in d),default=None)
    first=[]
    for node in different:
        if d.get(node)!=shortest: continue
        cursor=node; actions=[]
        while cursor!=a.anchor:
            cursor,action=previous[cursor]; actions.append(action)
        first.append(dict(node=node,pose=a.poses[node],actions=actions[::-1]))
    return dict(prefix_equal=all(a.signatures[n]==b.signatures[n] for n in a.prefix_nodes),
        prefix_different_indices=[i for i,n in enumerate(a.prefix_nodes) if a.signatures[n]!=b.signatures[n]],
        differing_pose_count=len(different),first_information_action_layer=shortest,
        first_information_witnesses=first,all_poses_connected=len(d)==len(a.poses))
