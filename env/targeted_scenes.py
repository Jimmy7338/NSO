"""Large procedural functional buildings with locally visible category cues.

Scene authors assign room functions before observation. Policies receive only
sensor-visible salience, never the function table, room masks or hidden areas.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label
from utils.grid_geometry import inflated_obstacles

FAMILIES=('office_spine','warehouse_bays','gallery_network')
CONDITIONS=('aligned','shuffled','constant','absent','noisy')


@dataclass
class TargetedScene:
    occupancy: np.ndarray
    room_labels: np.ndarray
    semantic_fields: dict
    rooms: list
    entrances: list
    landmarks: list
    descriptors: dict


def generate_scene(family, size=192, seed=0, resolution_m=.25):
    if family not in FAMILIES or size not in (192,256):
        raise ValueError('targeted scene requires a known family and size 192/256')
    rng=np.random.default_rng(seed)
    world=np.ones((size,size),np.uint8);regions=np.zeros((size,size),np.int16)
    mid=size//2;spine_half=5;low=8;high=size-8
    world[mid-spine_half:mid+spine_half+1,low:high]=0
    bays=size//32-1
    bounds=np.linspace(low+3,high-3,bays+1,dtype=int)
    # Exactly half productive rooms; placement independent of method or outcomes.
    productive=np.array([True]*bays+[False]*bays);rng.shuffle(productive)
    room_info=[];cue_masks=[];landmarks=[]
    for side in (-1,1):
        for j in range(bays):
            index=len(room_info);room_id=index+1
            c0,c1=int(bounds[j]+2),int(bounds[j+1]-2)
            door=(c0+c1)//2+int(rng.integers(-2,3))
            boundary=mid+side*(spine_half+1)
            vestibule=12 # common prefix, function cannot be read from full room geometry at entrance
            depth=(mid-18-int(rng.integers(0,8))) if productive[index] else vestibule+int(rng.integers(6,11))
            end=boundary+side*depth
            r0,r1=sorted((boundary+side,end));r1+=1
            # Narrow common entry, then a function-dependent room.
            nr0,nr1=sorted((boundary,boundary+side*vestibule));nr1+=1
            world[nr0:nr1,door-2:door+3]=0
            rr0,rr1=sorted((boundary+side*vestibule,end));rr1+=1
            world[rr0:rr1,c0:c1]=0
            regions[rr0:rr1,c0:c1]=room_id
            if productive[index] and family=='warehouse_bays':
                for r in range(rr0+7,rr1-5,12):
                    world[r:r+2,c0+4:c1-4]=1
                    # Fixed alternation leaves routes around the shelves.
                    opening=c0+4 if (r//12)%2 else c1-8
                    world[r:r+2,opening:opening+4]=0
            elif productive[index] and family=='office_spine':
                for r in range(rr0+8,rr1-5,18):
                    world[r:r+2,c0+5:c0+8]=1
            elif productive[index] and family=='gallery_network':
                center=(rr0+rr1)//2
                world[center-1:center+2,c0+5:c0+8]=1
            # The category cue is a small visible entrance patch, not a density
            # proportional to hidden area. All cue supports are identical.
            cue=np.zeros(world.shape,bool)
            cue_row=boundary-side
            cue[cue_row-1:cue_row+2,door-3:door+4]=True
            cue_masks.append(cue)
            kind=('workplace' if family=='office_spine' else 'storage_aisle' if family=='warehouse_bays' else 'public_gallery') if productive[index] else 'utility_room'
            # One evaluator-only functional landmark, on free floor in each room.
            cells=np.argwhere((regions==room_id)&(world==0))
            point=tuple(map(int,cells[len(cells)//2]))
            landmarks.append(dict(room_id=room_id,position=list(point),productive=bool(productive[index])))
            room_info.append(dict(room_id=room_id,function=kind,productive=bool(productive[index]),
                                  side=side,column=door,bounds=[rr0,rr1,c0,c1],entrance=[cue_row,door]))
    # Selected public rooms form cross-room connections; utility rooms remain
    # terminal branches. Edges are evaluator annotations, not policy inputs.
    links=0
    if family=='gallery_network':
        for side in (-1,1):
            rooms=[r for r in room_info if r['side']==side]
            for a,b in zip(rooms,rooms[1:]):
                if not(a['productive'] and b['productive']):continue
                lo=max(a['bounds'][0],b['bounds'][0])+5
                hi=min(a['bounds'][1],b['bounds'][1])-5
                r=(lo+hi)//2
                world[r-2:r+3,a['bounds'][3]-1:b['bounds'][2]+1]=0
                links+=1
    salience=np.where(productive,.9,.1).astype(np.float32)
    permutation=np.random.default_rng(seed+90000).permutation(len(salience))
    values={'aligned':salience,'shuffled':salience[permutation],
            'constant':np.full(len(salience),.5),'absent':np.zeros(len(salience)),
            'noisy':np.where(np.random.default_rng(seed+91000).random(len(salience))<.2,1-salience,salience)}
    fields={}
    for condition in CONDITIONS:
        field=np.zeros(world.shape,np.float32)
        for mask,value in zip(cue_masks,values[condition]):field[mask]=value
        fields[condition]=field
    # Landmark coordinates can be occupied by later links only if a bug occurs.
    assert all(world[tuple(x['position'])]==0 for x in landmarks)
    blocked=inflated_obstacles(world,.2/resolution_m)
    components,n=label(~blocked)
    entrances=[(mid,low+3),(mid,high-4)]
    component=components[entrances[0]]
    if not component or any(components[p]!=component for p in entrances):raise ValueError('disconnected entrances')
    if any(not np.any((regions==r['room_id'])&(components==component)) for r in room_info):raise ValueError('unreachable room')
    for room in room_info:
        room['reachable_area_m2']=float(np.count_nonzero((regions==room['room_id'])&(components==component))*resolution_m**2)
    return TargetedScene(world,regions,fields,room_info,entrances,landmarks,
        dict(family=family,seed=seed,size=size,extent_m=size*resolution_m,
             resolution_m=resolution_m,room_count=len(room_info),productive_room_count=int(productive.sum()),
             reachable_area_m2=float(np.count_nonzero(components==component)*resolution_m**2),
             cross_room_links=links,cue_count=len(cue_masks),
             cue_task_assumption='function category predicts larger public/work area; cue strengths fixed by category, not hidden area',
             shuffled_productive_high_fraction=float(np.mean(values['shuffled'][productive]>.5)),
             noisy_realized_error_fraction=float(np.mean((values['noisy']>.5)!=productive))))
