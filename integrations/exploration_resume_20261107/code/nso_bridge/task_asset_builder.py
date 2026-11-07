"""Compile declared nominal box CAD and measured background to V35 task inputs.

Forecasts are priors, never observed reconstruction. No planner rule is changed.
CAD coordinates are transformed by odom_from_cad; boxes use x0,x1,y0,y1,z0,z1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def rigid(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,0],[0,0,0,1.]])


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3,:3].T + matrix[:3,3]


def nearest_indices(samples, values, circular=False):
    """Exact nearest selected sample without a rays-by-endpoints allocation."""
    samples=np.asarray(samples); values=np.asarray(values)
    if circular:
        samples=np.mod(samples,2*np.pi);values=np.mod(values,2*np.pi)
    order=np.argsort(samples); sorted_samples=samples[order]
    right=np.searchsorted(sorted_samples,values)
    if circular:
        left=(right-1)%len(samples); right=right%len(samples)
        ld=abs(np.arctan2(np.sin(values-sorted_samples[left]),np.cos(values-sorted_samples[left])))
        rd=abs(np.arctan2(np.sin(values-sorted_samples[right]),np.cos(values-sorted_samples[right])))
    else:
        left=np.clip(right-1,0,len(samples)-1);right=np.clip(right,0,len(samples)-1)
        ld=abs(values-sorted_samples[left]);rd=abs(values-sorted_samples[right])
    return order[np.where(ld<=rd,left,right)]


def ray_boxes(origin, directions, boxes):
    """Return ray parameter; axial depth for unnormalised pinhole directions."""
    directions = np.asarray(directions)
    nearest = np.full(directions.shape[:-1], np.inf)
    for box in boxes:
        bounds = np.asarray(box).reshape(3,2)
        near = np.zeros_like(nearest); far = np.full_like(nearest,np.inf)
        for axis in range(3):
            d = directions[...,axis]; parallel = abs(d)<1e-12
            safe = np.where(parallel,1.,d)
            a,b = (bounds[axis]-origin[axis])[:,None] / safe.ravel()[None,:]
            a,b = a.reshape(d.shape),b.reshape(d.shape)
            near = np.maximum(near,np.where(parallel,-np.inf,np.minimum(a,b)))
            far = np.minimum(far,np.where(parallel,np.inf,np.maximum(a,b)))
            far = np.where(parallel & ((origin[axis]<bounds[axis,0]) | (origin[axis]>bounds[axis,1])),-np.inf,far)
        nearest = np.minimum(nearest,np.where((far>=near)&(far>1e-8),np.where(near>1e-8,near,far),np.inf))
    return nearest


def exterior_patches(boxes):
    """Exact exterior rectangular partition, shared splits; centre quadrature."""
    boxes=np.asarray(boxes,float).reshape(-1,3,2)
    splits=[sorted(set(boxes[:,a,:].ravel())) for a in range(3)]
    # Match original DirectionGeometryV33 horizontal/vertical .5m quadrature.
    splits=[sorted(set(v for lo,hi in zip(values[:-1],values[1:])
                       for v in np.linspace(lo,hi,int(np.ceil((hi-lo)/.5))+1)))
            for values in splits]
    result=[]; keys=set()
    for box in boxes:
        for axis in (0,1):  # Original potential integrates vertical surfaces.
            tangents=[a for a in range(3) if a!=axis]
            for side,sign in ((0,-1),(1,1)):
                grids=[[v for v in splits[a] if box[a,0]<=v<=box[a,1]] for a in tangents]
                for lo,hi in zip(grids[0][:-1],grids[0][1:]):
                    for low,high in zip(grids[1][:-1],grids[1][1:]):
                        point=np.zeros(3); point[axis]=box[axis,side]
                        point[tangents]=[(lo+hi)/2,(low+high)/2]
                        outside=point.copy(); outside[axis]+=sign*1e-7
                        if np.any(np.all((outside>boxes[:,:,0])&(outside<boxes[:,:,1]),axis=1)): continue
                        normal=np.zeros(3);normal[axis]=sign
                        key=tuple(point)+tuple(normal)
                        if key in keys: continue
                        keys.add(key);result.append(dict(point=point.tolist(),normal=normal.tolist(),owner=0,area=float((hi-lo)*(high-low)),vertical=True,physical_key=list(key)))
    return result


def source_reference(path, kind, description, identifier):
    path=Path(path).resolve()
    return dict(id=identifier,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind=kind,description=description)


def build_assets(cad_path, graph_path, calibration_path, sample_path, output):
    from .task_bundle import canonical_json_sha256, load_task
    cad=json.loads(Path(cad_path).read_text()); graph=json.loads(Path(graph_path).read_text())
    calibration=json.loads(Path(calibration_path).read_text()); calibration=calibration.get('calibration',calibration)
    if not graph.get('poses'): raise ValueError('explicit nonempty physical graph required')
    binding=graph['binding']; poses=graph['poses']; n=len(poses)
    odom_from_cad=np.asarray(cad.get('odom_from_cad',cad.get('physical_from_chair')),float); cad_from_odom=np.linalg.inv(odom_from_cad)
    rgbd=calibration['rgbd']; scan=calibration['scan']; K=np.asarray(rgbd['intrinsic_3x3'])
    rows=np.asarray(rgbd['row_indices']); cols=np.asarray(rgbd['column_indices'])
    vv,uu=np.meshgrid(rows,cols,indexing='ij'); rays=np.stack(((uu-K[0,2])/K[0,0],(vv-K[1,2])/K[1,1],np.ones_like(uu)),axis=-1)
    with np.load(sample_path,allow_pickle=False) as sample:
        raw=sample['depth_m']; world_from_camera=sample['world_from_camera']
        receipt=json.loads(sample['receipt_json'].item())
        scan_source=np.asarray(sample['scan_ranges_raw']); scan_ids=np.arange(len(scan_source))
        scan_valid=np.isfinite(scan_source)&(scan_source>=scan['range_min_m'])&(scan_source<=scan['range_max_m'])
        scan_angles=scan['angle_min_rad']+scan_ids[scan_valid]*scan['angle_increment_rad']
        scan_points=np.stack((scan_source[scan_valid]*np.cos(scan_angles),scan_source[scan_valid]*np.sin(scan_angles),np.zeros(sum(scan_valid))),axis=-1)
        measured_scan=transform(transform(scan_points,np.asarray(receipt['historical_tf']['odom_from_laser']['matrix'])),cad_from_odom)
        # Only valid measured endpoints enter the shared background; no free-space fill.
        v,u=np.indices(raw.shape); valid=np.isfinite(raw)&(raw>0)
        points=np.stack(((u[valid]-K[0,2])*raw[valid]/K[0,0],(v[valid]-K[1,2])*raw[valid]/K[1,1],raw[valid]),axis=-1)
        background=transform(transform(points,world_from_camera),cad_from_odom)
    envelope=np.asarray(cad['background_target_removal_bounds']).reshape(3,2)
    keep=~np.all((background>=envelope[:,0])&(background<=envelope[:,1]),axis=1)
    background=background[keep]
    scan_keep=~np.all((measured_scan>=envelope[:,0])&(measured_scan<=envelope[:,1]),axis=1)
    background=np.concatenate((background,measured_scan[scan_keep]))
    # A fixed 2 cm voxel endpoint representation is common to both forecasts.
    _,ids=np.unique(np.floor(background/.02).astype(int),axis=0,return_index=True);background=background[ids]
    depths=np.zeros((2,n,len(rows),len(cols)),np.float32); ranges=np.zeros((2,n,len(scan['indices'])),np.float32)
    tables=[]; prefix_nodes=[graph['anchor']]
    for action in graph['prefix_actions']: prefix_nodes.append(dict(graph['edges'][prefix_nodes[-1]])[action])
    if prefix_nodes[-1]!=graph['anchor']: raise ValueError('prefix does not restore anchor')
    floor=graph['floor_cells']
    for h,hypothesis in enumerate(cad['hypotheses']):
        boxes=[b['bounds'] if isinstance(b,dict) else b for b in hypothesis['boxes']]; patches=exterior_patches(boxes); masks=[]
        for node,(x,y,heading) in enumerate(poses):
            yaw=binding['yaw_rad']+np.pi/2-heading*np.pi/2
            xy=transform([[x,y,0]],rigid(*binding['translation_xy_m'],binding['yaw_rad']))[0]
            cad_from_base=cad_from_odom@rigid(xy[0],xy[1],yaw)
            camera=cad_from_base@np.asarray(calibration['extrinsics']['base_from_camera'])
            directions=rays@camera[:3,:3].T
            depth=ray_boxes(camera[:3,3],directions,boxes)
            # Reproject measured endpoints. Unseen/invalid rays remain unknown zero.
            local=transform(background,np.linalg.inv(camera)); positive=local[:,2]>0
            local=local[positive]; pixel=local[:,:2]/local[:,2,None]
            u=K[0,0]*pixel[:,0]+K[0,2];v=K[1,1]*pixel[:,1]+K[1,2]
            ci=nearest_indices(cols,u);ri=nearest_indices(rows,v)
            spacing_u=max(1,np.max(np.diff(cols))) if len(cols)>1 else 1
            spacing_v=max(1,np.max(np.diff(rows))) if len(rows)>1 else 1
            inside=(abs(cols[ci]-u)<=spacing_u/2)&(abs(rows[ri]-v)<=spacing_v/2)
            background_depth=np.full_like(depth,np.inf)
            np.minimum.at(background_depth,(ri[inside],ci[inside]),local[inside,2])
            depth=np.minimum(depth,background_depth)
            depths[h,node]=np.where(np.isfinite(depth),depth,0)
            laser=cad_from_base@np.asarray(calibration['extrinsics']['base_from_laser'])
            angles=scan['angle_min_rad']+np.asarray(scan['indices'])*scan['angle_increment_rad']
            lrays=np.stack((np.cos(angles),np.sin(angles),np.zeros_like(angles)),axis=-1)@laser[:3,:3].T
            forecast=ray_boxes(laser[:3,3],lrays,boxes)
            # Measured 3-D endpoints in the scan plane contribute shared returns.
            bp=transform(background,np.linalg.inv(laser)); plane=abs(bp[:,2])<.02;bp=bp[plane]
            if len(bp):
                ba=np.arctan2(bp[:,1],bp[:,0]);idx=nearest_indices(angles,ba,circular=True)
                delta=np.arctan2(np.sin(angles[idx]-ba),np.cos(angles[idx]-ba))
                accepted=abs(delta)<=abs(scan['angle_increment_rad'])*max(1,np.max(np.diff(scan['indices'])))/2
                np.minimum.at(forecast,idx[accepted],np.linalg.norm(bp[accepted,:2],axis=1))
            ranges[h,node]=np.where(np.isfinite(forecast)&(forecast>=scan['range_min_m'])&(forecast<=scan['range_max_m']),forecast,0)
            mask=0
            for i,patch in enumerate(patches):
                p=np.asarray(patch['point']); delta=p-camera[:3,3]; cp=transform([p],np.linalg.inv(camera))[0]
                if cp[2]<=0 or np.dot(-delta,patch['normal'])<=0:continue
                pu=K[0,0]*cp[0]/cp[2]+K[0,2];pv=K[1,1]*cp[1]/cp[2]+K[1,2]
                if not (0<=pu<rgbd['width'] and 0<=pv<rgbd['height']):continue
                hit=ray_boxes(camera[:3,3],delta[None,:],boxes)[0]
                image_row=abs(rows-pv).argmin(); image_col=abs(cols-pu).argmin()
                if hit>=1-1e-6 and background_depth[image_row,image_col]>=cp[2]-.02: mask|=1<<i
            # Original floor-centre analytic visibility proxy at actual laser height.
            for i,cell in enumerate(floor):
                endpoint=transform([[*cell,0]],rigid(*binding['translation_xy_m'],binding['yaw_rad']))[0]
                endpoint=transform([endpoint],cad_from_odom)[0];endpoint[2]=laser[2,3]
                delta=endpoint-laser[:3,3];distance=np.linalg.norm(delta)
                laser_delta=delta@laser[:3,:3]
                angle=np.arctan2(laser_delta[1],laser_delta[0])
                ray_index=abs(np.arctan2(np.sin(angles-angle),np.cos(angles-angle))).argmin()
                if distance<=scan['range_max_m'] and (distance<1e-8 or (ray_boxes(laser[:3,3],delta[None,:],boxes)[0]>=1-1e-6 and forecast[ray_index]>=distance-.02)):mask|=1<<(len(patches)+i)
            masks.append(mask)
        prefix_mask=0
        for node in prefix_nodes:prefix_mask|=masks[node]
        tables.append(dict(poses=poses,edges=graph['edges'],anchor=graph['anchor'],targets=patches,target_bits=len(patches),exact_reference_vertical_areas={'0':sum(p['area'] for p in patches)},floor_cells=floor,observed_masks_hex=[hex(m) for m in masks],prefix_mask_hex=hex(prefix_mask),prefix_nodes=prefix_nodes))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'geometry.json').write_text(json.dumps(dict(tables=tables,scope='nominal_CAD_ideal_potential_not_reconstruction',forecast_construction=dict(background='valid measured RGBD and scan endpoints outside declared target envelope',endpoint_voxel_m=.02,background_projection='nearest selected pixel/ray within half sample spacing; minimum endpoint depth',missing_predictions='zero; no inferred measured free space',target_surface='exact box-union vertical exterior face partition with original .5m maximum centre-quadrature spacing',visibility_background_tolerance_m=.02))))
    np.savez_compressed(output/'predictions.npz',poses=np.asarray(poses,int),depth=depths,ranges=ranges,calibration_sha256=canonical_json_sha256(calibration),binding_sha256=canonical_json_sha256(binding))
    sources=[source_reference(cad_path,'nominal_public_CAD','Explicit paired nominal hypotheses; hidden geometry is not measured truth','cad'),source_reference(sample_path,'measured_RGBD','Shared observed background endpoints, target envelope removed','sample'),source_reference(graph_path,'physical_graph','Declared shared physical action graph','graph'),source_reference(calibration_path,'sensor_calibration','Declared sensor sampling and transforms','calibration')]
    manifest=dict(schema='nso_task_bundle_v35/1',task_id=cad.get('task_id',cad.get('target','physical_target')+'_nominal_public_CAD_20261029'),scope='physical_experiment',execution_allowed=bool(graph.get('execution_allowed',False)),geometry=source_reference(output/'geometry.json','table','nominal potential','geometry'),sensor_predictions=source_reference(output/'predictions.npz','forecast','prior forecast','predictions'),physical_sources=sources,hypotheses=[dict(description=h['description'],source_ids=['cad']) for h in cad['hypotheses']],prefix_actions=graph['prefix_actions'],total_budget=graph['total_budget'],binding=binding,calibration=calibration,limitations=['Single-scene real-I/O and decision/execution feasibility only; no semantic superiority claim.','Nominal CAD forecasts and ideal potential are not reconstruction accuracy.','Background uses measured RGBD and scan endpoints only; unseen and invalid rays stay unknown.'])
    path=output/'task.json';path.write_text(json.dumps(manifest,indent=2));load_task(path)
    return path


def main():
    parser=argparse.ArgumentParser()
    for name in ('cad','graph','calibration','sample','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();print(build_assets(args.cad,args.graph,args.calibration,args.sample,args.output))


if __name__=='__main__':main()
