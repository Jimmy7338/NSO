"""Prediction-only virtual RGB-D/TSDF acquisition operator.

Acquired past and hypothesis-rendered future are fused in new local volumes.
No supplied mapper or mesh is mutated. Geometry/mixture probabilities are the
caller's model; this operator neither fits them nor consumes source truth.
"""
from dataclasses import dataclass
import hashlib,time
import numpy as np
import open3d as o3d

@dataclass(frozen=True)
class DepthFrame:
    depth_m: np.ndarray
    world_from_camera: np.ndarray
    intrinsic: np.ndarray
    provenance: str

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def frame_hash(frame):return hashlib.sha256((digest(frame.depth_m)+digest(frame.world_from_camera)+digest(frame.intrinsic)+frame.provenance).encode()).hexdigest()
def validate_frame(f,expected):
    d=np.asarray(f.depth_m);p=np.asarray(f.world_from_camera);k=np.asarray(f.intrinsic)
    if f.provenance!=expected:raise ValueError('future frames must be rendered hypotheses; past must be acquired')
    if d.ndim!=2 or p.shape!=(4,4)or k.shape!=(3,3)or not np.isfinite(p).all()or not np.isfinite(k).all():raise ValueError('valid image, pose and intrinsics required')
    if k[0,0]<=0 or k[1,1]<=0:raise ValueError('positive focal lengths required')
    if not np.allclose(p[3],[0,0,0,1])or not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-6):raise ValueError('rigid camera pose required')

def ray_scene(mesh):
    if not len(mesh.triangles):return None
    scene=o3d.t.geometry.RaycastingScene(nthreads=1);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return scene

def render(hypothesis_scene,poses,intrinsic,width,height,max_depth_m=8.,min_depth_m=.15):
    if not np.isfinite([min_depth_m,max_depth_m]).all()or min_depth_m<0 or max_depth_m<=min_depth_m:raise ValueError('valid sensor depth interval required')
    if hypothesis_scene is None:raise ValueError('hypothesis ray scene required')
    k=np.asarray(intrinsic,float);v,u=np.mgrid[:height,:width];local=np.column_stack([(u.ravel()-k[0,2])/k[0,0],(v.ravel()-k[1,2])/k[1,1],np.ones(width*height)])
    frames=[]
    for raw in poses:
        p=np.asarray(raw,float);delta=local@p[:3,:3].T;rays=np.column_stack([np.tile(p[:3,3],(width*height,1)),delta]).astype(np.float32)
        hit=hypothesis_scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy().reshape(height,width)
        depth=np.where(np.isfinite(hit)&(hit>min_depth_m)&(hit<=max_depth_m),hit,0.).astype(np.float32)
        f=DepthFrame(depth,p.copy(),k.copy(),'predicted_hypothesis');validate_frame(f,'predicted_hypothesis');frames.append(f)
    return frames

def integrate(frames,voxel_m=.06,truncation_m=.12,max_depth_m=8.):
    vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_m,sdf_trunc=truncation_m,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for f in frames:
        k=np.asarray(f.intrinsic);h,w=f.depth_m.shape;valid=np.isfinite(f.depth_m)&(f.depth_m>0);depth=np.where(valid,f.depth_m,0.).astype(np.float32)
        rgb=o3d.geometry.Image(np.full((h,w,3),127,np.uint8));rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,o3d.geometry.Image(np.ascontiguousarray(depth)),depth_scale=1.,depth_trunc=max_depth_m,convert_rgb_to_intensity=False)
        intr=o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2]);vol.integrate(rgbd,intr,np.linalg.inv(f.world_from_camera))
    return vol.extract_triangle_mesh()

def distance(mesh,points):
    ray=ray_scene(mesh)
    if ray is None:return np.full(len(points),np.inf)
    return ray.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32)),nthreads=1).numpy()

def forecast(past_frames,future_depth_frames,template_surface_points,template_area_weights,*,voxel_m=.06,truncation_m=.12,tolerance_m=.05):
    points=np.asarray(template_surface_points,float);area=np.asarray(template_area_weights,float)
    if points.ndim!=2 or points.shape[1]!=3 or area.shape!=(len(points),)or not np.isfinite(points).all()or not np.isfinite(area).all()or np.any(area<0):raise ValueError('finite template surface points and nonnegative areas required')
    if voxel_m<=0 or truncation_m<=0 or tolerance_m<0 or not np.isfinite([voxel_m,truncation_m,tolerance_m]).all():raise ValueError('valid fixed metric scales required')
    for f in past_frames:validate_frame(f,'acquired')
    for f in future_depth_frames:validate_frame(f,'predicted_hypothesis')
    hashes=[frame_hash(f)for f in [*past_frames,*future_depth_frames]];start=time.monotonic()
    before=integrate(past_frames,voxel_m,truncation_m);after=integrate([*past_frames,*future_depth_frames],voxel_m,truncation_m)
    d0=distance(before,points);d1=distance(after,points);tol=np.float32(tolerance_m);b=d0<=tol;a=d1<=tol
    new=~b&a;lost=b&~a;gain=area*new;loss=area*lost
    assert hashes==[frame_hash(f)for f in [*past_frames,*future_depth_frames]]
    receipt=dict(schema='prediction_only_acquisition_forward_v1',past_frames=len(past_frames),future_hypothesis_frames=len(future_depth_frames),
        voxel_m=voxel_m,truncation_m=truncation_m,tolerance_m=tolerance_m,template_points=len(points),template_area_m2=float(area.sum()),
        predicted_gross_new_m2=float(gain.sum()),predicted_lost_m2=float(loss.sum()),predicted_net_new_m2=float(gain.sum()-loss.sum()),
        before_vertices_sha256=digest(np.asarray(before.vertices)),before_triangles_sha256=digest(np.asarray(before.triangles)),
        after_vertices_sha256=digest(np.asarray(after.vertices)),after_triangles_sha256=digest(np.asarray(after.triangles)),
        frames_unchanged=True,measured_mapper_mutated=False,reference_geometry_read=False,elapsed_s=time.monotonic()-start)
    return dict(gain=gain,loss=loss,before_distance=d0,after_distance=d1,receipt=receipt)
