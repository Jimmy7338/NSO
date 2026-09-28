"""Bounded sensor response bench, not navigation or a calibrated ZED simulator.

One planar exterior, same central support, same Open3D TSDF settings as V18.
Reference-resolution disparity noise is converted to metric depth BEFORE fusion.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.reconstruction_metrics import surface_samples, ray_scene


CONFIG = dict(version='v19_sensor_probe', width=96, height=72, render_fx_px=48.,
    reference_fx_px=480., baseline_m=.12, voxel_m=.04, truncation_m=.12,
    max_depth_m=5., seeds=[1901, 1902, 1903], ranges_m=[4., 1.], frames=24,
    models={'ideal': [0., 0.], 'iid_010px': [.10, 0.],
            'iid_025px': [.25, 0.], 'bias_025px_iid_025px': [.25, .25]},
    plans=['far_one', 'far_copied_24', 'far_fresh_24', 'near_one', 'near_fresh_24'],
    predicted_crop_half_width_m=.4, reference_half_width_m=.3,
    note='Generic stereo sensitivity assumptions, NOT measured ZED parameters. '
         'Reference pixel units use fx=480; output raster uses fx=48. '
         'Perfect pose, planar static exterior, no materials/dropout/semantic input. '
         'Frame counts exclude travel costs: sensor response only, not policy efficacy.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def depth_sequence(distance, sigma, bias, seed):
    rng = np.random.default_rng(seed)
    shape = (CONFIG['frames'], CONFIG['height'], CONFIG['width'])
    fb = CONFIG['reference_fx_px'] * CONFIG['baseline_m']
    disparity = fb / distance + bias + rng.normal(0., sigma, shape)
    depth = fb / disparity
    return np.where((disparity > 0) & (depth > .15) & (depth < 5.), depth, 0.).astype('float32')


def integrate(sequence, distance):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=CONFIG['voxel_m'], sdf_trunc=CONFIG['truncation_m'],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    h, w = sequence.shape[1:]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, 48., 48., (w-1)/2, (h-1)/2)
    # Target is world z=0; optical axes align with world for this sensor bench.
    extrinsic = np.eye(4); extrinsic[2, 3] = distance
    color = o3d.geometry.Image(np.full((h, w, 3), 153, np.uint8))
    for depth in sequence:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,
            o3d.geometry.Image(depth.copy()), depth_scale=1., depth_trunc=5.,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, extrinsic)
    return volume.extract_triangle_mesh()


def evaluate(mesh):
    crop = mesh.crop(o3d.geometry.AxisAlignedBoundingBox([-.4, -.4, -.5], [.4, .4, .5]))
    points, _ = surface_samples(crop, 12000, 2026)
    axis = np.linspace(-.3, .3, 61)
    x, y = np.meshgrid(axis, axis)
    reference = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype('float32')
    errors = points[:, 2] if len(points) else np.empty(0)
    recall_dist = (ray_scene(mesh).compute_distance(o3d.core.Tensor(reference), nthreads=1).numpy()
                   if len(mesh.triangles) else np.full(len(reference), np.inf))
    output = dict(predicted_samples=len(points), reference_samples=len(reference),
        mean_abs_error_m=float(np.abs(errors).mean()) if len(errors) else None,
        signed_mean_error_m=float(errors.mean()) if len(errors) else None,
        rmse_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        p95_abs_error_m=float(np.percentile(np.abs(errors), 95)) if len(errors) else None,
        fixed_reference_distance_m=float(recall_dist.mean()) if np.isfinite(recall_dist).all() else None)
    for threshold in (.02, .05, .10):
        p = float(np.mean(np.abs(errors) <= threshold)) if len(errors) else 0.
        r = float(np.mean(recall_dist <= threshold))
        output[f'{round(threshold*100):02d}cm'] = dict(precision=p, recall=r,
            f1=2*p*r/(p+r) if p+r else 0.)
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    (root/'config.json').write_text(json.dumps(CONFIG, indent=2)+'\n')
    manifest = dict(status='running', configuration_sha256=digest(root/'config.json'),
        source_sha256={str(p): digest(p) for p in [Path(__file__), Path('utils/reconstruction_metrics.py')]},
        open3d_version=o3d.__version__, numpy_version=np.__version__)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(root/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in manifest['source_sha256']: archive.write(path, Path(path).name)
    rows = []
    for model, (sigma, bias) in CONFIG['models'].items():
        for seed in CONFIG['seeds']:
            far = depth_sequence(4., sigma, bias, seed)
            near = depth_sequence(1., sigma, bias, seed)
            np.savez_compressed(root/f'{model}_{seed}_depth.npz', far=far, near=near)
            sequences = [(far[:1], 4.), (np.repeat(far[:1], 24, axis=0), 4.),
                         (far, 4.), (near[:1], 1.), (near, 1.)]
            for plan, (sequence, distance) in zip(CONFIG['plans'], sequences):
                mesh = integrate(sequence, distance)
                row = dict(model=model, seed=seed, plan=plan, distance_m=distance,
                           frame_count=len(sequence), metrics=evaluate(mesh))
                row['mesh_sha256'] = hashlib.sha256(np.asarray(mesh.vertices).tobytes()
                    + np.asarray(mesh.triangles).tobytes()).hexdigest()
                rows.append(row)
            print(f'{model} seed={seed}: 5 conditions complete', flush=True)
    (root/'measurements.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary = []
    for model in CONFIG['models']:
        for plan in CONFIG['plans']:
            metrics = [r['metrics'] for r in rows if r['model']==model and r['plan']==plan]
            summary.append(dict(model=model, plan=plan,
                mean_abs_error_mm=1000*np.mean([m['mean_abs_error_m'] for m in metrics]),
                p95_abs_error_mm=1000*np.mean([m['p95_abs_error_m'] for m in metrics]),
                signed_mean_error_mm=1000*np.mean([m['signed_mean_error_m'] for m in metrics]),
                f1_05cm=np.mean([m['05cm']['f1'] for m in metrics]),
                f1_02cm=np.mean([m['02cm']['f1'] for m in metrics])))
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    manifest.update(status='complete', reconstructions=len(rows),
        artifact_sha256={p.name:digest(p) for p in root.iterdir() if p.name!='manifest.json'})
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__': main()
