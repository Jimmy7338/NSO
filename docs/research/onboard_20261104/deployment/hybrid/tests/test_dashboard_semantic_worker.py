"""Offline geometry/display contracts; no Torch, models, ROS or hardware."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

path = Path(__file__).resolve().parents[1]/'runtime/dashboard_semantic_worker.py'
spec = importlib.util.spec_from_file_location('dashboard_semantic_worker', path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def instance(label='chair', mask=None, **kwargs):
    return dict(source_label=label, detector_score=.73, instance_index=2,
                assigned_mask=np.ones((2, 2), dtype=bool) if mask is None else mask, **kwargs)


def packet(stamp=10_000_000_000):
    return dict(rgb=np.zeros((2,2,3),dtype=np.uint8),depth_m=np.full((2,2),2.),
                intrinsic=np.eye(3),world_from_camera=np.eye(4),stamp_ns=stamp)


class Frontend:
    def __init__(self, instances):self.instances=instances;self.calls=[]
    def predict(self, rgb, depth, **kwargs):
        self.calls.append((rgb,depth,kwargs))
        return dict(instances=self.instances)


class SemanticDisplayTest(unittest.TestCase):
    def test_actual_mask_depth_projects_rotated_odom_center_and_visible_hull(self):
        p=packet();p['world_from_camera']=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,0],[0,0,0,1.]])
        out=worker.project_instances([instance()],p['depth_m'],p['intrinsic'],p['world_from_camera'],p['stamp_ns'])
        self.assertEqual(len(out),1);obj=out[0]
        self.assertEqual((obj['x'],obj['y']),(9.,21.))
        self.assertEqual(set(map(tuple,obj['polygon'])),{(8.,20.),(10.,20.),(8.,22.),(10.,22.)})
        self.assertEqual(obj['measured_support_pixels'],4)
        self.assertEqual(obj['label_zh'],'椅子')
        self.assertEqual(obj['id'],'10000000000:2')
        self.assertEqual(obj['last_seen_stamp'],10.)
        self.assertIn('not_complete_object_outline',obj['polygon_kind'])

    def test_invalid_measured_depth_is_removed_without_zero_filling(self):
        depth=np.array([[2.,np.nan],[np.inf,-1.]])
        out=worker.project_instances([instance()],depth,np.eye(3),np.eye(4),10)
        self.assertEqual(out[0]['measured_support_pixels'],1)
        self.assertEqual((out[0]['x'],out[0]['y']),(0.,0.))
        self.assertEqual(out[0]['polygon'],[])
        self.assertEqual(worker.project_instances([instance()],np.zeros((2,2)),np.eye(3),np.eye(4),10),[])

    def test_empty_detections_floor_wall_and_bbox_only_do_not_invent_labels(self):
        p=packet()
        detections=[instance('floor'),instance('wall'),dict(source_label='cabinet',detector_score=.9,box_xyxy=[0,0,2,2])]
        self.assertEqual(worker.project_instances([],p['depth_m'],p['intrinsic'],p['world_from_camera'],10),[])
        self.assertEqual(worker.project_instances(detections,p['depth_m'],p['intrinsic'],p['world_from_camera'],10),[])
        result=worker.infer_packet(Frontend([]),p,now=10.)
        self.assertEqual(result['objects'],[]);self.assertEqual(result['status'],'ok')

    def test_empty_assigned_ownership_does_not_fall_back_to_larger_mask(self):
        p=packet();obj=instance(mask=np.zeros((2,2),dtype=bool),valid_depth_mask=np.ones((2,2),dtype=bool))
        self.assertEqual(worker.project_instances([obj],p['depth_m'],p['intrinsic'],p['world_from_camera'],10),[])
        del obj['assigned_mask']
        result=worker.project_instances([obj],p['depth_m'],p['intrinsic'],p['world_from_camera'],10)
        self.assertEqual(result[0]['mask_source'],'valid_depth_mask')
        self.assertEqual(result[0]['measured_support_pixels'],4)

    def test_bad_masks_transforms_and_intrinsics_are_rejected(self):
        p=packet()
        for bad in [np.ones((3,3),dtype=bool),np.ones((2,2),dtype=float)]:
            with self.assertRaises(ValueError):worker.project_instances([instance(mask=bad)],p['depth_m'],p['intrinsic'],p['world_from_camera'],10)
        for k in [np.zeros((3,3)),np.full((3,3),np.nan)]:
            with self.assertRaises(ValueError):worker.project_instances([],p['depth_m'],k,p['world_from_camera'],10)
        t=np.eye(4);t[0,0]=2
        with self.assertRaises(ValueError):worker.project_instances([],p['depth_m'],p['intrinsic'],t,10)

    def test_stale_future_and_slow_inference_keep_original_stamp_without_objects(self):
        p=packet();frontend=Frontend([instance()])
        for now,status in [(26.,'stale'),(9.,'error')]:
            result=worker.infer_packet(frontend,p,now=now,max_age=15.)
            self.assertEqual((result['stamp'],result['status'],result['objects']),(10.,status,[]))
        self.assertEqual(frontend.calls,[])
        with patch.object(worker.time,'time',side_effect=[10.,26.]):
            result=worker.infer_packet(frontend,p,max_age=15.)
        self.assertEqual((result['stamp'],result['status'],result['objects']),(10.,'stale',[]))

    def test_each_frame_replaces_current_detections_no_persistent_tracking(self):
        frontend=Frontend([instance()]);p=packet()
        first=worker.infer_packet(frontend,p,now=10.)
        frontend.instances=[];p['stamp_ns']=11_000_000_000
        second=worker.infer_packet(frontend,p,now=11.)
        self.assertEqual(len(first['objects']),1);self.assertEqual(second['objects'],[])
        self.assertEqual(second['stamp'],11.)
        self.assertFalse(first['camera_extrinsics_verified']);self.assertTrue(first['display_only'])

    def test_atomic_output_replaces_single_file_and_failed_write_preserves_prior_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'objects.json'
            worker.atomic_json(out,worker.base_payload(10_000_000_000,'ok'))
            worker.atomic_json(out,worker.base_payload(11_000_000_000,'stale'))
            self.assertEqual(json.loads(out.read_text())['stamp'],11.)
            with self.assertRaises(ValueError):worker.atomic_json(out,dict(stamp=float('nan')))
            self.assertEqual(json.loads(out.read_text())['stamp'],11.)
            self.assertEqual([p.name for p in Path(tmp).iterdir()],['objects.json'])

    def test_packet_validation_preserves_source_stamp_and_rejects_wrong_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'frame.npz';p=packet()
            np.savez(f,**p,frame='odom',frame_id=str(p['stamp_ns']))
            loaded=worker.packet_from_npz(f)
            self.assertEqual(loaded['stamp_ns'],p['stamp_ns'])
            self.assertEqual(loaded['frame_id'],str(p['stamp_ns']))
            np.savez(f,**p,frame='map',frame_id=str(p['stamp_ns']))
            with self.assertRaises(ValueError):worker.packet_from_npz(f)

    def test_once_failure_writes_error_with_source_stamp_and_no_model_or_fake_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'frame.npz';out=Path(tmp)/'objects.json';p=packet()
            np.savez(f,**p)
            with patch.object(worker,'make_frontend',side_effect=RuntimeError('mock model failed')),patch.object(worker.time,'time',return_value=10.):
                status=worker.main(['--input',str(f),'--output',str(out),'--once'])
            row=json.loads(out.read_text());self.assertEqual(status,1)
            self.assertEqual((row['stamp'],row['objects'],row['status']),(10.,[],'error'))
            self.assertIn('mock model failed',row['error'])

    def test_large_hull_is_bounded_and_uses_actual_finite_points(self):
        angles=np.linspace(0,2*np.pi,5000,endpoint=False);points=np.column_stack([np.cos(angles),np.sin(angles)])
        hull=worker.convex_hull_xy(points)
        self.assertLessEqual(len(hull),32)
        self.assertTrue(all(np.isfinite(v).all() for v in hull))
        self.assertTrue(all(any(np.array_equal(v,p) for p in points) for v in hull))


if __name__=='__main__':unittest.main()
