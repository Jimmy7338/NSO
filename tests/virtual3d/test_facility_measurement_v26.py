"""Synthetic observation/mesh boundary checks: no world or TSDF fusion."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import open3d as o3d
from nso.cpu_sensor_contract_v10 import digest
from nso.facility_measurement_v26 import FacilityMeasurementV26, VERSION, TASK_ASSET_COUNT
from tests.virtual3d.test_observed_state_v26 import packet, red, blue


def meshes():
    return (o3d.geometry.TriangleMesh.create_box(.8,.6,1.6).translate((2.6,5.1,0.)),
            o3d.geometry.TriangleMesh.create_box(.8,.6,1.6).translate((.5,5.1,0.)))


class SnapshotMapper:
    def __init__(self,frames,raw=None):
        self.frames=frames;self.raw=o3d.geometry.TriangleMesh() if raw is None else raw
        self.surface={};self.quality={};self.mesh_calls=0
    def mesh(self):
        self.mesh_calls+=1
        return o3d.geometry.TriangleMesh(self.raw)


class ReferenceGeometry:
    """Two analytic reference boxes, no simulator/sensor capabilities."""
    def __init__(self):
        self._meshes=meshes()
        self.objects=[dict(id=0,evaluation_bounds=[[2.5,5.,.01],[3.5,5.8,1.7]]),
                      dict(id=1,evaluation_bounds=[[.4,5.,.01],[1.4,5.8,1.7]])]
        self.calls=[]
    def instance_mesh(self,identifier):
        self.calls.append(identifier);return self._meshes[identifier]
    def sense(self):raise AssertionError('test reference cannot sense')
    def scan(self):raise AssertionError('test reference cannot scan')
    def step(self,action):raise AssertionError('test reference cannot act')


class FacilityMeasurementTests(unittest.TestCase):
    def evaluate(self,helper,snapshot,**kw):
        args=dict(coverage=.9,returned=True,collisions=0,failed=False,paid_actions=0,budget=400)
        args.update(kw)
        return helper.evaluate(snapshot,ReferenceGeometry(),**args)

    def test_no_marker_retains_both_empty_slots_and_zero_main_quality(self):
        helper=FacilityMeasurementV26();p=packet(depth=0.)
        # An injected semantic label cannot substitute for an actual RGB marker.
        p=replace(p,frame=replace(p.frame,semantic=np.full((16,16),3,np.int32)))
        helper.observe(p)
        raw=meshes()[0]+meshes()[1];mapper=SnapshotMapper(1,raw)
        snapshot=helper.snapshot(mapper)
        self.assertEqual(mapper.mesh_calls,1);self.assertEqual(len(snapshot['observed_meshes']),2)
        self.assertEqual(snapshot['metadata']['missing_seed_slots'],[0,1])
        self.assertTrue(all(len(m.triangles)==0 for m in snapshot['observed_meshes']))
        before=digest(snapshot['metadata']);result=self.evaluate(helper,snapshot)
        self.assertEqual(before,digest(snapshot['metadata']))
        main=result['main_observed']
        self.assertEqual(main['mission_asset_count'],2);self.assertEqual(main['missing_asset_count'],2)
        self.assertEqual(main['05cm']['outline_macro_quality'],0.)
        self.assertEqual(main['05cm']['joint_outline'],0.)
        self.assertGreater(result['raw_secondary']['05cm']['outline_macro_quality'],.99)
        self.assertTrue(main['eligible'])  # Detection failure is Q=0, not a rewritten mission gate.
        self.assertFalse(result['evaluation_association_and_minimum_separation_passed'])
        self.assertEqual(result['fixed_seed_association']['missing_reference_ids'],[0,1])

    def test_no_observation_snapshot_reports_empty_geometry_instead_of_inventing_seeds(self):
        helper=FacilityMeasurementV26();snapshot=helper.snapshot(SnapshotMapper(0))
        result=self.evaluate(helper,snapshot,coverage=0.,returned=False)
        self.assertEqual(snapshot['metadata']['backend_observe_calls'],0)
        self.assertEqual(result['main_observed']['missing_asset_count'],2)
        self.assertEqual(result['main_observed']['05cm']['outline_macro_quality'],0.)
        self.assertFalse(result['main_observed']['eligible'])

    def test_first_real_seed_and_full_pre_discovery_history_are_retained(self):
        helper=FacilityMeasurementV26();p0=packet();p1=packet(1,colors=[red()]);p2=packet(2,colors=[blue()])
        self.assertEqual(helper.observe(p0)['first_seed_receipts'],[])
        first=helper.observe(p1)['first_seed_receipts'][0];helper.observe(p2)
        self.assertEqual(first['action_id'],1);self.assertEqual(first['actual_marker_code'],3)
        self.assertEqual(helper.seeds[0],first);self.assertIsNone(helper.seeds[1])
        self.assertEqual(len(helper.backends[0].frames),3);self.assertEqual(len(helper.backends[1].frames),3)
        self.assertEqual([f['observation_id'] for f in helper.backends[0].frames],[0,1,2])
        self.assertEqual(len(helper.backends[0].seeds),1);self.assertEqual(len(helper.backends[1].seeds),0)
        r,c=first['pixel'];d=p1.frame.depth_m[r,c]
        xyz=np.array([(c-p1.frame.intrinsic[0,2])*d/p1.frame.intrinsic[0,0],
                      (r-p1.frame.intrinsic[1,2])*d/p1.frame.intrinsic[1,1],d])
        xyz=xyz@p1.frame.world_from_camera[:3,:3].T+p1.frame.world_from_camera[:3,3]
        np.testing.assert_array_equal(first['observed_seed_xyz'],xyz)
        snap=helper.snapshot(SnapshotMapper(3))
        self.assertEqual(snap['metadata']['missing_seed_slots'],[1])
        result=self.evaluate(helper,snap,paid_actions=2)
        self.assertEqual(result['fixed_seed_association']['rows'][0]['reference_id'],0)
        self.assertEqual(result['fixed_seed_association']['rows'][1]['status'],'unassigned')
        self.assertEqual(len(result['main_observed']['instances']),2)
        self.assertTrue(snap['metadata']['inference_disabled'])
        self.assertTrue(all(row['completion']['reason']=='disabled_by_frozen_measured_only_contract'
                            for row in snap['metadata']['instances']))

    def test_invalid_or_too_small_marker_depth_does_not_create_seed(self):
        for depth in (0.,.1):
            helper=FacilityMeasurementV26();helper.observe(packet(colors=[red()],depth=depth))
            self.assertEqual(helper.seeds,[None,None])
        p=packet(colors=[red()]);p.frame.depth_m[5,5]=0.
        helper=FacilityMeasurementV26();helper.observe(p)
        self.assertEqual(helper.seeds,[None,None])

    def test_extra_observed_tracks_remain_diagnostic_without_truth_selection(self):
        colors=[(slice(5,9),slice(a,a+4),(220,60,40)) for a in (1,6,11)]
        helper=FacilityMeasurementV26();receipt=helper.observe(packet(colors=colors))
        self.assertEqual(len(receipt['first_seed_receipts']),2)
        self.assertEqual(len(receipt['extra_track_receipts']),1)
        snap=helper.snapshot(SnapshotMapper(1));result=self.evaluate(helper,snap)
        self.assertEqual(len(snap['metadata']['observed_track_summary']),3)
        self.assertEqual(len(snap['observed_meshes']),TASK_ASSET_COUNT)
        self.assertEqual(result['fixed_seed_association']['observed_track_count'],3)
        self.assertFalse(result['fixed_seed_association']['seed_association_gate_passed'])

    def test_latest_retry_does_not_duplicate_backend_frames_and_bad_order_is_rejected(self):
        helper=FacilityMeasurementV26();p=packet(colors=[red()]);receipt=helper.observe(p)
        self.assertEqual(helper.observe(p),receipt);self.assertEqual(helper.frame_count,1)
        receipt['first_seed_receipts'][0]['observed_seed_xyz'][0]=123.
        self.assertNotEqual(helper.seeds[0]['observed_seed_xyz'][0],123.)
        for wrong in (packet(2),packet(1,episode='wrong'),packet(1,frame_id=p.frame_id)):
            with self.assertRaises(ValueError):helper.observe(wrong)
            self.assertEqual(helper.frame_count,1)
        helper.observe(packet(1))
        with self.assertRaises(ValueError):helper.observe(p)
        fresh=FacilityMeasurementV26()
        with self.assertRaises(ValueError):fresh.observe(packet(1))

    def test_snapshot_is_terminal_and_requires_matching_mapper_history(self):
        helper=FacilityMeasurementV26();p=packet();helper.observe(p)
        with self.assertRaises(ValueError):helper.snapshot(SnapshotMapper(0))
        snapshot=helper.snapshot(SnapshotMapper(1))
        with self.assertRaises(ValueError):helper.observe(packet(1))
        with self.assertRaises(ValueError):helper.snapshot(SnapshotMapper(1))
        self.assertEqual(snapshot['metadata']['observed_frames'],1)

    def test_evaluation_rejects_tampering_and_does_not_collapse_to_one_task(self):
        helper=FacilityMeasurementV26();helper.observe(packet())
        snapshot=helper.snapshot(SnapshotMapper(1))
        altered=dict(snapshot);altered['metadata']=deepcopy(snapshot['metadata'])
        altered['metadata']['instances'][0]['seed']=dict(observed_seed_xyz=[3.,5.1,1.])
        with self.assertRaises(ValueError):self.evaluate(helper,altered)
        with self.assertRaises(ValueError):self.evaluate(helper,snapshot,returned='false')
        bad=dict(snapshot);bad['raw_mesh']=meshes()[0]
        with self.assertRaises(ValueError):self.evaluate(helper,bad)
        reference=ReferenceGeometry();reference.objects=reference.objects[:1]
        with self.assertRaises(ValueError):helper.evaluate(snapshot,reference,.9,True,0,False,0)
        for changes in (dict(coverage=.79),dict(returned=False),dict(collisions=1),dict(failed=True),dict(paid_actions=401)):
            with self.subTest(changes=changes):
                self.assertFalse(self.evaluate(helper,snapshot,**changes)['main_observed']['eligible'])


if __name__=='__main__':unittest.main()
