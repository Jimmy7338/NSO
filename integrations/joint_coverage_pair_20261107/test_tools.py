import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from prepare_arm import prepare
from compare_pair import common_known,compare,wrap,arm

class PairTools(unittest.TestCase):
    def test_known_masks_identity_and_unknown(self):
        grid=np.array([[-1,0],[1,-1]])
        mask,meta=common_known(grid,np.eye(4),np.eye(4),(2,2))
        np.testing.assert_array_equal(mask,grid!=-1)
        self.assertEqual(meta['out_of_common_domain_cells'],0)
    def test_translation_and_domain_loss_explicit(self):
        grid=np.array([[0,-1],[-1,-1]])
        transform=np.eye(4);transform[0,3]=-.2
        mask,meta=common_known(grid,transform,np.eye(4),(2,2))
        self.assertTrue(mask[0,1]);self.assertEqual(mask.sum(),1)
        transform[0,3]=-2
        mask,meta=common_known(grid,transform,np.eye(4),(2,2))
        self.assertEqual(mask.sum(),0);self.assertEqual(meta['out_of_common_domain_cells'],1)
    def test_yaw_wrap(self):
        self.assertAlmostEqual(wrap(-3.13-3.13),.023185307179586445)
    def test_prepare_preserves_actual_source_and_unique_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);source=t/'fresh.json';raw=dict(configuration=dict(policy_variant='exploration_joint_v1',planner_revision='v10_3_1',score_mode='S',total_budget=40,episode_id='capture',robot_radius_m=.7212185868375829,sensor_config={'actual':True}),calibration={'navigation_map_topic':'/root_pending_actual_map','apply_approved_self_mask':False},semantic_configuration={'model':'same_actual_frontend'},execution_allowed=False)
            source.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                a,_=prepare(source,'C','synthetic',t/'prepared',t/'reports');b,_=prepare(source,'S','synthetic',t/'prepared',t/'reports')
            x=json.loads(a.read_text());y=json.loads(b.read_text());self.assertNotEqual(x['configuration']['episode_id'],y['configuration']['episode_id'])
            self.assertFalse(x['execution_allowed']);self.assertEqual(x['calibration']['navigation_map_topic'],raw['calibration']['navigation_map_topic']);self.assertEqual(x['semantic_configuration'],y['semantic_configuration'])
            self.assertEqual(json.loads(source.read_text()),raw)
            for k in raw['configuration']:
                if k not in ['score_mode','episode_id']:self.assertEqual(x['configuration'][k],y['configuration'][k])
    def test_zero_observation_failures_retained_as_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)
            for mode in ['C','S']:
                p=t/mode;p.mkdir();(p/'events.jsonl').write_text('')
                (p/'result.json').write_text(json.dumps(dict(status='failed',reason='synthetic_init_failure',method_failed=True,goals=0,successful_arrivals=[],actual_CAN_distance_m=0.,elapsed_run_s=2.)))
                (p/'joint_configuration.json').write_text(json.dumps(dict(configuration=dict(score_mode=mode,total_budget=40,shape=[2,2],planning_from_odom=np.eye(4).tolist(),return_anchor=[0,0,0]),calibration={})))
            with contextlib.redirect_stdout(io.StringIO()):r=compare(t/'C',t/'S',t/'summary')
            self.assertTrue(r['arms']['C']['method_failed']);self.assertIsNone(r['common_coordinate_coverage_proxy']);self.assertIsNone(r['arms']['S']['observed_area_proxy_m2']['initial']);self.assertFalse(r['pair_protocol_provenance_bound'])

    def test_distinct_initial_maps_and_quality_support_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)
            for mode,j in [('C',0),('S',1)]:
                p=t/mode;p.mkdir();events=[]
                for i in [0,1]:
                    grid=np.full((2,2),-1);grid[0,j]=0
                    if i:grid[1,j]=0
                    f=p/f'{i}.npz';np.savez(f,joint_belief_grid=grid,base_odom_pose=np.array([j*.01,0.,0.]),rgbd_timestamp_s=10.+j*10+i)
                    feedback=dict(paid_actions=i,observed_2d_area_m2=.04*(i+1),old_quality_proxy=.2-i*.01,fixed_old_support_count=2,fixed_old_support_sha256=mode)
                    response=dict(accepted=True,state=dict(pose=[0,0,0]),summary=dict(runtime=dict(modules=dict(feedback=feedback))))
                    events.append(dict(method='observe',params=dict(path=str(f)),response=response))
                (p/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
                (p/'result.json').write_text(json.dumps(dict(status='synthetic',reason='unit_fixture',successful_arrivals=[],method_failed=False)))
                (p/'joint_configuration.json').write_text(json.dumps(dict(configuration=dict(score_mode=mode,total_budget=40,shape=[2,2],planning_from_odom=np.eye(4).tolist(),return_anchor=[0,0,0],policy_variant='exploration_joint_v1',planner_revision='v10_3_1'),calibration=dict(joint_coverage_pair_provenance=dict(pair_id='unit',arm=mode)))))
            with contextlib.redirect_stdout(io.StringIO()):r=compare(t/'C',t/'S',t/'out')
            self.assertEqual(r['common_coordinate_coverage_proxy']['initial_known_mask_symmetric_difference_cells'],2)
            self.assertFalse(r['initial_inputs_byte_identical']);self.assertFalse(r['old_surface_support_identical'])
            self.assertIn('do not interpret absolute C/S',r['surface_comparison_scope'])
            self.assertTrue(r['arms']['C']['fixed_old_quality_proxy']['support_held_fixed_within_arm'])
            self.assertIn('within this arm',r['arms']['S']['fixed_old_quality_proxy']['comparison_scope'])
            self.assertTrue(r['pair_protocol_provenance_bound']);self.assertAlmostEqual(r['initial_actual_odom_delta_S_minus_C']['x_m'],.01)

    def test_running_episode_is_not_final_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'episode not finalized'):
                arm(tmp)

if __name__=='__main__':unittest.main()
