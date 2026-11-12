"""Stale sources zero immediately; no cached-command revival or hardware recovery."""
import unittest,types,struct
from test_visual_velocity_adapter import adapter,ready_gate,hardware,vision
import mission
class PauseTests(unittest.TestCase):
    def gate(self,profile):
        g=ready_gate();g.sensor_loss_profile=profile
        self.assertEqual(g.arm(2.,1002.),(True,'armed_once'))
        g.propose([.04,0,0,0,0,.1],2.001,'/nso_visual/move_base')
        return g
    def pause(self,g,end=4.):
        outputs=[]
        for tick in range(101,int(end*50)+1):
            t=tick/50;hardware(g,t);outputs.append(g.decision(t,1000+t))
        return outputs
    def test_stale_zero_persistent_pause_then_new_command_only(self):
        g=self.gate('recoverable_sensor_pause_v1');a=self.pause(g)
        self.assertIsNone(g.fault);self.assertTrue(g.armed)
        self.assertTrue(all(x[:2]==(0.,0.) for x in a if x[2].endswith('_stale')))
        self.assertIsNone(g.proposal)
        hardware(g,4.02);vision(g,4.02)
        self.assertEqual(g.decision(4.02,1004.02),(0.,0.,'await_new_proposal'))
        g.propose([.04,0,0,0,0,.1],4.021,'/nso_visual/move_base')
        hardware(g,4.04);vision(g,4.04)
        self.assertEqual(g.decision(4.04,1004.04),(.04,.1,'current_proposal'))
    def test_legacy_terminal_and_new_duration_remain(self):
        g=self.gate('legacy_terminal_v1');self.pause(g)
        self.assertEqual(g.fault,'source_loss:odom_stale')
        g=self.gate('recoverable_sensor_pause_v1');g.duration=1.;self.pause(g)
        self.assertEqual(g.fault,'duration_complete')
    def test_exact_reason_set_does_not_hide_other_failures(self):
        for reason in ['odom_missing','laser_source_missing_or_invalid','native_driver_status_stale','odom_health_stamp_mismatch']:
            g=self.gate('recoverable_sensor_pause_v1');g.sources_reason=lambda *a:reason
            self.pause(g);self.assertEqual(g.fault,'source_loss:'+reason)
        for kind in ['hardware','regression','nonfinite','identity']:
            g=self.gate('recoverable_sensor_pause_v1')
            if kind=='hardware':g.hardware(0x211,bytes([0,1,0,0,0,1,0,0]),2.02,2.02)
            if kind=='regression':g.observe('odom',1001.9,2.02,1002.02,caller=g.bridge_publisher,frame='odom',child_frame='base_link',x=0,y=0,yaw=0,v=0,w=0)
            if kind=='nonfinite':g.propose([float('nan'),0,0,0,0,0],2.02,g.proposal_publisher)
            if kind=='identity':g.propose([.04,0,0,0,0,0],2.02,'/wrong')
            self.assertIsNotNone(g.fault)
    def test_supervisor_only_valid_stale_sources(self):
        row=dict(valid=True,fault=None,source_stamp_ns=1000000000000)
        health=(1.,{k:dict(row) for k in ['odom','depth','laser_source']})
        odom=(1.,types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(to_sec=lambda:1000.))))
        fn=mission.recoverable_sensor_health_reason
        self.assertTrue(fn('bridge_health_receipt_stale',health,odom,3.,1002.))
        for reason in ['visual_map_missing','odom:invalid','laser_source_fault:x','native_driver_status_stale']:
            self.assertFalse(fn(reason,health,odom,3.,1002.))
        health[1]['odom'].update(valid=False,reason='odom_source_stale_or_missing')
        self.assertTrue(fn('odom:odom_source_stale_or_missing',health,odom,3.,1002.))
        health[1]['odom']['source_stamp_ns']=1003000000000
        self.assertFalse(fn('odom_source_stamp_stale_or_future',health,odom,3.,1002.))
        health[1]['odom']['source_stamp_ns']=1000000000000;health[1]['depth']['fault']='bad'
        self.assertFalse(fn('bridge_health_receipt_stale',health,odom,3.,1002.))
    def test_bridge_stale_laser_classification_exact_only(self):
        g=adapter.Gate(navigation_source='lidar_rgbd')
        row=dict(source='lslidar_N301',caller='/lslidar_driver_node',frame='laser_link',
                 valid=False,reason='laser_source_stale_or_missing',source_stamp_ns=1000000000000,fault=None)
        g.observe_laser_health(row,g.bridge_publisher,3.,1002.)
        self.assertEqual(g.laser_unavailable_reason,'laser_source_stale')
        row['reason']='invalid_geometry'
        g.observe_laser_health(row,g.bridge_publisher,3.,1002.)
        self.assertIsNone(g.laser_unavailable_reason)
        row['reason']='laser_source_stale_or_missing';row['source_stamp_ns']=1003000000000
        g.observe_laser_health(row,g.bridge_publisher,3.,1002.)
        self.assertIsNone(g.laser_unavailable_reason)
    def test_default_configuration_legacy(self):
        self.assertEqual(adapter.configuration({})['sensor_loss_profile'],'legacy_terminal_v1')
        self.assertEqual(adapter.Gate().sensor_loss_profile,'legacy_terminal_v1')
        with self.assertRaises(ValueError):adapter.configuration({'sensor_loss_profile':'arbitrary'})
if __name__=='__main__':unittest.main()
