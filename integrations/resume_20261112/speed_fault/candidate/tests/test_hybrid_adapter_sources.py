"""The hybrid profile must stop when raw laser evidence is lost or malformed."""
import unittest
from test_visual_velocity_adapter import adapter, ready_gate, hardware, vision


def laser(gate, now, source_time=None, **changes):
    row = dict(valid=True, fault=None, source='lslidar_N301',
               caller='/lslidar_driver_node', frame='laser_link',
               source_stamp_ns=round((1000.+(now if source_time is None else source_time))*1e9))
    row.update(changes)
    gate.observe_laser_health(row, '/nso_visual_sensor_bridge', now, 1000.+now)


class HybridAdapterSources(unittest.TestCase):
    def gate(self):
        gate=ready_gate()
        gate.navigation_source='lidar_rgbd'
        return gate

    def test_laser_missing_refuses_arm_despite_other_fresh_sources(self):
        gate=self.gate()
        self.assertEqual(gate.arm(2.,1002.),(False,'laser_source_missing_or_invalid'))
        laser(gate,2.)
        self.assertEqual(gate.arm(2.,1002.),(True,'armed_once'))

    def test_repeated_health_cannot_refresh_old_scan(self):
        gate=self.gate();laser(gate,2.)
        self.assertTrue(gate.arm(2.,1002.)[0])
        for tick in range(21,27):
            now=tick/10.;hardware(gate,now);vision(gate,now);laser(gate,now,source_time=2.)
        self.assertEqual(gate.laser_health['receipt'],2.)
        gate.propose([.05,0.,0.,0.,0.,0.],2.59,'/nso_visual/move_base')
        self.assertEqual(gate.decision(2.6,1002.6),(0.,0.,'laser_source_stale'))

    def test_invalid_laser_zeroes_before_other_data_expires(self):
        gate=self.gate();laser(gate,2.);self.assertTrue(gate.arm(2.,1002.)[0])
        laser(gate,2.01,valid=False)
        gate.propose([.05,0.,0.,0.,0.,0.],2.01,'/nso_visual/move_base')
        self.assertEqual(gate.decision(2.02,1002.02)[:2],(0.,0.))

    def test_wrong_source_future_and_regression_latch(self):
        for changes in ({'source':'visual'},{'caller':'/other'}, {'frame':'map'},
                        {'source_stamp_ns':round(1002.2*1e9)},
                        {'source_stamp_ns':round(1001.9*1e9)}, {'fault':'scan_malformed'}):
            with self.subTest(changes=changes):
                gate=self.gate();laser(gate,2.);laser(gate,2.01,**changes)
                self.assertIsNotNone(gate.fault)
                self.assertFalse(gate.arm(2.01,1002.01)[0])


if __name__=='__main__':unittest.main()
