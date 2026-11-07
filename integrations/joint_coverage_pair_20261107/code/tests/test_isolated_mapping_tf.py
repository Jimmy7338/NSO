"""Pure TF adapter tests; no ROS or devices."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as S
import unittest

spec=importlib.util.spec_from_file_location('isolated_mapping_tf',Path(__file__).parents[1]/'runtime/isolated_mapping_tf.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def transform(parent='nso_pair_map',child='odom',yaw=.7):
    return S(header=S(frame_id=parent,stamp=S(secs=1794062210,nsecs=123456789),seq=7),child_frame_id=child,
             transform=S(translation=S(x=1.2,y=-.3,z=.2),rotation=S(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))))

class Tests(unittest.TestCase):
    def test_inverse_roundtrip_and_stamp(self):
        t=transform();before=copy.deepcopy(t);i=m.inverse_transform(t);j=m.inverse_transform(i)
        self.assertEqual(t,before);self.assertEqual(i.header.stamp,t.header.stamp)
        self.assertEqual((i.header.frame_id,i.child_frame_id),('odom','nso_pair_map'))
        for name in ['x','y','z']:self.assertAlmostEqual(getattr(j.transform.translation,name),getattr(t.transform.translation,name),places=13)
        c,s=math.cos(.7),math.sin(.7)
        self.assertAlmostEqual(i.transform.translation.x,-c*1.2+s*.3)
        self.assertAlmostEqual(i.transform.translation.y,s*1.2+c*.3)
    def test_original_parent_and_inverse_never_forward(self):
        for t in [transform('map','odom'),transform('odom','nso_pair_map'),transform('nso_pair_map','odom')]:
            self.assertEqual(m.global_to_private([t],m.ODOM_CALLER),[])
    def test_odom_source_and_stamp_preserved(self):
        t=transform('odom','base_link');out=m.global_to_private([t],m.ODOM_CALLER)
        self.assertEqual(out,[t]);self.assertIsNot(out[0],t)
        self.assertEqual(m.global_to_private([t],'/untrusted'),[])
        self.assertEqual(m.global_to_private([t],'/zed2i/zed_node'),[])
    def test_private_authority_and_exact_edge(self):
        t=transform();self.assertEqual(len(m.private_to_global([t],m.TRIAL_CALLER)),1)
        self.assertEqual(m.private_to_global([t],'/nso_pair_tf_adapter'),[])
        self.assertEqual(m.private_to_global([transform('odom','base_link')],m.TRIAL_CALLER),[])
    def test_no_loop_with_both_callbacks(self):
        t=m.private_to_global([transform()],m.TRIAL_CALLER)
        self.assertEqual(m.global_to_private(t,'/nso_pair_tf_adapter'),[])
        t=m.global_to_private([transform('odom','base_link')],m.ODOM_CALLER)
        self.assertEqual(m.private_to_global(t,'/nso_pair_tf_adapter'),[])
    def test_invalid_transform(self):
        t=transform();t.transform.translation.x=float('nan')
        self.assertEqual(m.private_to_global([t],m.TRIAL_CALLER),[])
        t=transform();t.transform.rotation.w=2.
        self.assertEqual(m.global_to_private([t],m.ODOM_CALLER),[])
    def test_nonplanar_inverse_roundtrip(self):
        t=transform();q=t.transform.rotation
        q.x,q.y,q.z,q.w=.2,-.3,.1,math.sqrt(.86)
        result=m.inverse_transform(m.inverse_transform(t))
        for name in ['x','y','z']:
            self.assertAlmostEqual(getattr(result.transform.translation,name),getattr(t.transform.translation,name),places=13)
        for name in ['x','y','z','w']:
            self.assertAlmostEqual(getattr(result.transform.rotation,name),getattr(t.transform.rotation,name),places=13)
    def test_actual_launch_is_isolated(self):
        import xml.etree.ElementTree as ET
        root=ET.parse(Path(__file__).parents[1]/'config/paired_mapping.launch').getroot()
        n=root.find('group').find('node');self.assertEqual(n.attrib['name'],'gmapping')
        remaps={x.attrib['from']:x.attrib['to'] for x in n.findall('remap')}
        self.assertEqual(remaps['/tf'],'/nso_pair/tf');self.assertEqual(remaps['map'],'/nso_pair/map')
        self.assertEqual({x.attrib['name']:x.attrib['value'] for x in n.findall('param')}['map_frame'],'nso_pair_map')
        original=ET.parse(Path(__file__).parents[1]/'config/hybrid_sensors_csm.launch').getroot()
        old=next(x for x in original.findall('node') if x.attrib['name']=='slam_gmapping')
        params=lambda node:{x.attrib['name']:x.attrib['value'] for x in node.findall('param')}
        expected=params(old);expected['map_frame']='nso_pair_map'
        self.assertEqual(params(n),expected)

if __name__=='__main__':unittest.main()
