"""Sensor persistence must preserve the exact identities used by frozen replay."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.virtual3d import VirtualConfig, VirtualWorld
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
from nso.decision_replay_v13 import load_packet, save_packet


class PacketPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = VirtualConfig(width_px=32, height_px=24, depth_sigma_m=0., dropout=0.)
        world = VirtualWorld(config, seed=21, layout="rooms")
        cls.config = config
        cls.transform = GridTransform(world.shape, config.resolution_m)
        cls.packet = SensorPacket("scene", "episode", "frame-0", 0,
            world.sense(), world.scan(), tuple(map(int, world.position)), int(world.heading),
            "rendered_sensor", "simulator_exact")

    def test_disk_roundtrip_preserves_identity_and_independent_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.npz"
            save_packet(path, self.packet)
            restored = load_packet(path).validate(self.transform, self.config)
            self.assertEqual(restored.sha256(), self.packet.sha256())
            self.assertFalse(np.shares_memory(restored.frame.depth_m, self.packet.frame.depth_m))
            restored.frame.depth_m.flat[0] += .125
            self.assertNotEqual(restored.sha256(), self.packet.sha256())
            self.assertEqual(load_packet(path).sha256(), self.packet.sha256())

    def test_existing_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.npz"
            save_packet(path, self.packet)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_packet(path, self.packet)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
