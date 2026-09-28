"""Frozen V36 noise holdout; all geometry and sensor behaviour stays in V34.

The inherited constructor produces no observation.  The single preregistered
base seed is installed before this instance can produce its first packet.
Scene and sensor-contract identifiers stay V34; episode IDs identify the V36
batch externally.  There is no caller-selectable seed or changed noise model.
"""
from env.information_pixel_v34 import InformationPixelWorldV34


CONFIRMATION_NOISE_SEED_V36 = 350918


class InformationPixelWorldV36(InformationPixelWorldV34):
    def __init__(self, parent_id, hypothesis, episode_id, noise_model='clean'):
        super().__init__(parent_id, hypothesis, episode_id, noise_model=noise_model)
        self.noise_seed = CONFIRMATION_NOISE_SEED_V36
