"""V24.1: real tall work-station backstops, unchanged V24 v3 floor geometry.

The sole change is four existing L-backstop panels raised from 2.4 to 4.8 m.
The v3 world remains frozen. No camera, visibility, range or noise override.
"""
import numpy as np
from env.facility_choice_v24 import (FacilityChoiceWorldV24, PARENTS_V24,
    ASSIGNMENTS_V24, LAYOUTS_V24, prefix_spec_v24)
from env.virtual3d_inspection_v4 import union_surface_from_boxes

VERSION_V24_1 = 'two-facility-choice-world-v24-4-high-backstops'


class FacilityChoiceWorldV24_1(FacilityChoiceWorldV24):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raised = []
        for index, primitive in enumerate(self._solid_primitives):
            x,y,z,sx,sy,sz,owner = primitive
            backstop = owner == 1 and np.allclose([z,sz],[0.,2.4],atol=1e-12,rtol=0) and (
                np.allclose([sx,sy],[5.9,.2],atol=1e-12,rtol=0) or
                np.allclose([sx,sy],[.2,3.8],atol=1e-12,rtol=0))
            if backstop:
                self._solid_primitives[index] = (x,y,z,sx,sy,4.8,owner)
                raised.append(index)
        if len(raised) != 4:
            raise ValueError('Expected exactly four V24 v3 L-backstop panels')
        self.mesh,self.triangle_owners,self.surface_measure_audit = union_surface_from_boxes(self._solid_primitives)
        self.triangle_classes = self.triangle_owners.copy()
        for item in self.objects:
            self.triangle_classes[self.triangle_owners == item['owner']] = item['category']
        self.sensor_metadata.update(world_version=VERSION_V24_1,
            raised_backstop_panels=raised,backstop_height_m=4.8,
            v3_floor_geometry_and_paid_route_preserved=True)
