"""Same source, pose, and sensors with a separately frozen motion support cache."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
ORIGINAL_DIRECTORY = ANALYSES / "ras_20261126/heterogeneous/scenes"
sys.path.insert(0, str(ORIGINAL_DIRECTORY))
spec = spec_from_file_location("scanned_floor_original_adapter", ORIGINAL_DIRECTORY / "scene.py")
native = module_from_spec(spec)
spec.loader.exec_module(native)

# The pinned original module loaded its exact layout/start registration before
# changing the cache location. Original sensor geometry is still read verbatim.
native.HERE = HERE
SOURCES = native.SOURCES
SPECS = native.SPECS
LAYOUTS = native.LAYOUTS
BUDGET = native.BUDGET
source_scene = native.source_scene
build_scene = native.build_scene
make_world = native.make_world
sha = native.sha
