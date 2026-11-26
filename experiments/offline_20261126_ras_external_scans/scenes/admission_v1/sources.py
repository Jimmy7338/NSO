"""Pinned 3-D source inventory for an independent scanned-floor validation."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[2]
SOURCE_ASSETS = HERE.parent / "source_assets"
HM = ANALYSES / "semantic_scene_redesign_20261119/assets/hm3d_example"
HM_LICENCE = "https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data"
REPLICA_REVISION = "910c783fb954da8497ea5f811b843a76590ddddc"
REPLICA_RELEASE = f"https://huggingface.co/datasets/ai-habitat/habitat_test_scenes/tree/{REPLICA_REVISION}"


def hm(scene_id, role):
    name = scene_id.split("-", 1)[1]
    return dict(path=HM / "extracted" / scene_id / (name + ".glb"), role=role, source="HM3D v0.2",
                revision="9b74d5aa58e371c9bc16b2ccb64748cddf9bf1c6", location=scene_id,
                license=HM_LICENCE,
                source_url="https://github.com/matterport/habitat-matterport-3dresearch",
                original_real_scan=True)


SOURCES = {
    "replica_apartment1": dict(path=SOURCE_ASSETS / "apartment_1.glb", role="evaluation", source="Replica / Habitat Test Scenes",
                              revision=REPLICA_REVISION, location="apartment_1.glb",
                              license=f"{REPLICA_RELEASE}/README.md", source_url=REPLICA_RELEASE,
                              original_real_scan=True),
    "hm3d_00861": hm("00861-GLAQ4DNUx5U", "evaluation"),
    "hm3d_00337": hm("00337-CFVBbU9Rsyb", "development"),
    "hm3d_00770": hm("00770-NBg5UqG3di3", "evaluation_reserve"),
}


ADMISSION = {
    "reference": "Supplied textured triangle mesh, fixed and never exposed to actors.",
    "motion": "Metric 0.20 m cells, 0.20 m circular footprint and continuous swept capsules.",
    "body_collision_band_m": [0.15, 1.2],
    "scan_floor_support_band_m": [-0.09, 0.09],
    "scan_floor_normal_abs_cosine_minimum": 0.96,
    "floor_choice": "Largest upward horizontal area in a 0.05 m Z bin; weighted mean of that bin's face heights. Other floors remain in the renderer but are not part of the traversability reference.",
    "component_choice": "Largest floor-support/capsule connected component. Require at least 20 sq m and at least two geometry-separated room cores.",
    "room_core": "Connected after clearance threshold 0.48 m, minimum area 2 sq m; merge tiny regions by graph-distance assignment and report descriptive topology rather than official room labels.",
    "starts": "Two room-core starts, graph-separated by at least 3 m. Candidate near each largest/separated core's center, then descending capsule clearance; cardinal heading maximizing forward unclipped geometric visibility. Require valid RGB-D and planar scan with >0 unseen reachable cells at each frozen start.",
    "selection": "Use evaluation roles in registry order. Reserve fills a rejected HM3D source only. Source eligibility is completely established before inspecting any policy output.",
    "minimum_fraction_depth_hits": 0.5,
    "budget": 192,
    "assume_labels": "Both regional modes retain the inherited local S implementation with all-zero visible category/instance channels (geometry-only fallback).",
}
