"""Record geometry-only scene admission and prepare pre-registered cold-start pairs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt

from preflight import TexturedRayRenderer, floor_height, flood, geometry, grid_coords, inspect, largest_component, room_cores, sha, source_scene
from sources import ADMISSION, ADMISSION_ORDER, CANDIDATES, HERE, ORIGINAL, REGISTRY, SOURCES


REPORTS = HERE / "source_preflights"
CACHES = HERE / "precomputed_assets"
ADMISSION_CACHES = HERE / "admission_cache"


def read(path):
    return json.loads(path.read_text())


def save(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prior_layouts(floor):
    return [r for r in read(ORIGINAL / "scenes/STARTS.json")["layouts"].values() if r["source_floor"] == floor]


def fallback_starts(source, transform, shape, cores, clearance, edges):
    def ranked(group):
        center=np.mean(group,axis=0)
        return sorted(group,key=lambda p:(float(np.linalg.norm(np.asarray(p)-center)),-clearance[p],p))

    renderer=TexturedRayRenderer(source)
    inverse=np.linalg.inv(transform)
    config=geometry.ReplicaCADConfig(max_steps=192)
    focal=config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2)))
    k=np.array([[focal,0,(config.width_px-1)/2],[0,focal,(config.height_px-1)/2],[0,0,1.]])
    cardinal=np.array([[0,1,0],[1,0,0],[0,-1,0],[-1,0,0]],dtype=np.float32)
    rejected=[]

    def first_valid(group):
        for cell in group:
            xyz=np.array([*grid_coords(shape,cell),config.camera_height_m,1.])
            origin=(inverse @ xyz)[:3]
            rays=np.column_stack((np.tile(origin,(4,1)),cardinal)).astype(np.float32)
            raw=renderer.scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)["t_hit"].numpy()
            heading=int(np.argmax(np.minimum(raw,config.max_depth_m)))
            pose=inverse @ geometry.camera_pose(cell,heading,config,shape[0])
            frame=renderer.render(k,pose,config.width_px,config.height_px,max_depth_m=config.max_depth_m)
            hit_fraction=float((frame["depth_m"]>0).mean())
            if hit_fraction>=ADMISSION["minimum_fraction_depth_hits"]:
                return {"start":cell,"heading":heading,"xy_world_m":grid_coords(shape,cell),"xyz_source_m":origin.tolist(),
                        "depth_hit_fraction":hit_fraction,"initial_cardinal_range_m":[float(x) if np.isfinite(x) else None for x in raw]}
            rejected.append({"cell":cell,"heading":heading,"depth_hit_fraction":hit_fraction})
        return None

    ranks=[ranked(group) for group in cores]
    first=first_valid(ranks[0])
    if first is None:
        raise ValueError("No cold RGB-D acquisition satisfies the original depth coverage in the largest geometry core")
    distance=flood(edges,tuple(first["start"]))
    remote=sorted(ranks[1:],key=lambda group:(-distance[group[0]],group[0]))
    for group in remote:
        second=first_valid([cell for cell in group if distance[cell]*.20>=3.])
        if second is not None:
            return [first,second],float(distance[second["start"]]*.20),rejected
    raise ValueError("No other eligible core has a 3 m separate, cold-depth-valid observation")


def fallback(floor):
    source_cache=ORIGINAL / f"scenes/precomputed_assets/{floor}.npz"
    source_admission=ORIGINAL / f"scenes/source_preflights/{floor}.json"
    origin=read(source_admission)
    assert origin["eligible"] and sha(source_cache)==origin["capsule_cache_sha256"]
    with np.load(source_cache,allow_pickle=False) as archive:
        valid, edges = archive["robot_valid"], archive["edge_valid"]
    main,_=largest_component(valid,edges)
    clearance=distance_transform_edt(main)*.20
    old_starts=[tuple(r["start"]) for r in prior_layouts(floor)]
    old_distances=[flood(edges,p) for p in old_starts]
    min_steps=int(np.ceil(REGISTRY["fallback_minimum_swept_distance_from_every_old_start_m"]/.20-1e-9))
    allowed=np.logical_and.reduce([d>=min_steps for d in old_distances])
    cores=[[cell for cell in core if allowed[cell]] for core in room_cores(main,edges,clearance)]
    cores=[cells for cells in cores if cells]
    proposed,separation,rejected=fallback_starts(source_scene(floor),np.asarray(origin["world_from_source"]),
                                                tuple(origin["shape"]),cores,clearance,edges)
    minimum_distances=[]
    for start in proposed:
        minimum_distances.append(min(int(d[tuple(start["start"])])*.20 for d in old_distances))
    receipt={
        **origin, "eligible": True, "starts": proposed, "graph_start_separation_m":separation,
        "prior_source_role": SOURCES[floor]["prior_source_role"], "fallback":True,
        "inherited_source_preflight_sha256":sha(source_admission),
        "old_starts":old_starts,"new_start_min_distance_from_all_old_starts_m":minimum_distances,
        "rejected_geometry_ordered_initial_depth_candidates":rejected,
        "guard": "Reused the same strict source-geometry thresholds for admission; new distinct starts only.",
        "actor_select_calls":0,
    }
    (ADMISSION_CACHES/f"{floor}.npz").hardlink_to(source_cache)
    return receipt


def preflight(floor):
    if (REPORTS/f"{floor}.json").exists():
        return
    if prior_layouts(floor):
        report=fallback(floor)
    elif not SOURCES[floor]["path"].is_file():
        incomplete=Path(str(SOURCES[floor]["path"])+".part")
        report={"source_floor":floor,"eligible":False,"reason":"public source download incomplete at registration; geometry unassessed",
                "source_url":SOURCES[floor]["source_url"],"geometry_queried":False,"actor_select_calls":0,
                "downloaded_partial_bytes_at_preregistration":incomplete.stat().st_size if incomplete.is_file() else 0,
                "expected_native_asset_bytes":{"bundlefusion_apt0":326274432,"bundlefusion_apt1":275289752}.get(floor)}
    else:
        try:
            report=inspect(floor)
        except RuntimeError as error:
            if str(error) != "Source geometry has no obstacle or floor triangles":
                raise
            source=source_scene(floor)
            z, bins=floor_height(source)
            report={"source_floor":floor,"eligible":False,"reason":"dominant_upward_area_bin_has_no_inherited_body_obstacle_and_floor_support",
                    "source_file_sha256":sha(SOURCES[floor]["path"]),"dominant_source_floor_z_m":z,
                    "upward_elevation_bins_m2":bins,"raw_scan_bounds_m":source.bounds.tolist(),"actor_select_calls":0}
        if (CACHES/f"{floor}.npz").exists():
            (CACHES/f"{floor}.npz").rename(ADMISSION_CACHES/f"{floor}.npz")
    save(REPORTS/f"{floor}.json",report)
    print(json.dumps({k:report.get(k) for k in ("source_floor","eligible","reason","navigable_area_m2","starts")}),flush=True)


def freeze():
    reports,chosen={},[]
    for floor in ADMISSION_ORDER:
        path=REPORTS/f"{floor}.json"
        if path.exists():
            reports[floor]=read(path)
            if reports[floor]["eligible"] and len(chosen)<2:
                chosen.append(floor)
        elif len(chosen)==2:
            reports[floor]={"eligible":None,"reason":"lower-priority source not required after two earlier candidates passed"}
        else:
            raise ValueError(f"A higher-priority scene is pending official geometry-only admission: {floor}")
    if len(chosen)!=2:
        raise ValueError("Two eligible scanned floors are required; unavailable assets and failures are retained")
    layouts={}
    for floor in chosen:
        report=reports[floor]
        for index,start in enumerate(report["starts"],1):
            name=f"b26v5_{floor}_s{index}"
            layouts[name]=start | {"source_floor":floor,"start_index":index,"shape":report["shape"],
                                   "world_from_source":report["world_from_source"],
                                   "seed":REGISTRY["layout_seed_base"]+10*list(SOURCES).index(floor)+index,
                                   "scene_role":"new_starts_on_previously_used_scan" if report.get("fallback") else "independent_test",
                                   "source_preflight_sha256":sha(REPORTS/f"{floor}.json")}
    assert len(layouts)==4
    metadata={floor:{"eligible":r["eligible"],"fallback":bool(r.get("fallback")),"reason":r.get("reason"),
                     "preflight_sha256":sha(REPORTS/f"{floor}.json") if (REPORTS/f"{floor}.json").exists() else None} for floor,r in reports.items()}
    save(HERE/"STARTS.json",{"schema":"nso.reacquisition_geometry_only_preregistered_starts.v1","frozen_unix_s":time.time(),
                            "actor_select_calls":0,"candidate_registry_sha256":sha(CANDIDATES),
                            "admitted_test_floors":chosen,"admission":ADMISSION,"layouts":layouts,"verdicts":metadata,
                            "holdout_floor_count":sum(not bool(reports[f].get("fallback")) for f in chosen),
                            "frozen_budget":192})
    print("Geometry-preregistered layouts:",", ".join(layouts),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["preflight","freeze"])
    p.add_argument("floor",choices=list(SOURCES),nargs="?")
    args=p.parse_args()
    for d in (REPORTS,CACHES,ADMISSION_CACHES):
        d.mkdir(exist_ok=True)
    if args.action=="preflight":
        preflight(args.floor)
    else:
        freeze()


if __name__ == "__main__":
    main()
