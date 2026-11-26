"""Register the external-scene start pairs before preparing any actor."""

import json
import time

from sources import ADMISSION, HERE, SOURCES


def main():
    preflights = {floor: json.loads((HERE / "source_preflights" / f"{floor}.json").read_text())
                  for floor in ("replica_apartment1", "hm3d_00861", "hm3d_00337")}
    chosen = [f for f in ("replica_apartment1", "hm3d_00861") if preflights[f]["eligible"]]
    if "hm3d_00861" not in chosen:
        reserve = HERE / "source_preflights/hm3d_00770.json"
        preflights["hm3d_00770"] = json.loads(reserve.read_text())
        if preflights["hm3d_00770"]["eligible"]:
            chosen.append("hm3d_00770")
    if not preflights["hm3d_00337"]["eligible"]:
        raise ValueError("Unrelated development scene failed geometry admission")
    all_layouts = {}
    for floor in [*chosen, "hm3d_00337"]:
        report = preflights[floor]
        development = SOURCES[floor]["role"] == "development"
        for index, start in enumerate(report["starts"][:1] if development else report["starts"], 1):
            name = f"b26_{floor}_s{index}"
            all_layouts[name] = start | {"source_floor": floor, "start_index": index, "shape": report["shape"],
                                          "world_from_source": report["world_from_source"],
                                          "seed": 2026112600 + 10*list(SOURCES).index(floor) + index,
                                          "scene_role": "unrelated_development" if development else "independent_test"}
    payload = dict(frozen_unix_s=time.time(), actor_select_calls=0, admitted_test_floors=chosen,
                   admission=ADMISSION, layouts=all_layouts,
                   verdicts={floor: {"eligible": row["eligible"], "reason": row.get("reason")}
                             for floor, row in preflights.items()})
    with (HERE / "STARTS.json").open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Registered actor-free starts: {', '.join(all_layouts)}", flush=True)


if __name__ == "__main__":
    main()
