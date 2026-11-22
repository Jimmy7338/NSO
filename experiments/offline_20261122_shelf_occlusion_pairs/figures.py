#!/usr/bin/env python3
"""Render measured paired outcomes and one prospectively selected mechanism case."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from study import ROOT, read, sha, write


COLORS = {"open": "#687285", "screen": "#D07927", "S": "#075D9D", "D": "#9A381C"}
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                            "pdf.fonttype": 42, "svg.fonttype": "none"})


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(string):
    return float(string) if string not in (None, "") else None


def save(fig, directory, stem):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(directory/f"{stem}.{extension}", dpi=240)
    plt.close(fig)


def contrasts(data, papers, sealed):
    rows = table(data/"pairs.csv")
    names = [f"{index+1:02d}" for index, _ in enumerate(sealed["parents"])]
    assert len(rows) == len(names)
    keys = ("open_S_minus_D_96_m2", "screen_S_minus_D_96_m2")
    fig, ax = plt.subplots(figsize=(6.6, 3.65), layout="constrained")
    values = [number(row[key]) for row in rows for key in keys]
    observed = [v for v in values if v is not None]
    limit = max(0.1, *(abs(v) for v in observed)) if observed else .1
    ax.axvline(0, color="#344055", lw=1, zorder=0)
    for i, row in enumerate(rows):
        open_value, screened = [number(row[key]) for key in keys]
        endpoints = [(open_value, i-.10, "open", "o"), (screened, i+.10, "screen", "D")]
        if open_value is not None and screened is not None:
            ax.plot([open_value, screened], [i-.10, i+.10], lw=1.2, c="#A9AFB8", zorder=1)
        for value, y, status, marker in endpoints:
            if value is not None:
                ax.scatter(value, y, marker=marker, s=43, color=COLORS[status], edgecolors="white",
                           linewidths=.6, zorder=2)
        missing = [word for value, _, word, _ in endpoints if value is None]
        if missing:
            ax.annotate("/".join(missing) + " unavailable", (1,i), xycoords=("axes fraction","data"),
                        xytext=(-4,0), textcoords="offset points", ha="right", va="center",
                        fontsize=7, color="#606B74")
    ax.set(yticks=range(len(names)), yticklabels=[f"Unit {name}" for name in names],
           xlim=(-limit*1.37, limit*1.50), xlabel=r"Net shelf area difference at action 96, S − D (m²)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DCE2E9", linewidth=.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    legend = [Line2D([], [], ls="", marker=m, color=COLORS[c], label=t) for c,m,t in
              (("open","o","Front unobstructed"),("screen","D","Front partly screened"))]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0,1), ncol=2, frameon=False, borderaxespad=0)
    save(fig, papers, "paired_shelf_area")


def execution(placement, arm):
    receipt = read(ROOT/"execution/formal"/placement/"placement_receipt.json")
    result = receipt["arms"][arm]
    if result["status"] != "verified_native_execution":
        return None
    run = Path(result["run_directory"])
    assert sha(run/"manifest.json") == result["manifest_sha256"]
    assert sha(run/"fork_bank.json") == result["fork_bank_sha256"]
    return {"path":run, "manifest":read(run/"manifest.json"), "bank":read(run/"fork_bank.json"),
            "steps":read(run/"steps.json")}


def complete_pair(items):
    return all(x and x["manifest"]["returned"] and x["manifest"]["checkpoints"]["paid_96"]["status"]=="captured"
               for x in items)


def diverged(items):
    a,b = items
    return a["bank"]["status"] == b["bank"]["status"] == "triggered" and any(
        any(row[key]!=other[key] for key in ("action","position","heading"))
        for row,other in zip(a["steps"],b["steps"]))


def select_case(sealed):
    cases = {name:[execution(name,x) for x in ("S","D")] for name in sealed["placements"]}
    ordered = sorted(cases, key=lambda name: not name.endswith("_screen"))
    for condition in (lambda pair: complete_pair(pair) and diverged(pair), complete_pair,
                      lambda pair: any(x and x["bank"]["status"]=="triggered" for x in pair)):
        for name in ordered:
            if condition(cases[name]):
                return name,cases[name]
    return ordered[0],cases[ordered[0]]


def mechanism(data, papers, sealed):
    case, runs = select_case(sealed)
    declared_case=read(data/"summary.json")["representative_mechanism_placement"]
    assert declared_case is None or case==declared_case, "The figure and measured case selection disagree."
    common = next((r for r in runs if r and r["bank"]["status"]=="triggered"), None)
    witness = common or next((r for r in runs if r), None)
    fig, axes = plt.subplots(2,2,figsize=(7.4,5.7),layout="constrained")
    image_ax, map_ax, early_ax, late_ax = axes.ravel()
    fork = common["bank"]["paid_before"] if common else None
    for ax, title in zip(axes.ravel(), ("A  Common camera observation","B  Executed routes",
                                       "C  Exclusive shelf surface at action 96","D  Still exclusive at return"),strict=True):
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=10)
    camera_transform = None
    if witness:
        with np.load(witness["path"]/"common_prefix.npz", allow_pickle=False) as prefix:
            k = -1 if fork is not None else 0
            image_ax.imshow(prefix["color_rgb"][k])
            camera_transform = prefix["world_from_camera"][k]
        image_ax.text(.015,.04,f"Action {fork}" if fork is not None else "Cold start · no qualifying fork",
                      transform=image_ax.transAxes, c="white", fontsize=8,
                      bbox=dict(facecolor="#243245",alpha=.85,edgecolor="none",pad=2))
    else:
        image_ax.text(.5,.5,"Camera observation unavailable",ha="center",transform=image_ax.transAxes)
    image_ax.set_axis_off()
    with np.load(ROOT/"scenes/evaluator_references"/case/"evaluator_truth.npz",allow_pickle=False) as truth:
        names = ("occupancy","true_occupied","occupied")
        raster = next((truth[key] for key in names if key in truth),None)
    init = read(ROOT/"scenes/prepared"/case/"initial_state.json")
    res = init["grid_resolution_m"]
    height = init["shape"][0]
    if raster is not None:
        map_ax.imshow(raster, cmap="Greys", alpha=.22, origin="upper",
                      extent=(0,raster.shape[1]*res,0,raster.shape[0]*res))
    for arm,run in zip(("S","D"),runs,strict=True):
        if not run:
            continue
        rc = np.asarray([init["anchor"][:2],*(r["position"] for r in run["steps"])])
        x,y = (rc[:,1]+.5)*res,(height-rc[:,0]-.5)*res
        first = fork or 0
        if arm=="S" and first>0:
            map_ax.plot(x[:first+1],y[:first+1],color="#667886",lw=2,label="Common prefix")
        event=run["manifest"]["route_event"]
        after=event.get("actual_paid")
        after=len(x)-1 if after is None else min(max(after,first),len(x)-1)
        map_ax.plot(x[after:],y[after:],color=COLORS[arm],alpha=.30,lw=.9)
        map_ax.plot(x[first:after+1],y[first:after+1],color=COLORS[arm],
                    ls="--" if arm=="D" else "-",lw=1.5,label=arm)
        if run["bank"]["status"]=="triggered":
            nominal=np.asarray(run["bank"]["arms"][arm]["outbound_states"])
            tx,ty=(nominal[:,1]+.5)*res,(height-nominal[:,0]-.5)*res
            map_ax.plot(tx,ty,color=COLORS[arm],lw=1,ls=":",alpha=.65)
            map_ax.scatter(tx[-1],ty[-1],marker="s",facecolor="white",edgecolors=COLORS[arm],s=23,zorder=3)
            if event["status"] != "arrived":
                map_ax.scatter(x[after],y[after],marker="x",color=COLORS[arm],s=28,zorder=4)
        map_ax.scatter(x[first],y[first], marker="*",s=55,color="#303947",zorder=3,
                       label="Common fork" if arm=="S" else None)
    map_ax.set(xlabel="x (m)",ylabel="y (m)",aspect="equal")
    if any(runs):
        map_ax.legend(loc="upper right",frameon=False,fontsize=7)
    raw_masks = data/f"{case}_masks.npz"
    measured = {arm:{} for arm in ("S","D")}
    if raw_masks.exists() and complete_pair(runs):
        with np.load(raw_masks,allow_pickle=False) as masks:
            xyz = masks["reference_shelf_xyz_m"]
            weights = masks["reference_shelf_weights_m2"]
            dynamic = all(key in masks for key in ("fork_shelf","S_paid_96_shelf","D_paid_96_shelf",
                                                     "S_terminal_shelf","D_terminal_shelf"))
            if dynamic:
                early, returned = {},{}
                for arm,other in (("S","D"),("D","S")):
                    early[arm] = masks[f"{arm}_paid_96_shelf"] & ~masks[f"{other}_paid_96_shelf"] & ~masks["fork_shelf"]
                    returned[arm] = early[arm] & masks[f"{arm}_terminal_shelf"] & ~masks[f"{other}_terminal_shelf"]
                    measured[arm] = {"exclusive_96_m2":float(weights[early[arm]].sum()),
                                     "retained_exclusive_m2":float(weights[returned[arm]].sum())}
        projected = (xyz-xyz.mean(axis=0)) @ camera_transform[:3,:3] if camera_transform is not None else xyz[:,[0,2,1]]
        x,y = projected[:,0]+.38*projected[:,2], -projected[:,1]+.17*projected[:,2]
        for target,layer in ((early_ax,early if dynamic else {}),(late_ax,returned if dynamic else {})):
            target.scatter(x,y,c="#C8D0D8",s=2,alpha=.32,linewidths=0,rasterized=True)
            for arm in ("S","D"):
                if arm in layer:
                    mask = layer[arm]
                    target.scatter(x[mask],y[mask],c=COLORS[arm],s=5,alpha=.85,linewidths=0,rasterized=True)
            target.set(aspect="equal",xticks=[],yticks=[])
            if dynamic:
                for arm,xpos in (("S",.04),("D",.55)):
                    area=float(weights[layer[arm]].sum())
                    target.text(xpos,-.02,f"{arm}  {area:.3f} m²",transform=target.transAxes,
                                c=COLORS[arm],fontsize=8,va="top")
            else:
                target.text(.5,.5,"Common fork mask unavailable",ha="center",transform=target.transAxes)
    else:
        for target in (early_ax,late_ax):
            target.text(.5,.5,"Matched surface comparison unavailable",ha="center",transform=target.transAxes)
            target.set(xticks=[],yticks=[])
    for ax in (early_ax,late_ax):
        ax.spines[:].set_visible(False)
    save(fig,papers,"mechanism_case")
    result = {"placement":case,"common_fork_paid":fork,"complete_return_pair":complete_pair(runs),
              "actual_route_divergence":bool(complete_pair(runs) and diverged(runs)),
              "source_masks_sha256":sha(raw_masks) if raw_masks.is_file() else None,"exclusive":measured}
    target = papers/"figure_case.json"
    if target.exists():
        assert read(target)==result
    else:
        write(target,result)


def main():
    data,papers = ROOT/"results/formal",ROOT/"paper"
    papers.mkdir(exist_ok=True)
    sealed = read(ROOT/"FORMAL_FREEZE.json")
    contrasts(data,papers,sealed)
    mechanism(data,papers,sealed)


if __name__=="__main__":
    main()
