#!/usr/bin/env python3
"""Write precise English manuscript passages from the measured paired matrix."""

import csv

from study import ROOT, read, sha


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def f(value, digits=3):
    return f"{value:+.{digits}f}"


def describe(stat):
    if not stat["n"]:
        return f"0/{stat['scheduled']} measured units"
    return (f"{stat['n']}/{stat['scheduled']} measured units (mean {f(stat['mean'])} m², "
            f"median {f(stat['median'])} m², {stat['win']}/{stat['tie']}/{stat['loss']} wins/ties/losses)")


def numeric(row, name):
    value = row.get(name)
    return float(value) if value not in ("", None) else None


def mean_with_denominator(stat):
    if not stat["n"]:
        return f"unavailable (0/{stat['scheduled']})"
    return f"{f(stat['mean'])} m² ({stat['n']}/{stat['scheduled']})"


def retained_text(data, condition, side, policy):
    cohorts = [r for r in data if r["condition"]==condition and r["pair"]=="S_minus_D"]
    eligible = [r for r in cohorts if r["status"]=="measured"]
    initial = [numeric(r,f"{side}_exclusive_96_m2") for r in eligible]
    final = [numeric(r,f"{side}_still_exclusive_return_m2") for r in eligible]
    informative = [(a,b) for a,b in zip(initial,final,strict=True) if a and b is not None]
    label = "screened" if condition=="screen" else "unobstructed"
    if not informative:
        return (f"In the {label} condition, {len(eligible)}/{len(cohorts)} units had matched action-96 and return "
                f"masks. None of those units showed a nonzero action-96 exclusive cohort for {policy}.")
    total_initial = sum(a for a,_ in informative)
    total_retained = sum(b for _,b in informative)
    total_shared = sum(numeric(r,f"{side}_now_shared_return_m2") or 0 for r in eligible)
    return (f"In the {label} condition, {policy} had an exclusive cohort at action 96 in "
            f"{len(informative)}/{len(eligible)} units with verified dual return "
            f"({len(eligible)}/{len(cohorts)} scheduled). Summed across those cohorts, "
            f"{total_retained:.3f} of {total_initial:.3f} m² remained exclusive at return "
            f"({100*total_retained/total_initial:.1f}%), and {total_shared:.3f} m² became shared.")


def primary_table(rows, tolerance):
    keys=("open_S_minus_D_96_m2","screen_S_minus_D_96_m2","occlusion_interaction_96_m2",
          "screen_S_minus_G_96_m2","screen_D_plus_minus_D_96_m2")
    def format_area(value):
        if value is None:
            return "—"
        return "0.000" if abs(value) <= tolerance else f(value)
    result=["| Unit | Open S−D | Screened S−D | Occlusion interaction | Screened S−G | Screened D+−D |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for index,row in enumerate(rows,1):
        values=[format_area(numeric(row,key)) for key in keys]
        result.append(f"| {index:02d} | " + " | ".join(values) + " |")
    return result


def paired_conclusion(screen, interaction):
    if not screen["n"] or not interaction["n"]:
        return "Incomplete matched outcomes limit a conclusion about the response to occlusion."
    tolerance=screen["tie_tolerance_m2"]
    if screen["mean"] > tolerance:
        screened = "The structure-guided rule recovered more net shelf area on average in the screened units."
    elif screen["mean"] < -tolerance:
        screened = "Direct inspection recovered more net shelf area on average in the screened units."
    else:
        screened = "The mean screened contrast lay within the numerical repeatability tolerance."
    if interaction["mean"] > tolerance:
        shift = ("Screening shifted the relative result toward S in the complete pairs, "
                 "narrowing its disadvantage." if screen["n"] == interaction["n"] and screen["mean"] < -tolerance
                 else "Screening shifted the relative result toward S in the complete pairs.")
    elif interaction["mean"] < -tolerance:
        shift = "Screening shifted the relative result toward D in the complete pairs."
    else:
        shift = "The mean interaction lay within the numerical repeatability tolerance."
    return f"{screened} {shift}"


def main():
    result=ROOT/"results/formal"
    paper=ROOT/"paper"
    summary=read(result/"summary.json")
    for name,digest in summary["table_sha256"].items():
        assert sha(result/name)==digest
    sealed=read(ROOT/"FORMAL_FREEZE.json")
    assert summary["phase_freeze_sha256"]==sha(ROOT/"FORMAL_FREEZE.json")
    assert summary["parents"]==sealed["parents"] and summary["registered_policy_assignments"]==48
    statistics=summary["statistics"]
    open_stat=statistics["open_S_minus_D_96_m2"]
    screen=statistics["screen_S_minus_D_96_m2"]
    interaction=statistics["S_minus_D_interaction_96_m2"]
    evidence=[read(ROOT/"scenes/pair_audits"/f"{p}.json") for p in sealed["parents"]]
    optical=[100*(e["front_visible_open_fraction"]-e["front_visible_screen_fraction"]) for e in evidence]
    side=[e["hidden_area_seen_by_side_pose_m2"] for e in evidence]
    comparisons=table(result/"comparisons.csv")
    comparisons96=[r for r in comparisons if r["clock"]=="96"]
    episodes=table(result/"episodes.csv")
    retained=table(result/"retention.csv")
    failures=table(result/"failures.csv")
    choices={p:next(r for r in episodes if r["placement"]==p and r["arm"]=="G")
             for p in sealed["placements"]}
    forks=sum(r.get("triggered")=="True" for r in choices.values())
    cold_forks=sum(r.get("triggered")=="True" and r.get("fork_paid")=="0" for r in choices.values())
    tolerance=summary["tie_tolerance_m2"]
    measured=summary["measured_native_assignments"]
    unique=summary["unique_completed_executions"]
    both96=sum(r.get("status_96")=="measured" for r in episodes)
    unique_returned=len({r["manifest_sha256"] for r in episodes if r["status"]=="verified_execution" and r["returned"]=="True"})
    execution_rows={r["manifest_sha256"]:r for r in episodes if r["status"]=="verified_execution"}
    distinct_dispatched=[r for r in execution_rows.values() if r["triggered"]=="True"]
    distinct_arrived=sum(r["route_status"]=="arrived" for r in distinct_dispatched)
    attributed=0
    wins_with_neutral_match=0
    screened_fork_agreement={}
    for name in ("S_minus_G","D_plus_minus_D"):
        trials=[r for r in comparisons96 if r["condition"]=="screen" and r["pair"]==name
                and r["same_selected_route"] in ("True","False")]
        screened_fork_agreement[name]=(sum(r["same_selected_route"]=="True" for r in trials),len(trials))
    for r in comparisons96:
        delta=numeric(r,"net_m2_delta")
        if r["condition"]!="screen" or r["pair"]!="S_minus_D" or delta is None or delta<=tolerance:
            continue
        sg=next(x for x in comparisons96 if x["placement"]==r["placement"] and x["pair"]=="S_minus_G")
        sdg=numeric(sg,"net_m2_delta")
        attributed+=bool(sdg is not None and sdg>tolerance and sg["first_actual_difference_paid"] not in (None,""))
        wins_with_neutral_match+=sg["same_execution"]=="True"
    pair_order = statistics["screen_S_minus_D_96_shelf_and_J_same_order"]
    screen_c = statistics["screen_S_minus_D_96_C_free"]
    screen_f = statistics["screen_S_minus_D_96_F1_scene_05cm"]
    screen_j = statistics["screen_S_minus_D_96_J"]
    if screen_j["n"]:
        map_text=(f"In the screened placements at action 96, mean S-minus-D differences were "
                  f"{f(100*screen_c['mean'],2)} percentage points in correct free-area coverage, "
                  f"{f(100*screen_f['mean'],2)} points in scene F1, and {f(100*screen_j['mean'],2)} points in J "
                  f"across {screen_j['n']}/6 units.")
    else:
        map_text="The screened placements had no comparable action-96 joint map scores."
    terminal_j=statistics["screen_S_minus_D_terminal_J"]
    if terminal_j["n"]:
        terminal_order=statistics["screen_S_minus_D_terminal_shelf_and_J_same_order"]
        map_text += (f" At equal-action terminal times the mean J contrast was "
                     f"{f(100*terminal_j['mean'],2)} points across {terminal_j['n']}/6 screened units. "
                     f"Shelf area and J agreed in {terminal_order['agreement']}/"
                     f"{terminal_order['n_quantifiable_shelf_difference']} non-tied screened terminal comparisons.")
    secondary=[]
    for condition, label in (("open","unobstructed"),("screen","screened")):
        keys = (f"{condition}_S_minus_D_48_m2",f"{condition}_S_minus_D_144_m2",
                f"{condition}_S_minus_D_equal_paid_terminal_m2")
        first, later, terminal = [mean_with_denominator(statistics[k]) for k in keys]
        secondary.append(f"In the {label} condition, mean S-minus-D area was {first} at action 48, "
                         f"{later} at action 144, and {terminal} at equal-action trial endpoints.")
    motion=[numeric(r,"path_length_m") for r in execution_rows.values() if numeric(r,"path_length_m") is not None]
    scoring=[numeric(r,"fork_instrumented_policy_ranking_wall_s") for r in episodes
             if r["triggered"]=="True" and numeric(r,"fork_instrumented_policy_ranking_wall_s") is not None]
    if motion:
        secondary.append(f"Path length ranged from {min(motion):.2f} to {max(motion):.2f} m "
                         f"across {len(motion)}/{unique} distinct trajectories.")
    if scoring:
        secondary.append(f"Recorded fork-ranking time ranged from {min(scoring):.3f} to {max(scoring):.3f} s "
                         f"over {len(scoring)}/{48} policy assignments during concurrent simulation.")
    missing96=48-both96
    sg_same, sg_forks=screened_fork_agreement["S_minus_G"]
    dd_same, dd_forks=screened_fork_agreement["D_plus_minus_D"]
    scope="The results describe this matched simulation design with ideal visible labels."
    if cold_forks==forks and forks:
        scope += " The measured intervention is an early route choice at the cold start."
    parts = [
        "# Results", "",
        f"Frontal screening reduced initial visibility of the independently sampled shelf surface by "
        f"{min(optical):.1f}–{max(optical):.1f} percentage points across the six paired units. "
        f"A physically reachable lateral viewpoint exposed {min(side):.3f}–{max(side):.3f} m² of the "
        f"surface hidden from the screened front. The paired sweeps and reachable cells agreed exactly. "
        f"The first eligible selection occurred in {forks}/12 placements, with {cold_forks}/{forks} forks at the "
        f"cold start. The budget of 192 charged actions "
        f"included turns and return. Of the 48 scheduled policy assignments, "
        f"{measured} had verified native outcomes supported by {unique} distinct executed trajectories. "
        f"{unique_returned}/{unique} distinct trajectories returned. The nominated oriented pose was reached in "
        f"{distinct_arrived}/{len(distinct_dispatched)} distinct trajectories with a dispatched intervention. "
        f"Action 96 was observed in {both96}/48 assignments, with {missing96} scheduled values unavailable.", "",
        f"The action-96 S-minus-D shelf contrast covered {describe(open_stat)} with the front unobstructed "
        f"and {describe(screen)} with the front screened. The within-parent occlusion interaction used "
        f"{describe(interaction)}. {paired_conclusion(screen,interaction)}",
        "",
        "Action-96 shelf contrasts in m² for the six independent room–shelf units. "
        "An em dash denotes an unmeasured contrast.", "",
        *primary_table(table(result/"pairs.csv"),tolerance), "",
        f"In the screened condition, the category-prior comparison (S minus G) covered "
        f"{describe(statistics['screen_S_minus_G_96_m2'])}. The corresponding supported-hidden increment "
        f"(D+ minus D) covered {describe(statistics['screen_D_plus_minus_D_96_m2'])}. The nominated route matched "
        f"for S and G in {sg_same}/{sg_forks} comparable screened forks and for D+ and D in {dd_same}/{dd_forks}. "
        f"Among measured screened S-over-D wins, "
        f"{attributed} had an executed S-versus-G difference and a positive shelf contrast, and "
        f"{wins_with_neutral_match} shared the verified neutral trajectory. The latter wins provide evidence for the "
        f"route used under those conditions and leave its advantage unattributed to category conditioning.", "",
        retained_text(retained,"screen","left","S")+" "+retained_text(retained,"screen","right","D"), "",
        " ".join(secondary), "",
        map_text + f" Shelf area and J favored the same rule in {pair_order['agreement']}/"
        f"{pair_order['n_quantifiable_shelf_difference']} screened units with a non-tied action-96 shelf contrast.", "",
        "# Discussion", "",
        "The paired design isolates the introduction of frontal occlusion in these simulated rooms. It "
        "measures recoverable surface relative to the actually fused cold-start map of each placement. "
        "Different initial visibility changes the surface available to recover, while the paired shelf "
        "samples and footprint feasibility remain fixed. The comparison tests route selection with "
        "visible class labels, exact pose and a shared neutral continuation. The two ablations localize "
        "potential effects of class conditioning and of the supported increment on measured inspection.",
        "",
        "The retention outcome concerns the early surface samples that were both absent at the fork and "
        "exclusive to one trajectory at action 96. Continued neutral exploration may let the other "
        "trajectory observe the same surface, while fusion may cause formerly matched samples to move "
        "outside the tolerance. Either event can reduce measured exclusivity at return. Whole-scene J "
        "includes free-area coverage and surfaces outside the shelf, allowing it to respond differently "
        "to a change in shelf inspection.",
        "",
        f"The independent sample comprises six authored room–shelf units with shared shelf assets across some units. "
        f"The analysis retains {len(failures)} assignment rows with a guard, early-termination or technical disposition "
        f"under the registered policy. Measured denominators accompany each contrast, and unexecuted action clocks "
        f"remain missing. The numerical tie tolerance was {tolerance:.0e} m², fixed from repeated development "
        f"measurements. {scope}", "",
    ]
    (paper/"manuscript_results_discussion.md").write_text("\n".join(parts))
    mechanism=read(paper/"figure_case.json")
    caps=[
        "# Figure captions","",
        "**Paired action-96 net shelf area.** Each row is one independent room–shelf unit. "
        "The two symbols are the structure-guided minus direct-inspection area difference with the same "
        "front unobstructed or partly screened. Net area is the change in shelf reference area represented "
        "by the measured mesh relative to the common cold map within each placement. The vertical line "
        "marks equal area. Missing symbols indicate that the corresponding comparison could not be measured.","",
        f"**Mechanism example, {mechanism['placement']}.** The registered case selection scans screened placements first "
        "and favors the earliest verified pair with an executed divergence and return. The common camera frame and "
        "the complete executed traces locate the fork. Dotted lines are the independently nominated outbound plans. "
        "Dark lines show their realized motion before arrival or interruption, and pale lines show neutral continuation. "
        "Squares locate the nominated goals and crosses mark an interruption. "
        "Both surface panels use the same oblique projection of fixed "
        "reference samples registered to the common camera axes. Colored points at action 96 indicate samples that "
        "were unrepresented at the fork and measured by only S or D. The final panel displays samples that retain "
        "that same exclusive status at actual return. The grey reference is evaluator-only geometry.", "",
    ]
    (paper/"captions.md").write_text("\n".join(caps))


if __name__=="__main__":
    main()
