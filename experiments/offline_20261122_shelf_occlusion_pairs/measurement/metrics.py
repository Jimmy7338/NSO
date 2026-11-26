"""Mask arithmetic and paired shelf outcomes on the frozen action clock."""

from statistics import mean, median

import numpy as np


ARMS = ("G", "S", "D", "D+")
COMPARISONS = (("S", "D"), ("S", "G"), ("D+", "D"))
CLOCKS = (48, 96, 144, 192, "terminal")
NAMES = {"D+": "D_plus"}


def pair_name(left, right):
    return f"{NAMES.get(left, left)}_minus_{NAMES.get(right, right)}"


def area(mask, weights):
    return float(np.sum(weights[mask], dtype=np.float64))


def shelf_at_clock(start, end, weights):
    if start.shape != end.shape or weights.shape != end.shape:
        raise ValueError("Start, current shelf mask and fixed weights have different sample indices.")
    gross = area(end & ~start, weights)
    lost = area(start & ~end, weights)
    return {
        "gross_m2": gross, "lost_m2": lost, "net_m2": gross - lost,
        "represented_m2": area(end, weights), "cold_support_m2": area(start, weights),
        "evaluable_m2": float(weights.sum()), "evaluable_samples": len(end),
    }


def snap_key(clock):
    return "terminal" if clock == "terminal" else f"paid_{clock}"


def first_actual_route_difference(a, b):
    for left, right in zip(a, b, strict=False):
        if left["step"] != right["step"]:
            raise ValueError("Paired actual action clocks differ.")
        if any(left[key] != right[key] for key in ("action", "position", "heading")):
            return left["step"]
    return None


def classify(value, tolerance):
    if value is None:
        return "missing"
    if abs(value) <= tolerance:
        return "tie"
    return "win" if value > 0 else "loss"


def compare_pair(left, right, weights, common_fork, tolerance):
    fact = {"same_execution": None, "same_selected_route": None, "first_actual_difference_paid": None,
            "difference_during_both_bound_fork_routes": None}
    both = left.get("status") == right.get("status") == "verified_execution"
    if both:
        fact["same_execution"] = left["manifest_sha256"] == right["manifest_sha256"]
        if left["triggered"] and right["triggered"]:
            fact["same_selected_route"] = left["route_identity_sha256"] == right["route_identity_sha256"]
        difference = first_actual_route_difference(left["trace"], right["trace"])
        fact["first_actual_difference_paid"] = difference
        if difference is not None and common_fork is not None:
            fork = left["fork_paid"]
            together = min(left["actual_bound_outbound_actions"], right["actual_bound_outbound_actions"])
            fact["difference_during_both_bound_fork_routes"] = fork < difference <= fork + together
    rows, retention, export = [], {}, {}
    for clock in CLOCKS:
        key = snap_key(clock)
        a = left.get("snapshots", {}).get(key)
        b = right.get("snapshots", {}).get(key)
        row = fact | {"clock": clock, "left_paid": a["paid"] if a else None, "right_paid": b["paid"] if b else None,
                      "measured": bool(a and b), "equal_paid_clock": bool(a and b and a["paid"] == b["paid"])}
        for field in ("net_m2", "C_free", "F1_scene_05cm", "J"):
            row[f"{field}_delta"] = a[field] - b[field] if a and b else None
        row["shelf_outcome"] = classify(row["net_m2_delta"], tolerance)
        if row["same_execution"] and row["measured"] and any(
            row[f"{field}_delta"] != 0 for field in ("net_m2", "C_free", "F1_scene_05cm", "J")
        ):
            raise ValueError("One verified trajectory has inconsistent aliases.")
        rows.append(row)

    early = (left.get("masks", {}).get("paid_96"), right.get("masks", {}).get("paid_96"))
    terminal = (left.get("masks", {}).get("terminal"), right.get("masks", {}).get("terminal"))
    archive = left.get("returned", False) and right.get("returned", False)
    retention["status"] = (
        "no_observed_fork" if common_fork is None else
        "missing_paid_96" if any(mask is None for mask in early) else
        "missing_dual_return" if not archive or any(mask is None for mask in terminal) else "measured"
    )
    if common_fork is not None and all(mask is not None for mask in early):
        for index, side in enumerate(("left", "right")):
            cohort = early[index] & ~early[1-index] & ~common_fork
            retention[f"{side}_exclusive_96_m2"] = area(cohort, weights)
            retention[f"{side}_exclusive_96_samples"] = int(cohort.sum())
            export[f"{side}_exclusive_96"] = cohort
            if retention["status"] == "measured":
                own, other = terminal[index], terminal[1-index]
                categories = {"still_exclusive": own & ~other, "now_shared": own & other,
                              "only_other": ~own & other, "neither": ~(own | other)}
                for category, present in categories.items():
                    mask = cohort & present
                    retention[f"{side}_{category}_return_m2"] = area(mask, weights)
                    retention[f"{side}_{category}_return_samples"] = int(mask.sum())
                    export[f"{side}_{category}_return"] = mask
                initial_area = retention[f"{side}_exclusive_96_m2"]
                retention[f"{side}_still_exclusive_fraction"] = (
                    retention[f"{side}_still_exclusive_return_m2"] / initial_area if initial_area else None
                )
                assert sum(int((cohort & present).sum()) for present in categories.values()) == int(cohort.sum())
    return rows, fact | retention, export


def describe(values, total, tolerance=None):
    known = [value for value in values if value is not None]
    data = {"n": len(known), "scheduled": total, "missing": total-len(known),
            "mean": mean(known) if known else None, "median": median(known) if known else None,
            "minimum": min(known) if known else None, "maximum": max(known) if known else None}
    if tolerance is not None:
        labels = [classify(value, tolerance) for value in known]
        data.update({label: labels.count(label) for label in ("win", "tie", "loss")}, tie_tolerance_m2=tolerance)
    return data
