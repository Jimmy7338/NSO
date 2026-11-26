"""Audit registered local score margins without rerunning actors or changing policies."""

import argparse
import csv
from hashlib import file_digest, sha256
import json
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checksum(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def near(left, right):
    return isclose(left, right, rel_tol=0, abs_tol=1e-12)


def number(value):
    return float(value) if value else None


def verified_run(assignment, placement_meta, phase_sha):
    directory = Path(assignment["run_directory"]).resolve()
    require(directory.is_relative_to(ROOT / "execution/formal" / placement_meta["placement"]),
            f"Wrong placement evidence directory {directory}")
    manifest_path = directory / "manifest.json"
    manifest, bank = load(manifest_path), load(directory / "fork_bank.json")
    require(digest(manifest_path) == assignment["manifest_sha256"]
            and digest(directory / "fork_bank.json") == assignment["fork_bank_sha256"]
            and manifest["status"] == "completed_execution" and manifest["layout"] == placement_meta["placement"]
            and manifest["study_freeze_sha256"] == bank["freeze_sha256"] == phase_sha,
            f"Mismatched execution receipt {directory}")
    for filename in ("fork_bank.json", "steps.json"):
        require(digest(directory / filename) == manifest["files_sha256"][filename],
                f"Volatile trace file {directory / filename}")
    return directory, manifest, bank


def score_stats(directory, manifest, bank, phase):
    candidates_path = directory / "fork_candidates.json"
    candidate_sha = digest(candidates_path)
    audit = load(candidates_path)
    bank_rows = audit["candidates"]
    require(candidate_sha == manifest["files_sha256"]["fork_candidates.json"]
            and checksum(bank_rows) == bank["candidate_scores_sha256"]
            and checksum([{k:v for k,v in row.items() if k!="scores"} for row in bank_rows]) == bank["candidate_pool_sha256"],
            f"Fork candidate bytes, scoring table or legal candidate pool drift: {directory}")
    eligible = {r["route_id"]:r for r in bank_rows if r["admitted"]}
    require(len(eligible) >= 2 and len({r["outbound_sha256"] for r in eligible.values()}) == len(eligible),
            f"Admitted fork has duplicate or insufficient outbound choices: {directory}")
    hidden = {r["route_id"]:r["supported_hidden_m2"] for r in audit["supported_hidden_by_route"]}
    ranks = {arm:{r["route_id"]:r for r in values} for arm,values in audit["rankings"].items()}
    require(set(ranks) == set(phase["arm_order"]) and set(hidden) == set(eligible)
            and all(set(ranked) == set(eligible) for ranked in ranks.values()),
            f"Ranking or hidden support does not cover all legal routes: {directory}")
    for route in eligible:
        direct, added = ranks["D"][route], ranks["D+"][route]
        gain = phase["hidden_scale"] * hidden[route]
        require(gain >= 0 and near(added["area_m2"]-direct["area_m2"], gain)
                and near(added["score_m2"]-direct["score_m2"], gain)
                and added["delta_unknown_union_m2"] == direct["delta_unknown_union_m2"]
                and added["extra_outbound_actions"] == direct["extra_outbound_actions"],
                f"D+ differs from D outside its registered supported area: {directory}, {route}")
        for arm in phase["arm_order"]:
            row = ranks[arm][route]
            weight = phase["weights"]
            score = (row["area_m2"] + weight["unknown_weight"] * row["delta_unknown_union_m2"]
                     - weight["action_cost"] * row["extra_outbound_actions"])
            require(near(score, row["score_m2"]) and near(row["area_m2"], eligible[route]["scores"][arm]),
                    f"Recorded weighted {arm} score is inconsistent: {directory}, {route}")
    route = {arm: record["route_id"] for arm,record in bank["arms"].items()}
    score = lambda arm, path: ranks[arm][path]["score_m2"]
    for arm, selected in route.items():
        best = max(score(arm, name) for name in eligible)
        require(best - score(arm, selected) <= phase["local_route"]["score_equality_tolerance"],
                f"Registered selected route is not a maximal common-pool choice: {directory}/{arm}")
    incumbent = route["D"]
    challenger = max((name for name in eligible if name != incumbent), key=lambda name:(score("D",name),name))
    plus_challenger = max((name for name in eligible if name != incumbent), key=lambda name:(score("D+",name),name))
    result = {
        "actor_fork_candidates_sha256":candidate_sha,
        "actor_fork_candidate_pool_sha256":bank["candidate_pool_sha256"],
        "actor_admitted_routes":len(eligible),
        "actor_D_plus_nonzero_hidden_routes":sum(gain>0 for gain in hidden.values()),
        "actor_D_plus_max_supported_hidden_m2":max(hidden.values()),
        "actor_D_winner_supported_hidden_m2":hidden[incumbent],
        "actor_D_top_alternative":challenger,
        "actor_D_top_margin_m2_equivalent":score("D",incumbent)-score("D",challenger),
        "actor_D_plus_top_alternative_to_D_winner":plus_challenger,
        "actor_D_plus_margin_of_D_winner_m2_equivalent":score("D+",incumbent)-score("D+",plus_challenger),
        "actor_D_plus_top_alternative_hidden_difference_to_D_winner_m2":hidden[plus_challenger]-hidden[incumbent],
        "actor_D_plus_changed_route":bank["arms"]["D"]["route_identity_sha256"] != bank["arms"]["D+"]["route_identity_sha256"],
        "actor_S_G_different_route":bank["arms"]["S"]["route_identity_sha256"] != bank["arms"]["G"]["route_identity_sha256"],
        "actor_G_preference_for_G_over_S_route_m2_equivalent":score("G", route["G"])-score("G", route["S"]),
        "actor_S_preference_for_S_over_G_route_m2_equivalent":score("S", route["S"])-score("S", route["G"]),
    }
    return result | {f"actor_selected_route_{a.replace('+','_plus')}":r for a,r in route.items()}


def measured(fields, placement_rows, receipt, phase_sha):
    comparison = fields["S_minus_G"]
    result = {"measured_paid96_S_minus_G_shelf_net_m2":number(comparison["net_m2_delta"]),
              "measured_paid96_S_minus_D_shelf_net_m2":number(fields["S_minus_D"]["net_m2_delta"]),
              "measured_paid96_D_plus_minus_D_shelf_net_m2":number(fields["D_plus_minus_D"]["net_m2_delta"]),
              "measured_divergence_while_both_following_bound_route":comparison["difference_during_both_bound_fork_routes"] or None}
    if not all(placement_rows[arm]["status"] == "verified_execution" for arm in ("G","S")):
        return result | {"measured_S_G_first_divergent_paid":None, "measured_S_G_status":"technical_missing"}
    traces = []
    for arm in ("G", "S"):
        d,_,_ = verified_run(receipt["arms"][arm], receipt, phase_sha)
        traces.append(load(d / "steps.json"))
    difference = next((a["step"] for a,b in zip(*traces)
                       if any(a[k] != b[k] for k in ("action", "position", "heading"))), None)
    declared = int(comparison["first_actual_difference_paid"]) if comparison["first_actual_difference_paid"] else None
    require(difference == declared, f"S/G actual execution divergence not reproducible: {receipt['placement']}")
    return result | {"measured_S_G_first_divergent_paid":difference, "measured_S_G_status":"both_actual"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    phase_path = ROOT / "FORMAL_FREEZE.json"
    phase, phase_sha = load(phase_path), digest(phase_path)
    summary_path = args.results / "summary.json"
    summary = load(summary_path)
    require(summary["phase"]=="formal" and summary["phase_freeze_sha256"]==phase_sha,
            "This diagnostic requires sealed formal measurement inputs")
    for name in ("episodes.csv", "comparisons.csv"):
        require(digest(args.results/name)==summary["table_sha256"][name], f"Changed verified measurement: {name}")
    episodes = {(r["placement"],r["arm"]):r for r in rows(args.results/"episodes.csv")}
    fixed = {(r["placement"],r["pair"]):r for r in rows(args.results/"comparisons.csv") if r["clock"]=="96"}
    results = []
    for placement in phase["placements"]:
        if placement not in summary["conditions"]:
            continue
        receipt_path = ROOT / "execution/formal" / placement / "placement_receipt.json"
        receipt = load(receipt_path)
        require(digest(receipt_path)==summary["conditions"][placement]["placement_receipt_sha256"],
                f"Formal placement record drift: {placement}")
        selected = next((r for r in receipt["arms"].values()
                         if r["status"]=="verified_native_execution" and not r["shared_execution"]), None)
        result = {"placement":placement,
                  **measured({k:v for (p,k),v in fixed.items() if p==placement},
                             {a:episodes[placement,a] for a in phase["arm_order"]}, receipt,phase_sha)}
        if selected is None:
            results.append(result | {"actor_status":"technical_missing"})
            continue
        directory,manifest,bank = verified_run(selected,receipt,phase_sha)
        result |= {"actor_status":bank["status"], "actor_verified_bank_source_arm":selected["execution_arm"],
                   "actor_actual_fork_paid":bank["paid_before"]}
        if bank["status"]=="triggered":
            result |= score_stats(directory,manifest,bank,phase)
        results.append(result)
    require(len(results)==len(summary["conditions"]), "Not all measured placements were audited")
    totals = {"placement_count":len(results),
              "actor_forks":sum(r["actor_status"]=="triggered" for r in results),
              "actor_nonzero_hidden_support":sum(r.get("actor_D_plus_nonzero_hidden_routes",0)>0 for r in results),
              "actor_D_plus_changed_route":sum(r.get("actor_D_plus_changed_route",False) for r in results),
              "actor_S_G_changed_route":sum(r.get("actor_S_G_different_route",False) for r in results),
              "measured_S_G_actual_divergence":sum(r["measured_S_G_first_divergent_paid"] is not None for r in results)}
    args.out.mkdir(parents=True)
    columns = list(dict.fromkeys(field for row in results for field in row))
    with (args.out/"placements.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,columns); writer.writeheader(); writer.writerows(results)
    with (args.out/"summary.json").open("w") as stream:
        json.dump({"formal_freeze_sha256":phase_sha, "measurement_summary_sha256":digest(summary_path),
                   "placements_csv_sha256":digest(args.out/"placements.csv"), **totals},stream,indent=2)
        stream.write("\n")
    print(json.dumps(totals,indent=2))


if __name__ == "__main__":
    main()
