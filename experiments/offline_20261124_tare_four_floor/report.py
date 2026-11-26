"""Plot-free, paper-ready paired results generated from the checked evaluator rows."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/formal"
NAMES = {"ind26_vertical_spine": "直线主走廊", "ind26_bent_corridor": "折线走廊",
         "ind26_offset_ring": "错位环道", "ind26_double_loop": "双环"}


def metric(row, name):
    value = row[name]
    return f"{100 * float(value):.2f}%" if value else "缺测"


def main():
    results = json.loads((RESULTS / "summary.json").read_text())
    episodes = {row["layout"]: row for row in results["cases"]}
    with (RESULTS / "pairs.csv").open() as stream:
        pairs = {(row["layout"], row["checkpoint"]): row for row in csv.DictReader(stream)}
    with (ROOT.parent / "region_independent_floors_20261123/results/formal/episodes.csv").open() as stream:
        nso = {row["layout"]: row for row in csv.DictReader(stream) if row["arm"] == "combined_S"}

    def threshold(paid, final_c):
        return str(paid) if paid else f"未达（{100 * float(final_c):.2f}%）"

    lines = ["# TARE 对照结果", "",
             "| 楼层 | A95 NSO | A95 TARE | 第 96 步 C NSO／TARE | 第 96 步 F1 NSO／TARE | 第 96 步 J NSO／TARE | J 差（百分点） |",
             "| --- | ---: | ---: | --- | --- | --- | ---: |"]
    for layout, name in NAMES.items():
        episode = episodes[layout]
        row = pairs[layout, "paid_96"]
        a = threshold(nso[layout]["A95_paid"], nso[layout]["C_free"])
        b = threshold(episode.get("A95_paid"), episode["C_free"]) if "C_free" in episode else "缺测"
        cells = ["／".join(metric(row, f"{method}_{key}") for method in ("nso", "explore"))
                 for key in ("C_free", "F1_scene_05cm", "J")]
        delta = row["nso_minus_explore_J_pp"]
        lines.append("| " + " | ".join([name, a, b, *cells, f"{float(delta):+.2f}" if delta else "缺测"]) + " |")

    lines.extend(["", "| 楼层 | 终点步 NSO／TARE | 终点 J NSO／TARE | 终点召回 NSO／TARE | TARE 返航／碰撞 |",
                  "| --- | --- | --- | --- | --- |"])
    for layout, name in NAMES.items():
        episode = episodes[layout]
        row = pairs[layout, "terminal"]
        cells = ["／".join(metric(row, f"{method}_{key}") for method in ("nso", "explore"))
                 for key in ("J", "r_scene_05cm")]
        clocks = "／".join(row.get(f"{method}_actual_paid") or "缺测" for method in ("nso", "explore"))
        disposition = "缺测" if "actual_return" not in episode else f"{'是' if episode['actual_return'] else '否'}／{episode['collisions']}"
        lines.append("| " + " | ".join([name, clocks, *cells, disposition]) + " |")

    mean = results["mean_nso_minus_explore_A96_J_pp"]
    average = f"{mean:+.2f} 个百分点" if mean is not None else "缺测"
    lines.extend(["", f"第 96 步有 {results['paired_A96_J_floors']}/4 个可配对楼层，NSO 相对 TARE 的 J 平均差为 {average}。"
                  f"胜／平／负为 {results['nso_A96_J_win']}／{results['nso_A96_J_tie']}／{results['nso_A96_J_loss']}。"
                  "比较单位为楼层。终点动作数不同时，终点质量描述各自实际停止时的地图。", "",
                  "A95 为首次达到 95% 正确可达自由区覆盖的计费动作数。J 为该覆盖率与全场景表面 F1@5 cm 的乘积。每个楼层只运行一次原生随机采样。", "",
                  "[逐楼所有检查点](results/formal/pairs.csv) · [逐动作覆盖](results/formal/coverage.csv) · [运行和核验](README.md)", ""])
    with (ROOT / "RESULTS.md").open("x") as stream:
        stream.write("\n".join(lines))


if __name__ == "__main__":
    main()
