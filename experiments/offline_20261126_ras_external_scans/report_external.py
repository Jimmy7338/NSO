"""Render start-paired and floor-equal comparisons from frozen measurements."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/formal"
ARMS = ("combined_S", "region_S", "F_strict")


def load_csv(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def percent(value):
    return f"{100 * float(value):.2f}%" if value not in ("", None) else "缺测"


def difference(value):
    return f"{float(value):+.2f}" if value not in ("", None) else "缺测"


def main():
    protocol = json.loads((ROOT / "FORMAL_PROTOCOL.json").read_text())
    episodes = {(r["layout"], r["arm"]): r for r in load_csv("episodes.csv")}
    capture = {(r["layout"], r["arm"], r["checkpoint"]): r for r in load_csv("checkpoints.csv")}
    group = {(r["source_floor"], r["control"]): r for r in load_csv("external_source_floor_means.csv")}
    average = json.loads((RESULTS / "external_source_summary.json").read_text())
    text = ["# 异源三维楼层验证", "",
            "策略顺序为完整区域规划、原区域规划、严格最近可达前沿。每个楼层登记两个起点，每个起点共用首帧和 192 步预算。语义条件、真实扫描来源与几何准入见协议。",
            "", "| 来源楼层／起点 | A95（步） | 第 96 步 C | 第 96 步 F1@5 cm | 第 96 步 J |",
            "| --- | --- | --- | --- | --- |"]
    for start in protocol["starts"]:
        layout = start["layout"]
        e = [episodes[layout, arm] for arm in ARMS]
        a95 = "／".join((r["A95_paid"] or f"未达（{percent(r['C_free'])}）")
                         if r["status"] == "completed_execution" else "技术失败" for r in e)
        mid = [capture.get((layout, arm, "paid_96"), {}) for arm in ARMS]
        data = ["／".join(percent(r.get(metric)) for r in mid)
                for metric in ("C_free", "F1_scene_05cm", "J")]
        text.append("| " + " | ".join([layout, a95, *data]) + " |")
    text.extend(["", "| 来源楼层 | 参照方法 | 已配对起点数 | J96 平均差（百分点） | J96 胜／平／负 | 共同达标起点 | A95 平均动作差 |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for (floor, control), row in sorted(group.items()):
        wtl = "／".join(row[k] for k in ("revised_J96_wins", "revised_J96_ties", "revised_J96_losses"))
        text.append(f"| {floor} | {control} | {row['paired_J96_starts']}/{row['registered_starts']} | "
                    f"{difference(row['mean_delta_J_96_pp'])} | {wtl} | {row['paired_A95_starts']} | "
                    f"{difference(row['mean_A95_delta'])} |")
    for arm, row in average["source_floor_equal_means"].items():
        text.append(f"\n相对 {arm} 的楼层等权 J96 平均差为 {difference(row['floor_equal_mean_delta_J_96_pp'])} 个百分点"
                    f"（可比较楼层 {row['J96_available_source_floors']}/{row['registered_source_floors']}）。")
    text.extend(["", "A95 仅在两种方法都达标时计算动作差。J 是正确可达自由区覆盖率与全场景 F1@5 cm 的乘积。各检查点的缺测留空。每个起点的实际终点、返航、碰撞和距离见逐轮数据。",
                 "", "[逐轮数据](results/formal/episodes.csv) · [固定时点及真实终点](results/formal/checkpoints.csv) · [起点配对](results/formal/external_start_pairs.csv) · [楼层等权汇总](results/formal/external_source_summary.json)", ""])
    with (ROOT / "RESULTS.md").open("x") as stream:
        stream.write("\n".join(text))


if __name__ == "__main__":
    main()
