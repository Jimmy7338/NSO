#!/usr/bin/env python3
"""Publish measured outcomes and the paired comparison with fixed endpoints."""

from pathlib import Path
import sys

from runner import PREVIOUS, ROOT


sys.path.append(str(PREVIOUS))
from regression import read, sha


def value(val, digits=2):
    return "—" if val is None else f"{val:.{digits}f}"


def percentage(val):
    return "—" if val is None else f"{100*val:.2f}"


def signed_pp(val):
    return "—" if val is None else f"{100*val:+.2f}"


def pair_table(summary):
    rows = ["| 楼层 | 对照 | Δ 正确自由覆盖（百分点） | Δ F1（百分点） | Δ J（百分点） | Δ A95（步） | Δ 重复路程（m） |",
            "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"revision_vs_regional": "修订 − 原区域", "revision_vs_strict_frontier": "修订 − 严格前沿",
              "regional_vs_strict_frontier": "原区域 − 严格前沿",
              "revision_vs_archived_native_S": "修订 − 原生 S"}
    for row in summary["contrasts"]:
        cells = [row["layout"], labels[row["contrast"]]]
        cells.extend(signed_pp(row.get(key)) for key in ("delta_C_free", "delta_F1_scene_05cm", "delta_J"))
        cells.extend((value(row.get("delta_A95_paid"), 0), value(row.get("delta_repeat_distance_m"), 1)))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def episode_table(rows):
    table = ["| 楼层 | 策略 | 实付步 | 覆盖 % | F1 % | J % | A95 | 重复路程（m） | 实访房间 | 返回 | 碰撞 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        fields = [row["layout"], row["arm"], str(row.get("actual_paid", "—")),
                  *(percentage(row.get(k)) for k in ("C_free", "F1_scene_05cm", "J")),
                  str(row.get("A95_paid") or "—"), value(row.get("repeat_distance_m"), 1),
                  str(row.get("rooms_visited", "—")), {True: "是", False: "否"}.get(row.get("returned"), "—"),
                  str(row.get("collisions", "—"))]
        table.append("| " + " | ".join(fields) + " |")
    return table


def main():
    output = ROOT / "results"
    dev, holdout = (read(output / scope / "summary.json") for scope in ("development", "formal"))
    for scope, summary in (("development", dev), ("formal", holdout)):
        assert summary["status"] == "evaluated" and summary["pending_trials"] == 0
        assert summary["run_freeze_sha256"] == sha(ROOT / f"{scope.upper()}_FREEZE.json")
    historic = read(PREVIOUS / "results/formal/summary.json")
    retained = read(ROOT / "PROTOCOL.json")["development"]
    controls = [row for row in historic["episodes"] if row["arm"] in retained["archived_controls"]
                and row["layout"] in retained["layouts"]]
    attempts = ROOT / "execution/formal"
    recovered = [r for p in attempts.glob("*/*/attempt_*/exit.json") if (r:=read(p))["exit_code"] != 0]
    result = ["# 区域全局选点修订的配对验证", "",
              "只移除原有区域选点中可见未知面积的 2 m² 上限。各策略在同一楼层从同一首帧独立开始，"
              "上限为 192 个计费动作，含转向和返航。原区域与修订共用 S 局部策略和原生可行性、"
              "足迹保护及执行器。严格前沿作为独立参考。覆盖使用正确识别的可达自由面积，"
              "F1 的表面距离阈值为 5 cm，J 为覆盖和 F1 的乘积。A95 是第一次达到 95% 覆盖所用的实际动作数。"
              "未达到或缺测的数字以横线表示。", "",
              "## 未见楼层", "",
              f"六个预登记回合中有 {holdout['complete_endpoints']} 个可核验终点，"
              f"有 {holdout['technical_failures']} 个回合因技术中断缺少终点。"
              + (f"共有 {len(recovered)} 次技术中断尝试，均保留。恢复沿用原封存配置。"
                 if recovered else ""), "",
              *episode_table(holdout["episodes"]), "", *pair_table(holdout), "",
              "## 已见故障布局", "",
              "开发重跑中的历史对照来自正式归档。它们的冻结协议、起始数据、动作预算和源清单已核对。"
              "以下结果仅用于解释开发案例。", "",
              *episode_table(sorted(dev["episodes"] + controls, key=lambda row: (row["layout"], row["arm"]))),
              "", *pair_table(dev), "",
              "两个留出楼层分别代表走廊和环路，本批的差值按单栋呈现。未知空间估计沿用原可见性代理，"
              "尚未被看到的遮挡物会令代理高估。仿真给定精确位姿及已观测物体的理想类别。", "",
              "原始分歧见 [固定状态诊断](../diagnosis/README.md)。"
              "单位动作覆盖、到达和重建检查点见 [评估目录](formal/)。", ""]
    (output / "RESULTS.md").write_text("\n".join(result))


if __name__ == "__main__":
    main()
