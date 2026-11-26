"""Publish checked factorial results without imputing unavailable measurements."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from math import fsum, isclose
from os.path import relpath
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEDULE = ROOT.parent / "factor/PROTOCOL.json"
DEFAULT_INPUT = ROOT.parent / "factor/results/formal"
ARMS = ("D0G0", "D1G0", "D0G1", "D1G1")
FLOORS = {
    "ind26_vertical_spine": ("直线主走廊", "Straight corridor"),
    "ind26_bent_corridor": ("折线走廊", "Bent corridor"),
    "ind26_offset_ring": ("错位环道", "Offset ring"),
    "ind26_double_loop": ("双环", "Double loop"),
}
CONTRASTS = (
    ("D_at_G0", "D1G0", "D0G0", "终点收益下改变归属"),
    ("D_at_G1", "D1G1", "D0G1", "去程收益下改变归属"),
    ("G_at_D0", "D0G1", "D0G0", "站位归属下使用去程收益"),
    ("G_at_D1", "D1G1", "D1G0", "目的区归属下使用去程收益"),
)
INTERACTION = "D_by_G"
TOLERANCE = 1e-9


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, converter=float):
    value = row.get(key)
    return None if value in (None, "") else converter(value)


def percentage(value):
    return "缺测" if value is None else f"{100 * value:.2f}%"


def diff_pp(value):
    if value is None:
        return "缺测"
    return "0.00" if abs(value) <= TOLERANCE else f"{value:+.2f}"


def sign(value, threshold=0.0):
    if value is None:
        return "missing"
    return "positive" if value > threshold else "negative" if value < -threshold else "zero"


def threshold(episode):
    paid = number(episode, "A95_paid", int)
    if paid is not None:
        return str(paid)
    if episode["status"] == "completed_execution":
        return f"未达（{percentage(number(episode, 'C_free'))}）"
    return "技术故障"


def check_diff(value, expected, context):
    if value is None and expected is None:
        return
    if expected is None or value is None or not isclose(value, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Factor difference mismatch: {context}")


def read_evidence(directory):
    summary = json.loads((directory / "summary.json").read_text())
    layouts = json.loads(SCHEDULE.read_text())["layouts"]
    for name in ("checkpoints.csv", "episodes.csv", "factor_effects.csv"):
        with (directory / name).open("rb") as stream:
            sha = file_digest(stream, "sha256").hexdigest()
        if sha != summary["results_sha256"][name]:
            raise ValueError(f"Result file digest changed: {name}")
    episodes = {(r["layout"], r["arm"]): r for r in csv_rows(directory / "episodes.csv")}
    checkpoints = {(r["layout"], r["arm"]): r for r in csv_rows(directory / "checkpoints.csv")
                   if r["checkpoint"] == "paid_96"}
    effects = {(r["layout"], r["contrast"]): r for r in csv_rows(directory / "factor_effects.csv")}
    if len(episodes) != len(layouts)*len(ARMS) or len(effects) != len(layouts)*5:
        raise ValueError("Incomplete factorial matrix")
    for layout in layouts:
        for arm in ARMS:
            check = checkpoints[layout, arm]
            fields = [number(check, key) for key in ("C_free", "F1_scene_05cm", "J")]
            if check["status"] == "captured":
                c, f1, j = fields
                if not (number(check, "actual_paid", int) == 96 and isclose(j, c*f1, rel_tol=0, abs_tol=1e-10)):
                    raise ValueError(f"Invalid equal-clock joint score for {layout}/{arm}")
            elif any(value is not None for value in fields):
                raise ValueError(f"Missing checkpoint treated as measured: {layout}/{arm}")
        for code, left, right, _ in CONTRASTS:
            effect = effects[layout, code]
            if (effect["left_arm"], effect["right_arm"]) != (left, right):
                raise ValueError(f"Factor arm sign unexpectedly reversed: {layout}/{code}")
            ua, ub = [number(episodes[layout, arm], "A95_paid", int) for arm in (left, right)]
            expected = ua-ub if ua is not None and ub is not None else None
            if number(effect, "delta_A95_steps", int) != expected:
                raise ValueError(f"A95 censoring or sign error: {layout}/{code}")
            for metric in ("C_free", "F1_scene_05cm", "J"):
                va, vb = [number(checkpoints[layout, arm], metric) for arm in (left, right)]
                expected = 100*(va-vb) if va is not None and vb is not None else None
                check_diff(number(effect, f"delta_{metric}_96_pp"), expected, f"{layout}/{code}/{metric}")
        a, b, c, d = [number(checkpoints[layout, arm], "J") for arm in ARMS]
        expected = (d-b-c+a)*100 if None not in (a, b, c, d) else None
        check_diff(number(effects[layout, INTERACTION], "delta_J_96_pp"), expected, layout)
    return layouts, summary, episodes, checkpoints, effects


def format_table(header, rows):
    return ["| " + " | ".join(header) + " |",
            "| " + " | ".join(("---",)*len(header)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def paper_paragraph(layouts, episodes, aggregates, interactions):
    n = len(layouts)
    reached = {arm:sum(number(episodes[layout, arm], "A95_paid", int) is not None
                       for layout in layouts) for arm in ARMS}
    d0, d1 = aggregates["D_at_G0"], aggregates["D_at_G1"]
    g0, g1 = aggregates["G_at_D0"], aggregates["G_at_D1"]
    clause = ("门后区域归属在 G0、G1 下的第 96 步联合质量平均差分别为 "
              f"{diff_pp(d0)} 和 {diff_pp(d1)} 个百分点。"
              f"完整去程收益替换在 D0、D1 下的平均差分别为 {diff_pp(g0)} 和 {diff_pp(g1)} 个百分点。"
              "达到 95% 覆盖的楼层数为 " + "、".join(f"{arm} {reached[arm]}/{n}" for arm in ARMS) + "。")
    if interactions:
        mi, ma = min(interactions), max(interactions)
        clause += f"交互差在 {diff_pp(mi)} 至 {diff_pp(ma)} 个百分点之间。"
        if mi < -TOLERANCE and ma > TOLERANCE:
            clause += "两项改动的交互方向随布局变化。"
    return clause + "该对照估计四个冻结楼层上的条件效应。G1 的各收益计算变化共同替换了一个估计器。"


def make_report(input_dir, output_dir, data):
    layouts, summary, episodes, checkpoints, effects = data
    rows = []
    for layout in layouts:
        for arm in ARMS:
            episode, clock = episodes[layout, arm], checkpoints[layout, arm]
            distance = number(episode, "path_length_m")
            returned = {"True":"是", "False":"否"}.get(episode.get("returned"), "缺测")
            rows.append([FLOORS[layout][0], arm, "复用" if episode["reused"] == "True" else "新增",
                         threshold(episode), *(percentage(number(clock, key)) for key in ("C_free", "F1_scene_05cm", "J")),
                         episode.get("actual_paid") or "缺测", returned, episode.get("collisions") or "缺测",
                         "缺测" if distance is None else f"{distance:.1f}"])
    lines = ["# A 阶段四楼因子消融", "",
             f"核验完成新增闭环 {summary['completed_new']}/8 轮、复用完整轨迹 {summary['completed_reused']}/8 轮。"
             f"技术故障共 {summary['failed']} 轮。每个楼层是一个比较单位，预算上限为 192 个计费动作，含转向与返航。", "",
             "D1 将门口候选归到地图已识别的门后区域，其他情况沿用站位区域。G1 以去程累计覆盖与已测表面再观测替换封顶的终点收益。两因素均在相同候选生成规则与局部 S 策略下实现。", ""]
    lines.extend(format_table(["楼层", "组合", "来源", "A95（步）", "C96", "F1@5 cm（第 96 步）",
                              "J96", "终点步数", "返航", "碰撞次数", "路程（m）"], rows))
    rows = []
    for layout in layouts:
        row = [FLOORS[layout][0]]
        for name, *_ in CONTRASTS:
            result = effects[layout, name]
            steps = number(result, "delta_A95_steps", int)
            step_text = "—" if steps is None else f"{steps:+d}"
            row.append(f"{diff_pp(number(result, 'delta_J_96_pp'))}／{step_text}")
        row.append(diff_pp(number(effects[layout, INTERACTION], "delta_J_96_pp")))
        rows.append(row)
    lines.extend(["## 条件差值", "",
                  "四列条件差值各显示 ΔJ96（百分点）／ΔA95（步）。正的联合质量差表示提高。负的动作差表示更早达标。任一组合未达 95% 时动作差留空。交互项报告联合质量差。", ""])
    lines.extend(format_table(["楼层", "D 在 G0 下", "D 在 G1 下", "G 在 D0 下", "G 在 D1 下", "D×G（百分点）"], rows))
    aggregates, rows = {}, []
    for name, _, _, label in CONTRASTS:
        values = [number(effects[layout, name], "delta_J_96_pp") for layout in layouts]
        steps = [number(effects[layout, name], "delta_A95_steps", int) for layout in layouts]
        count = Counter(sign(value, TOLERANCE) for value in values)
        a95 = Counter(sign(value) for value in steps)
        known = [value for value in values if value is not None]
        aggregates[name] = fsum(known)/len(known) if known else None
        rows.append([label, f"{count['positive']}／{count['zero']}／{count['negative']}／{count['missing']}",
                     diff_pp(aggregates[name]), f"{a95['negative']}／{a95['zero']}／{a95['positive']}／{a95['missing']}"])
    lines.extend(format_table(["对照", "J96 胜／平／负／缺测", "J96 楼层等权均值（百分点）", "A95 早／同／晚／未作差"], rows))
    interactions = [number(effects[layout, INTERACTION], "delta_J_96_pp") for layout in layouts]
    lines.extend(["## 可写入稿件的结论", "",
                  paper_paragraph(layouts, episodes, aggregates, [x for x in interactions if x is not None]), "",
                  "C 为正确可达自由空间覆盖率，F1 使用 5 cm 匹配阈值。J 在比例尺度按 C×F1 计算，之后换算为百分比或百分点。A95 取首次达到 95% 覆盖的计费动作，未达者括号内为实际终点覆盖。", "",
                  f"[逐轮指标]({relpath(input_dir / 'episodes.csv', output_dir)}) · "
                  f"[完整检查点]({relpath(input_dir / 'checkpoints.csv', output_dir)}) · "
                  f"[条件效应 CSV]({relpath(input_dir / 'factor_effects.csv', output_dir)})", ""])
    (output_dir / "A_FACTOR_RESULTS.md").write_text("\n".join(lines))


def make_plot(output, data):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    layouts, _, _, clocks, contrasts = data
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5), sharex=True, sharey=True)
    styles = (("#0072B2", "o"), ("#D55E00", "s"))
    for index, (panel, layout) in enumerate(zip(axes.ravel(), layouts, strict=True)):
        for g, (color, marker) in enumerate(styles):
            values = [number(clocks[layout, f"D{d}G{g}"], "J") for d in range(2)]
            panel.plot([0, 1], [value*100 if value is not None else float("nan") for value in values],
                       color=color, marker=marker, lw=1.7, ms=5.5, zorder=3)
        interaction = number(contrasts[layout, INTERACTION], "delta_J_96_pp")
        detail = f"Interaction {diff_pp(interaction)} pp" if interaction is not None else "Interaction unavailable"
        panel.set_title(f"({chr(ord('a')+index)}) {FLOORS[layout][1]}\n{detail}", loc="left", fontsize=9)
        panel.set_xlim(-.20, 1.20)
        panel.set_xticks([0, 1], ["D0", "D1"])
        if index >= 2:
            panel.set_xlabel("Doorway region attribution")
        if index % 2 == 0:
            panel.set_ylabel("Joint quality at action 96 (%)")
        panel.grid(axis="y", color="#dddddd", linewidth=.62, zorder=0)
        panel.spines[["right", "top"]].set_visible(False)
    legend = [Line2D([0], [0], color=color, marker=marker, lw=1.7, ms=5.5, label=label)
              for (color, marker), label in zip(styles, ("G0  Endpoint", "G1  Outbound"), strict=True)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.50, .987), ncols=2, frameon=False)
    fig.subplots_adjust(top=.86, bottom=.12, left=.13, right=.97, hspace=.44, wspace=.13)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"factor_interactions.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    directory, out = args.input.resolve(), args.output.resolve()
    data = read_evidence(directory)
    out.mkdir(parents=True, exist_ok=True)
    make_report(directory, out, data)
    if not args.no_plot:
        make_plot(out, data)


if __name__ == "__main__":
    main()
