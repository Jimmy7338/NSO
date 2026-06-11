#!/usr/bin/env python3
"""从 main.py 评估日志与 dump 文件汇总 JSON 报告。"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_final_line(text: str, label: str) -> List[float]:
    m = re.search(rf"{re.escape(label)}:\s*\n([0-9., \n]+)", text)
    if not m:
        return []
    return [float(x.strip()) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def _parse_numeric_row(line: str) -> List[float]:
    line = line.strip()
    if not line:
        return []
    if line.startswith("[") and line.endswith("]"):
        line = line[1:-1]
    return [float(x) for x in line.split() if x]


def _episode_max_ratios(ratio_file: Path) -> List[float]:
    if not ratio_file.is_file():
        return []
    episodes: List[float] = []
    for line in ratio_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            vals = _parse_numeric_row(line)
        except ValueError:
            continue
        if vals:
            episodes.append(max(vals))
    return episodes


def _episode_max_areas(area_file: Path) -> List[float]:
    if not area_file.is_file():
        return []
    episodes: List[float] = []
    for line in area_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            vals = _parse_numeric_row(line)
        except ValueError:
            continue
        if vals:
            episodes.append(max(vals))
    return episodes


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    return {
        "mean": float(st.mean(vals)),
        "std": float(st.pstdev(vals)) if len(vals) > 1 else 0.0,
        "median": float(st.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "count": len(vals),
    }


def summarize(
    log_path: Path,
    dump_dir: Path,
    tag: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    ratio_curve = _parse_final_line(text, "Final Exp Ratio")
    area_curve = _parse_final_line(text, "Final Exp Area")
    ep_ratios = _episode_max_ratios(dump_dir / "explored_ratio.txt")
    ep_areas = _episode_max_areas(dump_dir / "explored_area.txt")

    paper_matches = re.findall(
        r"Paper\[cov=([0-9.]+)% drift=([0-9.]+)cm unr=(\d+)\]",
        text,
    )
    cov_vals = [float(m[0]) / 100.0 for m in paper_matches if float(m[0]) > 0]
    drift_vals = [float(m[1]) for m in paper_matches if float(m[1]) > 0]
    unr_vals = [int(m[2]) for m in paper_matches]

    report: Dict[str, Any] = {
        "tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "dump_dir": str(dump_dir),
        "meta": meta or {},
        "coverage_ratio_episode_max": _stats(ep_ratios),
        "explored_area_m2_episode_max": _stats(ep_areas),
        "coverage_ratio_step_curve_mean": _stats(ratio_curve),
        "explored_area_step_curve_mean": _stats(area_curve),
        "paper_online_coverage": _stats(cov_vals),
        "paper_online_drift_cm": _stats(drift_vals),
        "paper_online_unreachable": _stats([float(x) for x in unr_vals]),
        "raw": {
            "final_exp_ratio_curve": ratio_curve,
            "final_exp_area_curve": area_curve,
            "episode_max_coverage": ep_ratios,
            "episode_max_area_m2": ep_areas,
        },
    }
    return report


def main():
    p = argparse.ArgumentParser(description="汇总 NSO 评估结果为 JSON")
    p.add_argument("--log", required=True, help="train.log 路径")
    p.add_argument("--dump", required=True, help="dump 目录")
    p.add_argument("--tag", default="paper_fast")
    p.add_argument("--output", required=True, help="输出 JSON 路径")
    p.add_argument("--meta", default="{}", help="附加元信息 JSON 字符串")
    args = p.parse_args()

    meta = json.loads(args.meta)
    report = summarize(Path(args.log), Path(args.dump), args.tag, meta)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[汇总] 已写入 {out}")

    cov = report.get("coverage_ratio_episode_max", {})
    area = report.get("explored_area_m2_episode_max", {})
    if cov:
        print(
            f"[结果] {args.tag}: episodes={cov.get('count', 0)} "
            f"coverage={cov.get('mean', 0)*100:.2f}±{cov.get('std', 0)*100:.2f}% "
            f"area={area.get('mean', 0):.1f}±{area.get('std', 0):.1f} m²"
        )


if __name__ == "__main__":
    main()
