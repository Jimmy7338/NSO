# Main paper figure: controlled virtual evidence

**English caption.** (A) All four saved V36 layout–configuration pairs for active
geometry G and semantic policy S; two zero effects and two positive effects are
retained. Labels above the bars are the signed within-condition S−G differences
in J5. (B) The same four conditions for G, S, SWAP-inspection-inspired/CPU
(SWAP-I), and VISTA-view-inspired/CPU (VISTA-I), under the shared 42-paid-action
budget (including an 18-action prefix). Grey segments join methods within each
condition; they are not interpolation or uncertainty intervals. These are CPU
mechanism adaptations, not reproductions or rankings of the original systems.
The eight G/S score rows in B match A and therefore provide no additional
independent evidence. (C) Measured map coverage and surface F1 at the fixed 5 cm
threshold are plotted on separate axes. Coverage is identical across methods
within each condition, so the observed within-condition J5 differences arise
from F1, not increased map coverage. J5 = C_map × F1@5cm is the existing saved
metric; no J_nav or new score was constructed. Every absolute score axis starts
at zero. The common sensor model, public safe graph, CPU TSDF, evaluator, exact
simulated poses and fixed public facility ROI constrain interpretation.

The evidence covers **2 seen development layouts and 4 paired conditions,
not independent repeats**. Deterministic replay verifies execution and does
not increase sample size. Condition means are descriptive with equal weight
per condition (equivalently equal weight per layout here). No confidence
intervals, significance claim, unseen-layout generalization or original-system
ranking is asserted. Public-ROI-external errors are excluded from precision.

**中文图注。** (A) V36已保存的四个布局—构型配对条件，完整保留两个零增益和
两个正增益，柱顶标注各条件的S−G联合指标差。(B) G、S、SWAP-I和VISTA-I在同样
四个条件下的终点J5，灰线只连接同条件的各方法结果；后二者是CPU机制适配，不代表
原作者完整系统排名。B中的G/S与A为相同数值，不增加独立证据。(C) 地图覆盖率与
5 cm表面F1分轴呈现；每个条件各方法覆盖相同，联合指标差来自表面F1。
全部结果均为既有42付费动作预算（含18动作共同前缀）下的保存评分。
仅涉及2个已见开发布局／4个配对条件，不是独立重复；回放不增加样本量。
人工类别、精确模拟位姿、固定公共ROI和公共安全图仍为适用条件，不作显著性或未知场景泛化宣称。

## Source and scope

- A: `audit_results/v36_online_confirmation_20260918/result.json`, complete 8-row G/S matrix.
- B/C: `docs/thesis/figures/v39_external/measurements.csv`, complete 16-row four-method matrix.
- Means: existing `method_means.csv`, checked against all four conditions.
- The saved V36 parity audit reports exact measurements/actions/arrays; this
  plotting run rechecks saved metric equality and the referenced JSON/CSV hashes,
  not raw sensor arrays or the entire original experiment inventory.
- V35 supplies separate development evidence about misleading-prior correction;
  it is not pooled into these V36/V39 scores. Its h1/2 cm X−Xnf negative result
  remains in `docs/research/V35_ONLINE_SEMANTIC_RESULT_20260918.md` and must remain
  visible wherever feedback robustness is discussed. This figure itself is not
  a feedback ablation.
- Saved 2/10 cm secondary-threshold P/R/F/J values are retained in the exported
  full source CSVs. This main figure uses the previously specified 5 cm metric.
- No new World, sensor packet, controller/planner call, TSDF integration,
  surface evaluation, physical run, ROI selection or case selection.

Reproduce from the repository root with
`python3 scripts/plot_virtual_paper_evidence_20260928.py --output NEW_DIRECTORY`.
Existing output directories are never overwritten. `manifest.json` records
SHA256 for the inputs actually checked and every output except itself.
