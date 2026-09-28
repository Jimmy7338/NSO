# 方案修订可行性检验：保留全部开发结果

共 144 个完整开发回合，使用同一批 6 张已见地图；不是独立正式测试。
所有回合（包括早停和失败原型）均保留；重复基线不增加独立样本数。
结论：尚未通过进入独立验证的开发筛选，不批准扩展训练或写正向结论。

| 轮次 | 场景组 | 条件 | AUC | 最终覆盖 |
|---|---|---|---:|---:|
| dev | generic | nearest | 0.44580 | 70.69% |
| dev | generic | geometry | 0.47188 | 73.24% |
| dev | generic | combined | 0.37887 | 69.76% |
| dev | generic | combined_no_timeout | 0.38072 | 69.76% |
| dev | generic | geometry_cost | 0.47327 | 76.81% |
| dev | generic | combined_cost | 0.37878 | 68.79% |
| dev | generic | safe_single | 0.40163 | 70.68% |
| dev | generic | safe_bundle | 0.42170 | 66.50% |
| dev | generic | path_bundle | 0.41637 | 65.54% |
| dev | targeted | nearest | 0.18074 | 32.57% |
| dev | targeted | geometry | 0.20298 | 38.50% |
| dev | targeted | combined | 0.14502 | 25.92% |
| dev | targeted | combined_no_timeout | 0.15593 | 32.13% |
| dev | targeted | geometry_cost | 0.20088 | 36.95% |
| dev | targeted | combined_cost | 0.15040 | 28.78% |
| dev | targeted | safe_single | 0.02063 | 2.08% |
| dev | targeted | safe_bundle | 0.02063 | 2.08% |
| dev | targeted | path_bundle | 0.02063 | 2.08% |
| projection_dev | generic | geometry | 0.47188 | 73.24% |
| projection_dev | generic | combined | 0.37887 | 69.76% |
| projection_dev | generic | metric_gain | 0.48042 | 75.76% |
| projection_dev | generic | metric_cost | 0.46828 | 74.07% |
| projection_dev | generic | projected_single | 0.46246 | 77.71% |
| projection_dev | generic | projected_bundle | 0.45408 | 74.49% |
| projection_dev | generic | projected_path | 0.44772 | 73.34% |
| projection_dev | targeted | geometry | 0.20298 | 38.50% |
| projection_dev | targeted | combined | 0.14502 | 25.92% |
| projection_dev | targeted | metric_gain | 0.19885 | 36.90% |
| projection_dev | targeted | metric_cost | 0.20197 | 38.56% |
| projection_dev | targeted | projected_single | 0.20305 | 39.05% |
| projection_dev | targeted | projected_bundle | 0.18053 | 35.09% |
| projection_dev | targeted | projected_path | 0.18350 | 35.69% |
| route_dev | generic | geometry | 0.47188 | 73.24% |
| route_dev | generic | nearest | 0.44580 | 70.69% |
| route_dev | generic | geometry_cost | 0.47327 | 76.81% |
| route_dev | generic | route32 | 0.40668 | 54.14% |
| route_dev | generic | route64 | 0.44198 | 65.30% |
| route_dev | targeted | geometry | 0.20298 | 38.50% |
| route_dev | targeted | nearest | 0.18074 | 32.57% |
| route_dev | targeted | geometry_cost | 0.20088 | 36.95% |
| route_dev | targeted | route32 | 0.20582 | 30.61% |
| route_dev | targeted | route64 | 0.18311 | 33.16% |
| transfer_dev | generic | geometry | 0.47188 | 73.24% |
| transfer_dev | generic | route32 | 0.47216 | 75.83% |
| transfer_dev | generic | route64 | 0.44764 | 69.30% |
| transfer_dev | targeted | geometry | 0.20298 | 38.50% |
| transfer_dev | targeted | route32 | 0.21589 | 36.60% |
| transfer_dev | targeted | route64 | 0.18311 | 33.16% |

## 如何解释

- 第一轮检验代价、固定时限和安全观测；不支持只修这些环节便可使原组合有效。
- 第二轮修复新原型的候选抽样错误；恢复了合理行为，但收益不稳定。
- 第三轮加入朝向状态最短路及两个视点的边际覆盖；短时域误作可达性边界导致早停。
- 第四轮加入长距离转移回退；这是修复第三轮自己的缺陷，不能当作对旧系统的独立优势。
- 新原型仍未实现完整的区域访问序列，不能以这些结果断言完整分层方案已有效或必然无效。
- 本轮没有增加语义训练；原语义／结构组合的负证据继续有效。

## 复现

每个运行目录含原始配置、逐动作观测、候选日志、源码压缩包及哈希。
旧开发版本应从相应 sources.zip 解压到单独目录后运行归档配置；当前代码是第四轮版本。
最终版本可使用 configs/cpu/coverage_v2_transfer_dev.json 复现，输出目录必须不存在。
完整方案、文献与验收条件见 docs/PLANNER_REDESIGN_AND_FEASIBILITY.md。
