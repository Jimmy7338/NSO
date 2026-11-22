# 全局归属、沿途收益与局部巡检

| 工作 | 证据 |
|---|---|
| 原区域策略的实际派发 | [133 次派发与固定候选池四组复算](access_audit/README.md) |
| 全局评分修订 | [独立关闭或启用两项修订的同前缀分支与未见楼层](global_attribution/results/RESULTS.md) |
| 局部 S、直接巡检 D、D 加受支持隐面 | [六条去重候选路线和共同实测时钟](local_increment/RESULTS.md) |
| 继承的运行问题 | [东侧 Open3D 生命周期、接收走廊恢复和西侧返航诊断](local_technical/RESULTS.md) |

按照原已见楼层的实际场景记录，一次性更正门后区域归属与恢复原生沿途评分是独立的开发干预。独立楼层中两项均启用，比较对象为同一局部 S 下的原区域策略和最近可达前沿。全部估计在线读取当前实际取得的观测。源几何只用于仿真和离线评价。

## 复跑全局比较

在中控的原目录结构中保留上一轮已封存的执行记录。新场景的生成命令见 [场景说明](global_attribution/scenes/SCENES.md)。封存或运行需要写空的试验目录，已存在的收据与测量文件请先另行保留。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" access_audit/audit_dispatches.py --check
cd global_attribution
"$NSO_PY" verify_rank.py
"$NSO_PY" study_repair.py seal development
"$NSO_PY" study_repair.py run development --workers 4
"$NSO_PY" measure.py development
"$NSO_PY" study_repair.py seal formal
"$NSO_PY" study_repair.py run formal --workers 3
"$NSO_PY" measure.py formal
"$NSO_PY" report_repair.py
```

局部候选的准备、封存和执行见 [同目标分支的复跑说明](local_increment/README.md)。技术修复只作独立诊断，使用原有动作执行和返航规则。
