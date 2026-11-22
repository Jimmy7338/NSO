# 区域全局选点的覆盖项修订

上一轮长走廊的终点覆盖下降，环路的覆盖与重建综合指标输给严格前沿。这里先复算原来实际决策时保存的候选与评分，再只修改区域层的一项面积计算。每个新回合由封存首帧独立开始。已见长走廊和环路只用于开发诊断，与两栋由新种子、几何和首帧生成的验证楼层分开报告。

修订 S 将区域评分的 `min(可见未知格数 × 格面积, 2 m²)` 换为完整的预计面积。保留其他区域项、平局规则和原生候选可行性。原区域与修订共用类别 S 局部策略与相同运动执行。严格前沿使用原先单独的候选策略和共用建图器、足迹保护、返航锚点。预计未知面积以当前占据图为依据，遮挡尚不可知时可能高估。精确仿真位姿和已观测实例的理想类别在三组保持一致。

- [原布局的固定状态诊断](diagnosis/README.md)
- [独立楼层与源码说明](scenes/SCENES.md)
- [冻结实验设计](PROTOCOL.json)
- [开发与未见楼层结果](results/RESULTS.md)

## 复跑

原生运行器及评价来自上一轮区域正式试验。保留该试验的目录与封存源码。新场景的生成与独立参考见场景说明。若在新目录重跑，应先按封存清单检查原目录引用，并保留已有实测原件。原始测量网格保留在中控，归档记录每份网格的 SHA-256。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" study_revision.py seal development
"$NSO_PY" study_revision.py run development --workers 2
"$NSO_PY" evaluate_revision.py development
"$NSO_PY" study_revision.py seal formal
"$NSO_PY" bind_fixture.py
"$NSO_PY" study_revision.py run formal --workers 6
"$NSO_PY" evaluate_revision.py formal
"$NSO_PY" report.py
```

独立场景的继承渲染源码在第一次封存时漏列，正式演员启动前补入哈希。首帧、模拟器缓存、策略和评价数据没有重建。旧封存和修改回执随证据保留。
