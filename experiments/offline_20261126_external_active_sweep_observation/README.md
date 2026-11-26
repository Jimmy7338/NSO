# HM3D 出口地面补测与闭环复跑

在上一轮在线扫掠适配之上，为 HM3D 00770 起点 1 的指定出口 `[52,26]→[53,26]` 加入一次定向补测。HM3D 起点 1 的三种策略均可触发同一流程，另外三个起点沿用原流程。候选排序、计费、场景渲染、V2 模拟器和评价参考沿用上一轮。该试验验证指定出口的补测机制，结果的适用范围以本次场景和已知观测视角为限。[上一轮结果](../offline_20261126_external_online_sweep/FINDINGS.md)

机器人抵达出口起始格，且完整扫掠只因地面证据不足而失败时，才会尝试补测。从上一轮诊断登记的三个观测位姿中，优先选择 `[44,25]` 朝南。位姿必须在当时已观测的有向通行图内，去程及保留返程均须落在剩余预算内。到达后以真实执行产生的激光与 RGB-D 数据重新评估完整扫掠。规划器只能在该边当时获准且经过原运行授权时执行前进。若观测仍不足，可以尝试下一个登记视角。障碍回波、落差、外缘无回波及物理仿真限制继续生效。所有补测、转向和过边动作均计入 192 步预算。[实现](active_observation.py)

先以独立的 HM3D 起点 1 原区域策略回合验证实际换视角、完整扫掠放行、无碰撞穿过指定边及终点覆盖超过 `522/1,640`。通过后另启四起点、三策略的 12 轮正式矩阵。验证回合单独计录，同掩膜比较沿用上轮的 1,640 格参考。测量中的终点与固定时点各按真实计费步数记录。[验证冻结](GATE_PROTOCOL.json) · [独立核验](validation/GATE_AUDIT.json) · [正式冻结](FORMAL_PROTOCOL.json)

本轮已执行的主要命令如下。运行环境和原始数据来自上轮冻结，工作目录对应本次实验。

```sh
python study.py gate
python validation/audit_gate.py --run execution/gate/b26_hm3d_00770_s1/region_S/attempt_001/run --freeze GATE_FREEZE.json
python study.py seal
python study.py run --workers 6
python evaluate.py
python audit_formal_observation.py
```

[结果汇总](FINDINGS.md) · [同一评价掩膜逐步数据](results/formal/original_mask_matched_paid.csv)
