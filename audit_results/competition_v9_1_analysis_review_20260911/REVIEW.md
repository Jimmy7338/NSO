# V9.1 可变候选集适配审核

结论：executor、独立 replayer 和 analyzer 的可变候选数量适配通过静态检查与纯合成检查。未发现阻止当前固定 44 条路线执行的代码适配问题。此审核不替代独立结构放行与实际 raw 回放，也不证明任何真实效果。

- Q0、Q1、Q2：ID 0..5；Q3：ID 0..3。每个 parent 两个 arrangement 使用完全相同 pool，合计 44 条、8 个 history。
- 每条路线预算仍为 48 动作；三脚本内剩余 48 均属于每条路线预算或稀疏 AUC 窗口，没有把 44 分支数误当成路线预算。
- outcome 完整路径集合、score 长度、选择与 tie、oracle、metadata 和 verification 分支计数均改用本版本约定。
- 四个 parent 等权，各自两个 arrangement 等权；候选数量未成为统计权重。
- 预注册 progression 数值完全保留：20% 新面积、0.005 F1/C×F1、每个 parent 严格正、AUC 下限、S>X、M=G、类别换位、coverage 与 total-vs-rate 均未放宽。
- 执行前源码包包括分析器、helper 与独立 replayer；分析器核对自己的 SHA、helper SHA、verification reviewer SHA、结构审核 SHA。

36 个 progression 门的 46 项纯合成检查通过；另将真实 analyzer 的 family-selection 语句块用于 8 组纯合成 profile，验证 96 项“方法×评分族”选择和成本/ID tie。Q3 的 4 候选输入若带遗留 6 维评分向量，会明确拒绝。

本次未构造未来 world、未读取未来 branch outcome、未修改三主脚本。原 V9 的强制六角色结构失败仍需保留；V9.1 是前缀之后、未来结果之前单独冻结的版本。

## 精确源码

| 文件 | SHA-256 |
|---|---|
| `scripts/analyze_competition_v9_1.py` | `d0bb1bba81ee4b47b86dd92832dbfa358111c2ed2a9d254383ba3420926cad67` |
| `scripts/replay_competition_v9_1.py` | `1194c1abc3103ffdc3511404ffcf9993b3ed0b33849b1d7029a4ce5c2529b568` |
| `scripts/eval_competition_v9_1.py` | `c04cd98418a7337c9c3e7aa1c475fbd4f75216d1f1730c7e5549fa50dd4a0fda` |
| `scripts/analyze_competition_v8_1.py` | `457178f23d882ca684111d4a9bc47733351762d47a94acb7ea960dabf30c6ad9` |
| `configs/virtual3d/competition_v9_1_validation_protocol.json` | `8dba14b72f91033fc3ad48b8e23c6c5ca80886f985756ff792fbf9e6be000430` |

证据：`review.json`、`numeric_gate_checks.json`、`variable_pool_checks.json`。源码、协议、AST 提取块和检查入口均保存于本目录。
