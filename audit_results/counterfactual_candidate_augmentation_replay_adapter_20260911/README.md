# 新增候选回放格式适配

首次标准回放失败：原核验器要求每个候选池编号为0到n−1，新增候选保留12起的编号，供原12条与新增条目合并。失败记录保存在initial_verification_failed_id_format.json，不参与效能结论。

新核验器只将编号检查改为：原候选文件与原封存hash一致；新增ID从原最大ID后连续编号；新增数量不超过预定4；新旧端点不重复。传感器生成、付费动作、地图／网格、覆盖／F1／新增面积、参考面、源码与产物hash核验均保持。完整差异见verifier.diff，两个版本hash见manifest.json。

适配后16条分支全部独立回放通过，6个历史及2个不可用记录核对通过，耗时61.33秒。最终记录见[verification.json](../../eval_results/counterfactual_candidate_augmentation_v6_20260911/verification.json)。该结果证明数据可复现，不证明语义或完整方案有效。
