# 同光心名义曝光去重：局部修正与离线依据

2026-09-28。这里只定义新的共同规划代理及纯解析验证；未运行新策略、World、TSDF或表面质量评价。旧62份冻结源码和所有原终点保持原样。新消融是否执行及其预算由另行冻结协议决定。

## 问题与已保存的证据

原预测器将模板自可见采样点中、距实例实测支持超过8 cm的部分计为新增面积。历史相机只在内参、位置、朝向和图像形状全部相同时使整视角收益归零，没有扣除相邻朝向的重叠曝光。

在已完成的Ground AISLE G/B路径第103步，柜体的planar结构后验分别为0.994779/0.996262。正常尺度模板的新增面积为0，但0.85倍和1.15倍模板分别仍有1.282442与2.702165平方米“未见”面积；均匀三尺度平均得到1.328202平方米。此时实测支持3592点，未达到4096容量，故该例不能归因于缓存饱和。名义尺度偏差经过8 cm点距规则持续产生正收益，而改变yaw又绕过完全相同相机的重复规则。

这解释的是名义代理偏差。原路径中的实际支持和TSDF仍可能增长，不能把重复名义曝光写成真实三维观测完全无收益。

## 唯一修改

对固定模板采样点\(p\)，令\(V(p,v)\)为原实现的视锥、轴向近远裁剪、正面和模板自遮挡共同可见性，\(S_{i,t}\)为原实测支持。定义

\[
E_t(p,o)=\bigvee_{j\le t:\;o_j=o} V(p,v_j),
\qquad
A'_{ih}(v)=\sum_p w_p V(p,v)
\mathbf1[d(p,S_{i,t})>0.08]\,[1-E_t(p,o_v)].
\]

历史仅取已经取得的相机，包括原合同中的初始帧0。光心是完整三维平移向量，绝对容差1e−9 m，要求内参及图像形状一致；角度不要求一致。当前公开接口仍固定轴向范围0.1—4 m。对当前有效标签平面重新投影历史相机，保持同一模板采样坐标，不因旧缓存中的平面不同产生重复积分。

原后验、尺度平均、观测支持、已观测面积和控制器代价不变，只将已经尝试过的同光心名义曝光从候选资格中扣除。新朝向中原视锥外、原有效轴向范围外的区域保留；平移到不同光心的候选不受此规则扣减。所有G/B/S/NBV须共同使用该预测器。它不是“禁止同XY转向”，也不是把模板点写成已重建表面。

没有同时改尺度后验、外部遮挡模型、初始化、类别权重、诊断通道或观察次数。yaw改变像素采样、噪声和缺失深度恢复的潜在收益不在当前名义新增面积模型中，因此去重仍是任务代理，不保证最终质量提高。

## 离线复现与结果

正式脚本：[diagnose_article_exposure_overlap_20260928.py](/root/NSO/scripts/diagnose_article_exposure_overlap_20260928.py)。读取两条已完成且独立复核合格的Ground AISLE G/B记录，固定检查第27、60、100、103、105、107、113、128、130步。每条任务659个原artifact文件及62份冻结源码全部校验；18个保存预测按1e−12 m²绝对容差复现。只重算这些候选的名义面积，不重新运行候选选择或生成新路径。

| G保存步数 | 原名义新增面积/m² | 去重后/m² | 同光心历史覆盖占比 |
|---:|---:|---:|---:|
| 27 | 1.714704 | 1.569120 | 8.49% |
| 60 | 1.893480 | 1.496581 | 20.96% |
| 100 | 1.349227 | 1.349227 | 0.00% |
| 103 | 1.328245 | 0.000000 | 100.00% |
| 105 | 0.763108 | 0.000000 | 100.00% |
| 107 | 0.784606 | 0.002112 | 99.73% |
| 113 | 0.953501 | 0.527965 | 44.63% |
| 128 | 0.722136 | 0.033382 | 95.38% |
| 130 | 0.012473 | 0.012473 | 0.00% |

这九个时刻用于诊断已出现的重复收益，属于问题定位样本，不是全部重规划的无偏抽样。不同后验导致G/B期望数值略有差别，全部18行与12个尺度构型分量保存在[正式诊断包](/root/NSO/audit_results/article_stage_20260928/analysis_v1/same_center_exposure_v3_final/diagnosis.json)、[CSV](/root/NSO/audit_results/article_stage_20260928/analysis_v1/same_center_exposure_v3_final/candidate_overlap.csv)和[输入源/输出清单](/root/NSO/audit_results/article_stage_20260928/analysis_v1/same_center_exposure_v3_final/manifest.json)。包内保存新增执行源与测试源的字节副本，总输出约214 KiB。较早的`same_center_exposure_v3_diagnosis`目录是开发中只读计算，正式引用使用带源副本的`same_center_exposure_v3_final`。

## 接口与验证

新类为[ArticleExposureViewPredictorV3](/root/NSO/nso/view_quality_article_exposure_v3.py)，继承原`ArticleViewPredictorV1`。新controller子类应在GroundV2构造完成后、首次观测前使用同一个残差对象替换预测器：

```python
self._predictor = ArticleExposureViewPredictorV3(
    residual=self._residual,
    maximum_instances=self._configuration['maximum_instances'])
```

新controller和runner须另声明版本、记录预测器配置摘要并冻结源码；本次没有实现runner，也不更改旧全局类绑定。原`forecast`签名及`structure_new_surface_area_m2`保留，后者为去重后的规划面积。新增分量字段`new_surface_area_before_exposure_m2`和`excluded_previously_exposed_unseen_area_m2`明确保留去重前和移除面积；继承的`unexcluded_expected_new_surface_area_m2`仍指原完全相同相机整视角屏蔽之前，已经位于本次曝光去重之后，不能误当成新修正前的值。

[11项纯解析测试](/root/NSO/tests/test_view_quality_article_exposure_v3.py)通过，覆盖不同yaw完全重叠为零、新FOV保留、转向进入有效轴向范围保留、不同光心不扣减、无历史与原值一致、重复帧/历史顺序不改变union、模板自遮挡、K/形状不兼容、光心容差、面积守恒以及不确定关联/无平面回退。有效分量满足“原新增=修正后新增+重复曝光排除”，原实测支持、可见面积、已观测面积和原始传感包不变。物理计数器在测试前后相同。

新预测器SHA256：`0843b90ddf7c71a01a16cd629d355ffd272a653049a4d0cd32d17a75a86126cf`。该版本只修共同规划代理；新S/B收益、未知布局性能和真实小车效果均不能从本次离线计算推出。
