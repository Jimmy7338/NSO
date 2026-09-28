# 平台配置与简明复现说明

**发布迁移说明：** 独立项目清理移除了退役训练适配及部分历史源码归档。下面的版本、源码摘要和路径描述原实验执行状态；当前工作树不再满足所有旧源码闭包的逐字节校验。现用CPU算法、传感数据和结果保留，新采集须重新封存，完整旧源码恢复范围见[迁移记录](../research/STANDALONE_RELEASE_20260928.md)。

## 1 实验执行环境与文稿构建环境

当前论文使用的主比较、纠错和CPU机制适配分别来自已经保存的V36、V35和V39批次。三批的采集准备清单均记录Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4和Open3D 0.19.0。科学运行使用CPU路径，并在Python启动前将OpenMP、OpenBLAS和MKL线程数分别设为1；GPU推理不是当前控制器与TSDF管线的运行条件。

**表A1 科学执行与文稿构建的环境记录。**

| 项目 | 已归档实验执行记录 | 当前文稿构建工作区（2026-09-28读取） |
|---|---|---|
| CPU型号、核数 | 三批manifest未记录具体CPU型号和核数，不据此解释绝对耗时的跨平台优劣 | Intel Xeon Processor（SapphireRapids），可见4个逻辑CPU |
| 操作系统 | 三批manifest未独立记录发行版 | Ubuntu 24.04.2 LTS，x86_64，Linux 6.8.0-63-generic |
| Python | 3.12.3，记录编译器GCC 13.3.0 | 系统Python 3.12.3；排版入口使用`python3 -B` |
| 科学依赖 | NumPy 1.26.4、SciPy 1.11.4、Open3D 0.19.0 | 不以当前安装状态替代历史manifest |
| 科学线程设置 | OMP、OpenBLAS、MKL均为1 | 排版与预览不参与科学耗时统计 |
| 排版与预览 | 与科学实验无关 | Tectonic 0.17.0；PDF预览pypdfium2 4.30.0、Pillow 12.3.0 |

历史软件证据分别见[V35 manifest](/root/NSO/audit_results/v35_online_development_20260918/manifest.json)、[V36 manifest](/root/NSO/audit_results/v36_online_confirmation_20260918/manifest.json)与[V39 manifest](/root/NSO/audit_results/v39_external_cpu_20260920/manifest.json)。线程约束由三个采集入口检查，不以“CPU环境”等同于多核并行基准。当前CPU和系统信息来自本次只读的`/proc/cpuinfo`、`/etc/os-release`与Python平台接口；它们仅描述当前工作区，不能追溯证明历史任务执行在同一硬件上。排版版本见[环境复制记录](/root/NSO/docs/thesis/GRADUATION_ENVIRONMENT_COPY_20260928.json)。

## 2 固定实验配置与数据入口

场景输入为[双构型场景配置](/root/NSO/configs/virtual3d/v33_direction_scene_r1_20260917.json)。主实验使用P00、P01，每个布局包含h0、h1；每条件采用共同18步前缀及42步总预算。类别仅通过受控RGB输入，安全图、设施坐标系与双模板向对照共同开放。虚拟相机为96×72、焦距48 px、量程4 m；扫描为180束、360°、量程8 m。TSDF体素4 cm、截断12 cm。深度使用参考视差域数值噪声，位姿准确已知，相关尺度不作为真实ZED标定结果。

开发批次的基础种子为1901，确认批次为350918；随机源按父布局及付费步派生，类别不参与。0.9类别先验、0.99预测诊断、残差截断和缓存限制见方法章的参数表与[配置常量](/root/NSO/nso/observation_belief_v35.py)。评价在轨迹与预测地图保存后进行，主阈值5 cm，并保留2/10 cm辅助阈值；资格另检查覆盖、预算、碰撞和精确返航。

| 内容 | 保存入口 | 用途 |
|---|---|---|
| G/S主比较 | [V36批次](/root/NSO/audit_results/v36_online_confirmation_20260918) | 读取4个条件、8条主任务及完整回放，核对条件配对 |
| 错误先验与整体纠错政策 | [V35批次](/root/NSO/audit_results/v35_online_development_20260918) | 读取G/S/X/Xnf及t18、t28行为记录 |
| 共同CPU机制 | [V39批次](/root/NSO/audit_results/v39_external_cpu_20260920) | SWAP-I与VISTA-I机制适配；G/S复用比较不增加独立样本 |
| 科学依赖 | [requirements-3d.lock.txt](/root/NSO/requirements-3d.lock.txt) | 完整环境清单；关键版本以对应批次manifest核对 |
| 确认执行协议 | [V36协议](/root/NSO/docs/research/V36_ONLINE_CONFIRMATION_PROTOCOL_20260918.md) | 矩阵、种子、线程、采集与评价次序 |

原采集入口为`scripts/run_online_routes_v35.py`、`scripts/run_online_routes_v36.py`和`scripts/run_external_routes_v39.py`。批次采用不可覆盖的输出与封存规则，已经完成的目录不可原地追加或再次执行`prepare`。重新采集应从对应归档代码、配置与协议建立独立的新输出批次，保留旧证据；论文图文复核可以直接使用现有数据完成。

## 3 保存证据的最小复核

下列命令在仓库根目录执行。独立审计器只读已有包、控制器和地图，核验配对、字段哈希与指标算术，并将新报告写入未使用目录；不会启动新World、规划器、TSDF或表面距离评分。`tmp/thesis-review-v36`必须尚不存在，重复复核时更换为另一个新目录。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-3d/bin/python -B scripts/verify_online_confirmation_v36.py \
  --batch audit_results/v36_online_confirmation_20260918 \
  --output tmp/thesis-review-v36
```

已有审计报告位于[audit_results/v36_online_confirmation_review_20260918/result.json](/root/NSO/audit_results/v36_online_confirmation_review_20260918/result.json)。读者可先核查其输入绑定和逐条件结果，再决定是否重建运行环境。重新运行审计不增加实验任务样本，也不替代原完整新进程回放。

已有论文图的重新绘制可使用保存CSV及其SHA256绑定，无需重复采集：

```bash
.venv-3d/bin/python -B scripts/plot_virtual_paper_method_20260928.py \
  --output tmp/thesis-figures-review
python3 -B scripts/build_writing_deliverables_20260928.py thesis --render --export
```

第一条命令的输出目录同样必须未使用；第二条命令从分章Markdown和当前图件构建全文PDF、TeX及记录。排版所需缓存、字体和预览包由交付指南说明，和科学依赖分开管理。以上命令在本说明编辑时仅核对入口与参数，未再次执行科学审计或采集。

## 4 本轮新增预算与几何任务

上述历史入口之外，9月28日另执行了固定128项扩展，96项完成合格、32项为低预算规划可行性失败，详见[整批结果](/root/NSO/docs/research/THESIS_EXPANSION_RESULT_20260928.md)。配置、43份执行源摘要和6项输入由新批次的`source_freeze.json`绑定，原保护代码及历史批次保持不变。当前任务的实际软件、CPU与线程设置另存于[扩展环境记录](/root/NSO/audit_results/thesis_expansion_20260928/analysis_review/environment.json)。

论文应读取`analysis_review`下的完整端点、配对和分层统计，使用`qualified`同时检查执行状态及评价资格。原采集汇总中的`independent_layout_count`只表示基础布局计数，已在审阅汇总明确为`distinct_base_parent_count`；不得据该旧字段推断统计独立。32项失败部分地图由已存包离线重融合，记录为`partial_failure_endpoint`，没有新增动作或规划，也不纳入完成性能。

扩展运行入口为`scripts/run_thesis_expansion_20260928.py`，审阅入口为`scripts/summarize_thesis_expansion_20260928.py`。已封存目录采用不覆盖规则，不在原目录再次启动；图件可由`scripts/plot_thesis_expansion_20260928.py`读取正式汇总生成。正常文稿构建仍使用本地缓存，本次仅为新增公式补齐一个Tectonic数学字体，未修改科学依赖。
