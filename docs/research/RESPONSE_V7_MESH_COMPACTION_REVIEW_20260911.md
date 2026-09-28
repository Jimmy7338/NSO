# V7网格清理前审查

2026-09-11。仅审查与非修改性验证，没有对原run调用compact，没有删除任何原始证据，也没有查看候选收益值。审查版本SHA256：`2a0deb9b7fe87c9c72c3d3110da9c557520d3fb32b5db3000a46f8c8353e4e71`。详细记录及所审源码副本在[audit_results](/root/NSO/audit_results/response_v7_mesh_compaction_review_20260911/summary.json)。

**结论：该版本的固定删除范围和删除前门禁通过本次有界审查。** 每run只允许两family的candidate1..5、arrival/final网格，共20文件；原始RGB-D、雷达、参考、前缀及candidate0保留。失败run、部分回放、策略不符或输入hash异常不能通过删除前门槛。它不自动授权缺乏完整回放的其它run。

首次静态审查发现源码归档绑定只在删除后验证，以及Python -O会禁用assert。根代理已在删除前加入verification/source归档及源码清单绑定，并在模块入口拒绝-O。本次检查针对修正版。

T0的20个原网格逐个load后用np.savez_compressed写入BytesIO：全部重编码NPZ的整个文件SHA256与原文件及原manifest一致，合计44,753,500字节（约42.68MiB）。原文件内容、mtime及大小未改变；数组确为vertices/triangles，没有colors。这证明在当前NumPy/zlib环境下，已知数组可重新编码出完全相同文件，**尚不单独证明raw重新融合会生成相同数组**。下述独立恢复检查已验证后一环节。

16项临时小fixture检查全部通过：未完成／失败run、失败／partial／截断回放、11/12分支、未复查raw、artifact/source归档/source清单不符、策略不符、错误网格schema及缺raw均在unlink前拒绝；有效fixture的第一unlink被审查代码拦截，确认20份dtype/shape/数组SHA已先写入planned，planned状态被验证器拒收；-O入口拒绝。所有测试拦截了删除调用，不触碰原run。

原artifact manifest保持不变，补充compaction_manifest记录固定20网格的文件SHA及数组dtype/shape/字节SHA。验证器只允许该集合的已见证缺失，其余原资产仍须逐项存在且hash匹配。执行中的planned／partial不会冒称complete；失败时保留记录，但自动续传／恢复尚需独立工具。JSON采用临时文件替换实现原子更新，这没有声明断电持久化保证。

旧严格回放脚本在网格缺失后会拒绝原manifest，不能把清理前的passed_full称为清理后重新回放。应先根据留存raw和封存mapper恢复网格，匹配数组及原NPZ hash后再回放。另需注意：旧replay覆盖verification.json后会改变文件SHA，而当前compaction验证绑定原verification SHA；若要反复恢复与回放，必须保留原验证见证或让完整原资产存在时直接验证原manifest。此相容性问题已报告根代理，本审查没有修改清理工具。

当前NPZ编码复现依赖已核对的软件环境，不能无条件推广到任意NumPy/zlib版本。所有原raw仍保留；新的压缩见证不是删除唯一观察证据的依据。各run能节省多少空间应以各自实际文件账本为准，不能仅用T0乘8代替实际统计。

## 原始观测恢复验证（已完成）

新增[恢复工具](/root/NSO/scripts/restore_response_v7_meshes.py)，实测版本SHA256为`f7d6614fc7525776a59a9d79231205f091d4623ec46f6fc33a5738281ce0c200`。T0在已清理20网格后执行check-only：隔离子进程从run的sources.zip导入归档mapper，按各支路重新融合完整prefix与逐动作RGB-D/scan，在arrival/final重建全部20网格。dtype、shape、数组SHA与compaction见证逐项一致，重编码整个NPZ的SHA也与原artifact manifest完全一致。共重建44,753,500字节，耗时23.53秒，单worker。

[恢复核对记录](/root/NSO/audit_results/response_v7_mesh_compaction_review_20260911/raw_restore_check.json)显示`original_files_written=[]`；验证前后仍为20个已归档缺失网格、945项原manifest记录，T0目录没有写回网格，compaction_manifest hash保持不变。所有临时输出随检查结束清理。恢复进程没有实例化世界，也没有解释GT或候选收益；外围完整性验证器对原资产（包括参考和结果文件）做hash扫描，不能将该扫描误写成读取效果数值。

非check-only模式会先在临时目录验证全部20文件，匹配后才把缺失文件逐个以同目录临时文件、fsync和原子替换写回；已有网格若存在必须匹配原SHA，不覆盖异常文件。历史compaction_manifest保留。此次只测试了check-only，没有在原run执行写回模式。未清理且原网格齐全时，代码使用原数组和原manifest核对；该分支未在本轮另做完整重跑。

根代理另修正清理验证器的恢复兼容分支（SHA256 `2a0deb9b7fe87c9c72c3d3110da9c557520d3fb32b5db3000a46f8c8353e4e71`）：仅当历史compaction状态complete且全部原manifest资产重新存在时，先逐文件全hash核验，成功即可按完整原manifest接受；planned/partial不能走该例外。这样后续旧replay更新verification.json不会使完整恢复的数据被旧见证hash误拒收。该兼容修改没有改变此前固定删除范围及删除前门禁；本次已静态核对，并在T0缺失20网格的正常归档分支上实际验证通过。

可复用的非修改性检查：

```bash
/root/NSO/.venv-3d/bin/python /root/NSO/scripts/restore_response_v7_meshes.py \
  --run /root/NSO/eval_results/response_v7_training_20260911/T0 \
  --check-only \
  --report /root/NSO/audit_results/response_v7_mesh_compaction_review_20260911/raw_restore_check.json
```

本次不重新评价规划收益，不扩大语义效果主张。只有T0进行了全20网格raw恢复测试，不能把它写成其它已清理run全部再次恢复通过。
