# 论文结果段落

在对四栋留出建筑的事后空间核验中，我们沿用原有可评参考点、面积权重和 5 cm 点到三角面距离阈值。整场景召回由参考点上的匹配比例计算，精确率用每套原生网格的独立预测采样复算。四栋楼的终点正确自由空间覆盖率均为 100%，场景 F1 差值与原记录一致。类别策略 S 相对中性策略 G 的终点 (J) 差值依次为 +0.040、−0.016、0.000 和 +0.965 个百分点。仅环式 16 超过预设的 +0.10 个百分点，低于至少三栋的预注册判据。链式 15 的两套终点网格逐字节一致，参考点匹配差为零。

环式 16 在第 512 步的整场景召回由 97.886% 提高至 99.626%，精确率由 99.617% 提高至 99.792%，F1 由 98.744% 提高至 99.709%。S 独有匹配面积为 8.626 m²，G 独有面积为 0.286 m²，净差为 +8.340 m²，主要位于西北区（+7.583 m²）。在同一组整场景参考点内，其余建筑表面净增 8.404 m²，两个目标设施净减 0.065 m²。独立高密度设施参考复算得到净减 0.072 m²。两个面积分解使用各自的采样口径。

第 256、320 和 512 步的事后对照显示，环式 16 的整场景匹配净差依次为 −9.244、+6.919 和 +8.340 m²。第 256 至 320 步的反转主要对应东北区净差从 −14.936 变为 +0.572 m²。该区间内，S 在东北区记录了 31 个计费位姿，其中 21 个朝南。G 没有在该区记录计费位姿。西北区的优势主要在第 256 步前形成，此前 S 和 G 分别在西北记录了 79 和 42 个位姿。轨迹与匹配图支持区间和区域层面的对应关系。

# Manuscript results

In a post hoc spatial audit of the four held-out buildings, we used the original evaluable reference points, area weights, and 5 cm point-to-triangle distance threshold. We computed scene recall on the fixed reference points and recomputed precision from independent samples on each native mesh. Both policies achieved 100% correct free-space coverage at termination in all four buildings, and the recomputed scene F1 differences matched the archived values. The corresponding terminal differences in (J) between the category-conditioned policy S and the neutral policy G were +0.040, −0.016, 0.000, and +0.965 percentage points. Only Cycle 16 exceeded +0.10 percentage points, below the preregistered requirement of at least three buildings. The two terminal meshes in Chain 15 were byte-identical and gave the same pointwise matches.

At action 512 in Cycle 16, scene recall increased from 97.886% to 99.626%, precision from 99.617% to 99.792%, and F1 from 98.744% to 99.709%. The S-only and G-only matched reference areas were 8.626 and 0.286 m², yielding a net gain of 8.340 m². Most of the gain was in the northwestern quadrant (+7.583 m²). Within the whole-scene sample, the remaining architectural surfaces gained 8.404 m² and the target fixtures lost 0.065 m². A separate, denser reference sample of the target fixtures yielded a loss of 0.072 m². Each area decomposition uses the weights of its own reference sample.

The post hoc snapshots at actions 256, 320, and 512 show scene-level matched-area differences of −9.244, +6.919, and +8.340 m², respectively. The reversal between actions 256 and 320 coincided with the northeastern difference changing from −14.936 to +0.572 m². In that interval, S logged 31 charged poses in the northeastern quadrant, including 21 facing south. G logged no charged poses there. Most of the northwestern advantage was already present at action 256, by which point S and G had logged 79 and 42 poses there, respectively. The correspondence is at the level of regions and intervals.
