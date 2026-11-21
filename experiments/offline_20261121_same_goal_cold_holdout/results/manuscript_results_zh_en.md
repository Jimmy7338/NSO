# 论文结果段落

在两栋链式和两栋环式留出建筑上，每种策略从同一栋建筑的共同首帧启动，预算为 512 个计费动作，包含返航。扩池 G 和 S 使用相同的中性全局目标规则。在共同观测状态下，两臂评估相同的局部路径候选，仅表面收益的结构先验不同。仿真提供当前可见设施的理想类别标签。四种策略均在四栋建筑达到 100% 的参考可达自由空间覆盖率，全部返航且无模拟碰撞。因此，以实际停止时覆盖率与整场景 F1@5 cm 的乘积定义的主指标 \(J\)，其终点差距全部由 F1 决定。

按链式 13、环式 14、链式 15、环式 16 的顺序，扩池 S 相对 G 的终点 \(J\) 分别变化 +0.040、−0.016、0.000 和 +0.965 个百分点。仅环式 16 超过 +0.10 个百分点，未达到预注册要求的至少三栋。类别先验引起的实际动作及 RGB、深度、激光观测分岔分别出现在链式 13 的第 1 步、环式 14 的第 425 步和环式 16 的第 51 步。链式 15 的两条初始计划不同。两臂均在第 31 步后由原足迹检查触发重规划，随后选中同一路线，最终动作、传感记录与网格完全一致。

S 在链式 13 提前 17 步达到 95% 覆盖。达到 100% 覆盖时，链式 13 和环式 16 分别比 G 多用 14 步和 39 步，其余两栋持平。环式 16 是唯一超过门槛的建筑。该建筑在共同第 256 步的 \(J\) 差为 −1.012 个百分点，到第 320 步变为 +0.727 个百分点。S 的终点设施表面净匹配面积比 G 少 0.072 m²，取得的 F1 优势体现在整场景。

S 相对原 S 的终点 \(J\) 差分别为 +0.102、−1.865、−0.111 和 +1.870 个百分点，相对严格最近可达前沿的差分别为 +1.602、+0.104、−1.281 和 +2.930 个百分点。两项基线在四栋建筑均比扩池 S 更早达到全覆盖。严格前沿在 226–290 步依各自停止规则结束，扩池 S 和原 S 均运行 512 步。终点比较按各策略的实际停止时刻计算。建筑是独立比较单位，预注册门槛用于工程判断。

# Manuscript results

On four held-out buildings with two chain and two cycle layouts, all policies started with the same first observation within each building and had a budget of 512 charged actions, including return. Expanded-pool G and S used the same neutral global goal rule and evaluated the same local routes at common acquired states. They differed only in the structural prior used to score local surface gain. The simulator supplied ideal category labels for visible fixtures. All sixteen episodes reached 100% coverage of the reference reachable free space, returned to the start and incurred zero simulated collisions. Consequently, differences in the primary endpoint \(J\), the product of correct free-space coverage and whole-scene surface F1 at 5 cm at actual termination, came from F1.

For Chain 13, Cycle 14, Chain 15 and Cycle 16, respectively, terminal \(J\) under S changed by +0.040, −0.016, 0.000 and +0.965 percentage points relative to G. Only Cycle 16 exceeded +0.10 percentage points, below the preregistered requirement of at least three buildings. Differences in executed actions and RGB, depth and lidar observations first appeared at actions 1, 425 and 51 in Chain 13, Cycle 14 and Cycle 16, respectively. In Chain 15, the two initial plans differed. The original footprint check triggered replanning in both arms after action 31, both selected the same new route, and the executed trajectories, sensor records and final meshes were identical.

S reached 95% coverage 17 actions earlier in Chain 13. It took 14 more actions to reach full coverage in Chain 13 and 39 more in Cycle 16, with ties in the other two buildings. In Cycle 16, the only building exceeding the practical threshold, the paired \(J\) difference went from −1.012 percentage points at the common 256-action checkpoint to +0.727 at 320. The terminal net matched fixture surface area was 0.072 m² lower under S in that building. The measured F1 improvement was at the whole-scene level.

The terminal \(J\) differences against original S were +0.102, −1.865, −0.111 and +1.870 percentage points, and those against strict nearest reachable frontier were +1.602, +0.104, −1.281 and +2.930 percentage points. Both baselines reached full coverage earlier than expanded-pool S in all four buildings. The frontier episodes stopped after 226–290 actions under their native stopping rule, while both versions of S ran for 512. Endpoint comparisons use each policy's actual stopping time. The building was the independent unit, and the preregistered threshold was an engineering decision rule.
