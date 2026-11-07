from pathlib import Path
import argparse, difflib
parser=argparse.ArgumentParser(description='Generate isolated candidate; never modifies source tree')
parser.add_argument('--source', type=Path, required=True, help='base_local_planner source directory')
parser.add_argument('--output', type=Path, required=True, help='separate writable output directory')
args=parser.parse_args()
src=args.source.resolve();p=args.output.resolve()
if p == src or src in p.parents: raise ValueError('output must be outside source tree')
p.mkdir(parents=True, exist_ok=True)
h=(src/'include/base_local_planner/simple_trajectory_generator.h').read_text();s=(src/'src/simple_trajectory_generator.cpp').read_text();(p/'original.cpp').write_text(s)
(p/'checked_generator.h').write_text(h.replace('SimpleTrajectoryGenerator','CheckedTrajectoryGenerator').replace('SIMPLE_TRAJECTORY_GENERATOR_H_','CHECKED_TRAJECTORY_GENERATOR_H_'))
needle='  traj.resetPoints();'
guard='''  // Reject invalid intersections/targets; never clamp or replace measured vel.
  const double bound_eps = 1e-7;
  if (!pos.allFinite() || !vel.allFinite() || !sample_target_vel.allFinite()) return false;
  Eigen::Vector3f lower(limits_->min_vel_x, limits_->min_vel_y, -limits_->max_vel_theta);
  Eigen::Vector3f upper(limits_->max_vel_x, limits_->max_vel_y, limits_->max_vel_theta);
  if (use_dwa_) {
    if (!std::isfinite(sim_period_) || sim_period_ <= 0) return false;
    const Eigen::Vector3f delta = limits_->getAccLimits() * sim_period_;
    if (!delta.allFinite() || (delta.array() < 0).any()) return false;
    lower = lower.cwiseMax(vel - delta);
    upper = upper.cwiseMin(vel + delta);
  }
  if (!lower.allFinite() || !upper.allFinite()) return false;
  for (int axis = 0; axis < 3; ++axis) {
    if (lower[axis] > upper[axis] + bound_eps ||
        sample_target_vel[axis] < lower[axis] - bound_eps ||
        sample_target_vel[axis] > upper[axis] + bound_eps) return false;
  }
'''
assert needle in s;patched=s.replace(needle,needle+"\n"+guard);(p/'proposed_source.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),patched.splitlines(True),fromfile='a/base_local_planner/src/simple_trajectory_generator.cpp',tofile='b/base_local_planner/src/simple_trajectory_generator.cpp')));(p/'checked_generator.cpp').write_text(patched.replace('#include <base_local_planner/simple_trajectory_generator.h>','#include "checked_generator.h"').replace('SimpleTrajectoryGenerator','CheckedTrajectoryGenerator'))
