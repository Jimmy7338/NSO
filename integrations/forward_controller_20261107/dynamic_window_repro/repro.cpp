#include <base_local_planner/simple_trajectory_generator.h>
#include <base_local_planner/velocity_iterator.h>
#include "checked_generator.h"
#include <iostream>
#include <cassert>
#include <limits>
using namespace base_local_planner;
struct Case {const char*name;float v,w;std::vector<Eigen::Vector3f> extra;bool generic;};
struct Count {int samples=0,valid=0,reverse=0,angular=0,unreachable=0;};
template<class G> struct Inspect: G {using G::sample_params_;using G::vel_;};
template<class G> Count run(Case c, bool fixed) {
 Inspect<G> g;LocalPlannerLimits lim(.05,.02,.05,0,0,0,.3,.05,.25,.25,.8,.25,.08,.08,true,.01,.03);
 g.setParameters(1.7,.025,.01,true,.1);Eigen::Vector3f pos(0,0,0),vel(c.v,0,c.w),goal(.2,.1,0),samples(c.generic?10:0,1,20);
 g.initialise(pos,vel,goal,&lim,samples,c.extra);Count out;out.samples=g.sample_params_.size();
 while(g.hasMoreTrajectories()){Trajectory t;if(!g.nextTrajectory(t))continue;out.valid++;if(t.xv_ < -1e-7)out.reverse++;if(fabs(t.thetav_)>.3000001)out.angular++;if(fabs(t.xv_-c.v)>.0250001||fabs(t.thetav_-c.w)>.0800001)out.unreachable++;}
 std::cout<<"{\"case\":\""<<c.name<<"\",\"fixed\":"<<(fixed?"true":"false")<<",\"samples\":"<<out.samples<<",\"valid\":"<<out.valid<<",\"reverse\":"<<out.reverse<<",\"theta_outside_limit\":"<<out.angular<<",\"outside_dynamic_reach\":"<<out.unreachable<<"}\n";
 if(fixed){assert(!out.reverse&&!out.angular&&!out.unreachable);Trajectory rejected;assert(!g.generateTrajectory(pos,vel,Eigen::Vector3f(-.02,0,.05),rejected));assert(rejected.cost_<0 && rejected.getPointsSize()==0);assert(g.vel_[0]==vel[0] && g.vel_[2]==vel[2]);}
 return out;
}
int main(){VelocityIterator it(0,-.015,10);int neg=0;for(;!it.isFinished();it++)if(it.getVelocity()<0)neg++;assert(neg==9);
 std::vector<Case> cases={{"normal_stationary",0,0,{},true},{"negative_v_inverted",-.04,.1,{},true},{"recorded_like_negative_v",-.2346,-.1576,{},true},{"positive_v_inverted",.1329,.0683,{},true},{"positive_theta_inverted",0,.45,{},true},{"negative_theta_inverted",0,-.45,{},true},{"theta_boundary_reachable",0,.38,{},true},{"extras_negative_v",0,0,{Eigen::Vector3f(-.02,0,.05)},false},{"extras_excess_theta",0,0,{Eigen::Vector3f(0,0,.5)},false},{"extras_unreachable_legal_limit",0,0,{Eigen::Vector3f(.05,0,.05)},false},{"extras_bypass_empty_window",-.04,.1,{Eigen::Vector3f(.02,0,.1)},false},{"extras_valid",0,0,{Eigen::Vector3f(.02,0,.05)},false}};
 for(auto c:cases){auto old=run<SimpleTrajectoryGenerator>(c,false),fixed=run<CheckedTrajectoryGenerator>(c,true);if(std::string(c.name)=="normal_stationary"||std::string(c.name)=="extras_valid")assert(old.valid==fixed.valid);if(std::string(c.name)=="negative_v_inverted")assert(old.reverse>0&&fixed.valid==0);if(std::string(c.name)=="positive_theta_inverted")assert(old.angular>0&&fixed.valid==0);}
}
