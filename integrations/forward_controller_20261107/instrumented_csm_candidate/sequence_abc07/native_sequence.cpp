// Finite offline recursive A/B/C comparison; exact installed scan conversion/ICP.
#include <laser_scan_matcher/laser_scan_matcher.h>
#include <ros/serialization.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <vector>
#include <memory>
#include <chrono>
template<class T>T load(const std::string&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error(p);std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});ros::serialization::IStream in(b.data(),b.size());T m;ros::serialization::deserialize(in,m);return m;}
std::string pose(const tf::Transform&t){std::ostringstream s;s<<std::setprecision(17)<<"["<<t.getOrigin().x()<<","<<t.getOrigin().y()<<","<<tf::getYaw(t.getRotation())<<"]";return s.str();}
struct State{std::string mode;std::unique_ptr<scan_tools::LaserScanMatcher> m;LDP ref;ros::Time ref_stamp;int ref_index=0;};
int main(int argc,char**argv){
 ros::init(argc,argv,"recursive_csm_fixture");if(argc<2)return 2;
 std::ifstream jobs(argv[1]);std::string scan0_path,anchor_path;jobs>>scan0_path>>anchor_path;
 auto first=boost::make_shared<sensor_msgs::LaserScan>(load<sensor_msgs::LaserScan>(scan0_path));
 auto anchor=load<geometry_msgs::PoseStamped>(anchor_path);tf::Transform initial;tf::poseMsgToTF(anchor.pose,initial);
 tf::Transform base_laser;base_laser.setIdentity();base_laser.setOrigin(tf::Vector3(-.011,.0035,.451));
 std::vector<State> states;
 const std::string modes=argc>2?argv[2]:"ABC";
 for(char mode_character:modes){const std::string mode(1,mode_character);
  State s;s.mode=mode;ros::NodeHandle nh(std::string("/sequence/")+mode),priv(std::string("/sequence/")+mode+"/matcher");
  s.m.reset(new scan_tools::LaserScanMatcher(nh,priv));s.m->base_from_laser_=base_laser;s.m->laser_from_base_=base_laser.inverse();
  s.m->last_base_in_fixed_=initial;s.m->keyframe_base_in_fixed_=initial;s.m->last_icp_time_=first->header.stamp;
  s.m->createCache(first);s.m->laserScanToLDP(first,s.ref);s.ref_stamp=first->header.stamp;states.push_back(std::move(s));
 }
 int index;std::string scan_path;double dx,dy,da;
 while(jobs>>index>>scan_path>>dx>>dy>>da){
  auto scan=boost::make_shared<sensor_msgs::LaserScan>(load<sensor_msgs::LaserScan>(scan_path));
  for(auto&s:states){auto&m=*s.m;LDP curr;m.laserScanToLDP(scan,curr);
   for(int j=0;j<3;++j)s.ref->odometry[j]=s.ref->estimate[j]=s.ref->true_pose[j]=0.;
   m.input_.laser_ref=s.ref;m.input_.laser_sens=curr;
   const tf::Transform before=m.last_base_in_fixed_;const tf::Transform key=m.keyframe_base_in_fixed_;
   tf::Transform increment=m.getPrediction(scan->header.stamp);if(s.mode=="C")m.createTfFromXYTheta(dx,dy,da,increment);
   const tf::Transform predicted=m.last_base_in_fixed_*increment;
   const tf::Transform keyoffset=m.keyframe_base_in_fixed_.inverse()*predicted;
   const tf::Transform guess=m.laser_from_base_*keyoffset*m.base_from_laser_;
   m.input_.first_guess[0]=guess.getOrigin().x();m.input_.first_guess[1]=guess.getOrigin().y();m.input_.first_guess[2]=tf::getYaw(guess.getRotation());
   if(m.output_.cov_x_m){gsl_matrix_free(m.output_.cov_x_m);m.output_.cov_x_m=0;}
   if(m.output_.dx_dy1_m){gsl_matrix_free(m.output_.dx_dy1_m);m.output_.dx_dy1_m=0;}
   if(m.output_.dx_dy2_m){gsl_matrix_free(m.output_.dx_dy2_m);m.output_.dx_dy2_m=0;}
   auto start=std::chrono::steady_clock::now();sm_icp(&m.input_,&m.output_);const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
   tf::Transform measured;measured.setIdentity();
   if(m.output_.valid){tf::Transform laser;m.createTfFromXYTheta(m.output_.x[0],m.output_.x[1],m.output_.x[2],laser);measured=m.base_from_laser_*laser*m.laser_from_base_;m.last_base_in_fixed_=m.keyframe_base_in_fixed_*measured;}
   const bool original_switch=m.newKeyframeNeeded(measured);
   const bool replace=original_switch;  // B uses native zero keyframe thresholds from ROS parameters.
   const double jump=std::hypot(m.last_base_in_fixed_.getOrigin().x()-before.getOrigin().x(),m.last_base_in_fixed_.getOrigin().y()-before.getOrigin().y());
   const double angle=std::fabs(std::atan2(std::sin(tf::getYaw(m.last_base_in_fixed_.getRotation())-tf::getYaw(before.getRotation())),std::cos(tf::getYaw(m.last_base_in_fixed_.getRotation())-tf::getYaw(before.getRotation()))));
   const double dt=(scan->header.stamp-m.last_icp_time_).toSec();
   std::ostringstream out;out<<std::setprecision(17)<<"{\"mode\":\""<<s.mode<<"\",\"scan_index\":"<<index<<",\"scan_stamp_ns\":"<<scan->header.stamp.toNSec()<<",\"reference_index\":"<<s.ref_index<<",\"reference_stamp_ns\":"<<s.ref_stamp.toNSec()<<",\"previous_stamp_ns\":"<<m.last_icp_time_.toNSec()<<",\"previous_pose\":"<<pose(before)<<",\"keyframe_pose\":"<<pose(key)<<",\"motion_increment\":"<<pose(increment)<<",\"first_guess\":"<<pose(guess)<<",\"valid\":"<<(m.output_.valid?"true":"false")<<",\"icp_microseconds\":"<<us;
   if(m.output_.valid)out<<",\"x\":["<<m.output_.x[0]<<","<<m.output_.x[1]<<","<<m.output_.x[2]<<"],\"error\":"<<m.output_.error<<",\"nvalid\":"<<m.output_.nvalid<<",\"iterations\":"<<m.output_.iterations;
   else out<<",\"x\":null,\"error\":null,\"nvalid\":null,\"iterations\":null";
   out<<",\"result_pose\":"<<pose(m.last_base_in_fixed_)<<",\"planar_jump_m\":"<<jump<<",\"yaw_jump_rad\":"<<angle<<",\"existing_distance_bound_m\":"<<(.02+.20*dt)<<",\"existing_angle_bound_rad\":"<<(.04+.60*dt)<<",\"exceeds_existing_jump_bound\":"<<((jump>.02+.20*dt||angle>.04+.60*dt)?"true":"false")<<",\"original_switch_decision\":"<<(original_switch?"true":"false")<<",\"reference_replaced\":"<<(replace?"true":"false")<<"}";std::cout<<out.str()<<std::endl;
   if(replace){ld_free(s.ref);s.ref=curr;m.keyframe_base_in_fixed_=m.last_base_in_fixed_;s.ref_stamp=scan->header.stamp;s.ref_index=index;}else ld_free(curr);
   m.last_icp_time_=scan->header.stamp;
  }
 }
 for(auto&s:states)ld_free(s.ref);
 return 0;
}
