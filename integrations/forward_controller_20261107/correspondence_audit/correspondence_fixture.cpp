// Offline state fixture only. Link exact installed matcher conversion/CSM.
#include <laser_scan_matcher/laser_scan_matcher.h>
#include <ros/serialization.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <vector>

template<class T> T readMessage(const std::string& path) {
  std::ifstream stream(path,std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open "+path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});
  ros::serialization::IStream input(bytes.data(),bytes.size());
  T message; ros::serialization::deserialize(input,message); return message;
}
std::string poseJson(const tf::Transform& t) {
  std::ostringstream out; out<<std::setprecision(17)<<"["<<t.getOrigin().x()<<","<<t.getOrigin().y()<<","<<tf::getYaw(t.getRotation())<<"]";return out.str();
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"csm_state_fixture");
  if (argc<2) return 2;
  ros::NodeHandle nh("/fixture"),private_nh("/fixture/matcher");
  scan_tools::LaserScanMatcher matcher(nh,private_nh);
  tf::Transform base_laser; base_laser.setIdentity();
  base_laser.setOrigin(tf::Vector3(-.011,.0035,.451));
  matcher.base_from_laser_=base_laser;
  matcher.laser_from_base_=base_laser.inverse();
  std::ifstream jobs(argv[1]);std::string line;
  while(std::getline(jobs,line)) {
    if(line.empty())continue;
    int index;std::string ref_path,target_path,key_pose_path,prev_pose_path,target_pose_path;
    std::istringstream job(line);job>>index>>ref_path>>target_path>>key_pose_path>>prev_pose_path>>target_pose_path;
    auto ref=boost::make_shared<sensor_msgs::LaserScan>(readMessage<sensor_msgs::LaserScan>(ref_path));
    auto target=boost::make_shared<sensor_msgs::LaserScan>(readMessage<sensor_msgs::LaserScan>(target_path));
    auto key_pose=readMessage<geometry_msgs::PoseStamped>(key_pose_path);
    auto prev_pose=readMessage<geometry_msgs::PoseStamped>(prev_pose_path);
    auto target_pose=readMessage<geometry_msgs::PoseStamped>(target_pose_path);
    matcher.createCache(ref);
    LDP reference,current;
    matcher.laserScanToLDP(ref,reference);matcher.laserScanToLDP(target,current);
    for(int i=0;i<3;++i)reference->odometry[i]=reference->estimate[i]=reference->true_pose[i]=0.;
    matcher.input_.laser_ref=reference;matcher.input_.laser_sens=current;
    tf::poseMsgToTF(key_pose.pose,matcher.keyframe_base_in_fixed_);
    tf::poseMsgToTF(prev_pose.pose,matcher.last_base_in_fixed_);
    matcher.last_icp_time_=prev_pose.header.stamp;
    const tf::Transform pred_last_base_offset=matcher.getPrediction(target->header.stamp);
    const tf::Transform pred_base_in_fixed=matcher.last_base_in_fixed_*pred_last_base_offset;
    const tf::Transform pred_keyframe_base_offset=matcher.keyframe_base_in_fixed_.inverse()*pred_base_in_fixed;
    const tf::Transform first_guess=matcher.laser_from_base_*pred_keyframe_base_offset*matcher.base_from_laser_;
    matcher.input_.first_guess[0]=first_guess.getOrigin().x();
    matcher.input_.first_guess[1]=first_guess.getOrigin().y();
    matcher.input_.first_guess[2]=tf::getYaw(first_guess.getRotation());
    if(matcher.output_.cov_x_m){gsl_matrix_free(matcher.output_.cov_x_m);matcher.output_.cov_x_m=0;}
    if(matcher.output_.dx_dy1_m){gsl_matrix_free(matcher.output_.dx_dy1_m);matcher.output_.dx_dy1_m=0;}
    if(matcher.output_.dx_dy2_m){gsl_matrix_free(matcher.output_.dx_dy2_m);matcher.output_.dx_dy2_m=0;}
    sm_icp(&matcher.input_,&matcher.output_);
    // Inspect final retained associations only; no rematching or policy changes.
    for (int i=0;i<current->nrays;++i) {
      const auto& c=current->corr[i];
      if (!c.valid || c.j1<0 || c.j2<0 || c.j1>=reference->nrays || c.j2>=reference->nrays) continue;
      const auto& a=reference->points[c.j1]; const auto& b=reference->points[c.j2]; const auto& p=current->points[i];
      const double cs=std::cos(matcher.output_.x[2]),sn=std::sin(matcher.output_.x[2]);
      const double x=cs*p.p[0]-sn*p.p[1]+matcher.output_.x[0],y=sn*p.p[0]+cs*p.p[1]+matcher.output_.x[1];
      const double dx=b.p[0]-a.p[0],dy=b.p[1]-a.p[1],len=std::hypot(dx,dy);
      if (!(len>1e-12)) continue;
      const double residual=((x-a.p[0])*(-dy)+(y-a.p[1])*dx)/len;
      std::cerr<<std::setprecision(17)<<"CORR "<<index<<" "<<i<<" "<<c.j1<<" "<<c.j2<<" "<<c.type<<" "<<current->theta[i]<<" "<<current->readings[i]<<" "<<reference->readings[c.j1]<<" "<<residual<<" "<<std::hypot(x-a.p[0],y-a.p[1])<<" "<<x<<" "<<y<<" "<<-dy/len<<" "<<dx/len<<"\n";
    }
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"reference_index\":"<<index<<",\"reference_stamp_ns\":"<<ref->header.stamp.toNSec()
       <<",\"previous_stamp_ns\":"<<prev_pose.header.stamp.toNSec()<<",\"target_stamp_ns\":"<<target->header.stamp.toNSec()
       <<",\"actual_keyframe_pose\":"<<poseJson(matcher.keyframe_base_in_fixed_)
       <<",\"actual_previous_pose\":"<<poseJson(matcher.last_base_in_fixed_)
       <<",\"first_guess\":"<<poseJson(first_guess)<<",\"valid\":"<<(matcher.output_.valid?"true":"false");
    if(matcher.output_.valid) {
      tf::Transform laser_offset;matcher.createTfFromXYTheta(matcher.output_.x[0],matcher.output_.x[1],matcher.output_.x[2],laser_offset);
      const tf::Transform base_offset=matcher.base_from_laser_*laser_offset*matcher.laser_from_base_;
      const tf::Transform result=matcher.keyframe_base_in_fixed_*base_offset;
      tf::Transform actual;tf::poseMsgToTF(target_pose.pose,actual);
      const double jump=std::hypot(result.getOrigin().x()-prev_pose.pose.position.x,result.getOrigin().y()-prev_pose.pose.position.y);
      out<<",\"error\":"<<matcher.output_.error<<",\"nvalid\":"<<matcher.output_.nvalid<<",\"iterations\":"<<matcher.output_.iterations
         <<",\"laser_offset\":"<<poseJson(laser_offset)<<",\"result_pose\":"<<poseJson(result)<<",\"actual_target_pose\":"<<poseJson(actual)
         <<",\"planar_jump_m\":"<<jump<<",\"actual_target_xy_error_m\":"<<std::hypot(result.getOrigin().x()-actual.getOrigin().x(),result.getOrigin().y()-actual.getOrigin().y())
         <<",\"actual_target_yaw_error_rad\":"<<std::fabs(std::atan2(std::sin(tf::getYaw(result.getRotation())-tf::getYaw(actual.getRotation())),std::cos(tf::getYaw(result.getRotation())-tf::getYaw(actual.getRotation()))))
         <<",\"would_switch_keyframe\":"<<(matcher.newKeyframeNeeded(base_offset)?"true":"false");
    }
    out<<"}";std::cout<<out.str()<<std::endl;
    ld_free(reference);ld_free(current);
  }
  return 0;
}
