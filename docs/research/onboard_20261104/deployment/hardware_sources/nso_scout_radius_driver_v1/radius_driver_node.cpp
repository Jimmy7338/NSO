// New isolated node: never starts the vendor ROS messenger or its /cmd_vel
// callback. Only the independent sender below invokes SetMotionCommand.
#include "scout_radius_command_timeout_v1.hpp"
#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <ugv_sdk/mobile_robot/scout_robot.hpp>
#include <csignal>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

namespace {
volatile std::sig_atomic_t interrupted=0;
void signal_handler(int) { interrupted=1; }
std::string boot_id() {
  std::ifstream file("/proc/sys/kernel/random/boot_id");
  std::string value; std::getline(file,value); return value;
}
bool valid_session(const std::string& value) {
  if (value.empty() || value.size()>128) return false;
  for (char c:value) if (!((c>='a' && c<='z') || (c>='A' && c<='Z') ||
      (c>='0' && c<='9') || c=='_' || c=='-')) return false;
  return true;
}
class RadiusNode {
 public:
  RadiusNode(ros::NodeHandle& node,ros::NodeHandle& params): execute_(false),captured_(0) {
    params.param("execute",execute_,false);
    params.param<std::string>("session_id",session_,"");
    params.param<std::string>("guard_node",guard_node_,"/nso_radius_guard");
    std::string expected_boot,interface;
    params.param<std::string>("expected_boot_id",expected_boot,"");
    params.param<std::string>("can_interface",interface,"can0");
    bool simulated_time=false; node.param("/use_sim_time",simulated_time,false);
    if (simulated_time || !valid_session(session_) || expected_boot.empty() ||
        boot_id()!=expected_boot || interface!="can0" || guard_node_.empty() || guard_node_[0]!='/')
      throw std::runtime_error("session/boot/CAN/ROS-time admission mismatch");
    if (execute_) {
      // This vehicle's saved protocol is AGX_V2, Scout Mini non-omni.
      // Use the actual SDK API's enum type (it is not in westonrobot).
      typedef decltype(std::declval<westonrobot::ScoutRobot>().GetParserProtocolVersion()) Protocol;
      robot_.reset(new westonrobot::ScoutRobot(Protocol::AGX_V2,true));
      if (!robot_->Connect(interface)) throw std::runtime_error("SDK Connect failed");
      robot_->EnableCommandedMode();
    }
    sender_.reset(new nso_radius::PeriodicSender(gate_,[this](double v,double w) {
      if (execute_) robot_->SetMotionCommand(v,w);
      else captured_.fetch_add(1);  // Capture mode never constructs/opens SDK.
    }));
    sender_->start();
    command_=node.subscribe("/nso_radius/guarded_cmd",1,&RadiusNode::command,this,ros::TransportHints().tcpNoDelay());
    stop_=node.subscribe("/nso_radius/stop",1,&RadiusNode::stop,this);
    fault_=node.advertise<std_msgs::Int32>("/nso_radius/driver_fault",1,true);
    ROS_INFO_STREAM("radius sender ready; execute="<<execute_<<" session="<<session_);
  }
  ~RadiusNode() {
    if (sender_) {
      sender_->latch_stop();
      // Independent zero attempts continue during graceful shutdown.
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      sender_->stop_and_join(); sender_.reset();
    }
    robot_.reset();
  }
  void publish_fault() {
    std_msgs::Int32 value; value.data=static_cast<int>(gate_.fault()); fault_.publish(value);
  }
 private:
  void command(const ros::MessageEvent<geometry_msgs::TwistStamped const>& event) {
    const auto& msg=event.getMessage();
    if (event.getPublisherName()!=guard_node_ || msg->header.frame_id!=session_ ||
        !source_.accept(msg->header.seq,static_cast<std::int64_t>(msg->header.stamp.toNSec()),
                        static_cast<std::int64_t>(ros::Time::now().toNSec()))) {
      gate_.fail(nso_radius::Fault::external_guard); return;
    }
    const auto& t=msg->twist;
    gate_.submit_live(nso_radius::Command(t.linear.x,t.linear.y,t.linear.z,
                                         t.angular.x,t.angular.y,t.angular.z));
  }
  void stop(const std_msgs::Bool::ConstPtr& message) {
    if (message->data) gate_.fail(nso_radius::Fault::external_guard);
  }
  bool execute_;
  std::atomic<std::uint64_t> captured_;
  std::string session_,guard_node_;
  // Destruction/explicit cleanup joins the only sender before closing SDK.
  std::unique_ptr<westonrobot::ScoutRobot> robot_;
  nso_radius::CommandGate gate_;
  nso_radius::SourceContract source_;
  std::unique_ptr<nso_radius::PeriodicSender> sender_;
  ros::Subscriber command_,stop_;
  ros::Publisher fault_;
};
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"nso_scout_radius_driver_v1",ros::init_options::NoSigintHandler);
  std::signal(SIGINT,signal_handler); std::signal(SIGTERM,signal_handler);
  try {
    ros::NodeHandle node,params("~"); RadiusNode driver(node,params);
    while (ros::ok() && !interrupted) {
      ros::spinOnce(); driver.publish_fault();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("radius driver: "<<error.what()); return 2;
  }
  return 0;
}
