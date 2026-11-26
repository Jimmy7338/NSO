#ifndef NSO_SCOUT_RADIUS_COMMAND_TIMEOUT_V1_HPP
#define NSO_SCOUT_RADIUS_COMMAND_TIMEOUT_V1_HPP

// Independent C++11 v/w send gate. No ROS, CAN, geometry or navigation here.
// A separate current-pose/obstacle guard must supply the commands. The only
// SDK motion-send path must be PeriodicSender, never a subscription callback.
// The first attempted send is zero. Expiry/faults latch until process restart;
// the sender continues zero attempts even when the callback/main thread stalls.
// This does not cover stopped processes/hosts, blocked I/O or physical braking.

#include <atomic>
#include <array>
#include <ostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace nso_radius {
static const std::int64_t kTimeoutNs = 100000000;
static const std::int64_t kPeriodNs = 20000000;
static const std::int64_t kMaximumTickGapNs = 40000000;
static const double kMaximumLinear = 0.10;
static const double kMaximumAngular = 0.30;

enum class Fault {
  none = 0, invalid_command, first_command_not_zero, clock_regressed,
  input_expired, sender_late, lock_contention, transport_exception,
  external_guard, shutdown
};
struct Command {
  double linear_x, linear_y, linear_z, angular_x, angular_y, angular_z;
  Command(double x=0, double y=0, double z=0, double ax=0, double ay=0, double az=0)
      : linear_x(x), linear_y(y), linear_z(z), angular_x(ax), angular_y(ay), angular_z(az) {}
};
struct Motion {
  double linear, angular;
  Motion(double v=0, double w=0): linear(v), angular(w) {}
  bool zero() const { return linear == 0.0 && angular == 0.0; }
};
// Validate an unlatched source stream before its values reach CommandGate.
// Wall-clock source age is checked separately from the steady-clock send TTL.
class SourceContract {
 public:
  SourceContract(): seen_(false),sequence_(0),stamp_ns_(-1) {}
  bool accept(std::uint32_t sequence,std::int64_t stamp,std::int64_t wall_now) {
    if (stamp<=0 || wall_now<stamp || wall_now-stamp>80000000 ||
        (seen_ && (sequence<=sequence_ || stamp<=stamp_ns_))) return false;
    seen_=true; sequence_=sequence; stamp_ns_=stamp; return true;
  }
 private:
  bool seen_;
  std::uint32_t sequence_;
  std::int64_t stamp_ns_;
};
inline std::int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Sender thread alone writes this bounded ring; readers wait until join.
struct TimingRow {
  std::int64_t sleep_requested_ns=0, sleep_start_ns=0, wake_ns=0;
  std::int64_t sample_enter_ns=-1, lock_acquired_ns=-1, gate_now_ns=-1, gate_prev_tick_ns=-1, sample_exit_ns=-1;
  std::int64_t send_enter_ns=0, send_exit_ns=0; int fault_after_sample=0, fault_after_send=0;
};
class TimingRing {
 public:
  void append(const TimingRow& row) { if (!has_first_fault_ && (row.fault_after_sample || row.fault_after_send)) { first_fault_=row; has_first_fault_=true; } rows_[next_]=row; next_=(next_+1)%rows_.size(); if (size_<rows_.size()) ++size_; }
  void write(std::ostream& out) const {
    out << "{\"schema\":\"nso.native_sender_timing.v1\",\"clock\":\"steady_clock_ns\",\"rows\":[";
    for (std::size_t i=0;i<size_;++i) {
      const TimingRow& r=rows_[(next_+rows_.size()-size_+i)%rows_.size()];
      if (i) out << ',';
      write_row(out,r);
    }
    out << "],\"first_fault\":";
    if (!has_first_fault_) out << "null"; else write_row(out,first_fault_);
    out << "}\n";
  }
 private:
  static void write_row(std::ostream& out,const TimingRow& r) {
      out << "{\"sleep_requested_ns\":"<<r.sleep_requested_ns<<",\"sleep_start_ns\":"<<r.sleep_start_ns
          <<",\"wake_ns\":"<<r.wake_ns<<",\"sample_enter_ns\":"<<r.sample_enter_ns
          <<",\"lock_acquired_ns\":"<<r.lock_acquired_ns<<",\"gate_now_ns\":"<<r.gate_now_ns
          <<",\"gate_prev_tick_ns\":"<<r.gate_prev_tick_ns<<",\"sample_exit_ns\":"<<r.sample_exit_ns
          <<",\"send_enter_ns\":"<<r.send_enter_ns<<",\"send_exit_ns\":"<<r.send_exit_ns
          <<",\"fault_after_sample\":"<<r.fault_after_sample<<",\"fault_after_send\":"<<r.fault_after_send<<'}';
  }
  std::array<TimingRow,512> rows_{}; std::size_t next_=0,size_=0; TimingRow first_fault_{}; bool has_first_fault_=false;
};
class CommandGate {
 public:
  CommandGate(): fault_(Fault::none), armed_(false), desired_(),
      last_seen_ns_(-1), last_input_ns_(-1), last_tick_ns_(-1) {}
  CommandGate(const CommandGate&) = delete;
  CommandGate& operator=(const CommandGate&) = delete;
  Fault fault() const { return fault_.load(); }
  void fail(Fault value) {
    if (value == Fault::none) return;
    Fault expected=Fault::none;
    fault_.compare_exchange_strong(expected, value);
  }
  bool submit(const Command& command, std::int64_t now) {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    return submit_locked(command, now);
  }
  bool submit_live(const Command& command) {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    return submit_locked(command, steady_ns());
  }
  Motion sample(std::int64_t now) {
    std::unique_lock<std::timed_mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) { fail(Fault::lock_contention); return Motion(); }
    return sample_locked(now);
  }
  Motion sample_live(TimingRow* timing=nullptr) {
    return sample_live_with(
        [](std::unique_lock<std::timed_mutex>& lock, std::chrono::milliseconds limit) {
          return lock.try_lock_for(limit);
        }, []() { return steady_ns(); }, timing);
  }
 private:
  friend struct CommandGateTestAccess;
  // A callback briefly owns this lock to replace six command values. Permit a
  // bounded handoff, then evaluate the unchanged sender/input deadlines using
  // a fresh clock reading. Waiting never renews either original deadline.
  // The private policy seam lets tests control scheduling without sleep races.
  template <typename Waiter, typename Clock>
  Motion sample_live_with(Waiter wait, Clock clock, TimingRow* timing=nullptr) {
    std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
    if (!wait(lock, std::chrono::milliseconds(1))) {
      fail(Fault::lock_contention); return Motion();
    }
    if (timing) timing->lock_acquired_ns=steady_ns();
    return sample_locked(clock(),timing);
  }
  bool time_ok(std::int64_t now) {
    if (now<0 || (last_seen_ns_>=0 && now<last_seen_ns_)) {
      fail(Fault::clock_regressed); return false;
    }
    last_seen_ns_=now; return true;
  }
  bool submit_locked(const Command& value, std::int64_t now) {
    if (!time_ok(now) || fault()!=Fault::none) return false;
    if (!std::isfinite(value.linear_x) || !std::isfinite(value.linear_y) ||
        !std::isfinite(value.linear_z) || !std::isfinite(value.angular_x) ||
        !std::isfinite(value.angular_y) || !std::isfinite(value.angular_z) ||
        value.linear_y!=0 || value.linear_z!=0 || value.angular_x!=0 || value.angular_y!=0 ||
        std::abs(value.linear_x)>kMaximumLinear || std::abs(value.angular_z)>kMaximumAngular) {
      fail(Fault::invalid_command); return false;
    }
    const Motion requested(value.linear_x,value.angular_z);
    if (!armed_ && !requested.zero()) { fail(Fault::first_command_not_zero); return false; }
    if (armed_ && now-last_input_ns_>=kTimeoutNs) { fail(Fault::input_expired); return false; }
    armed_=true; desired_=requested; last_input_ns_=now;
    return true;
  }
  Motion sample_locked(std::int64_t now, TimingRow* timing=nullptr) {
    if (timing) { timing->gate_now_ns=now; timing->gate_prev_tick_ns=last_tick_ns_; }
    if (!time_ok(now) || fault()!=Fault::none) return Motion();
    if (last_tick_ns_>=0 && now-last_tick_ns_>kMaximumTickGapNs) {
      fail(Fault::sender_late); return Motion();
    }
    last_tick_ns_=now;
    if (!armed_) return Motion();
    if (now-last_input_ns_>=kTimeoutNs) { fail(Fault::input_expired); return Motion(); }
    return desired_;
  }
  std::atomic<Fault> fault_;
  std::timed_mutex mutex_;
  bool armed_;
  Motion desired_;
  std::int64_t last_seen_ns_,last_input_ns_,last_tick_ns_;
};

class PeriodicSender {
 public:
  typedef std::function<void(double,double)> Sender;
  PeriodicSender(CommandGate& gate, Sender sender)
      : gate_(gate),sender_(sender),stop_(false),started_(false) {
    if (!sender) throw std::invalid_argument("motion-send callback required");
  }
  PeriodicSender(const PeriodicSender&) = delete;
  PeriodicSender& operator=(const PeriodicSender&) = delete;
  void start() {
    if (started_) throw std::logic_error("radius sender cannot restart");
    started_=true; worker_=std::thread(&PeriodicSender::run,this);
  }
  // Fail first and leave running for continuous zero attempts during cleanup.
  void latch_stop() { gate_.fail(Fault::shutdown); }
  void stop_and_join() {
    latch_stop(); stop_.store(true);
    if (worker_.joinable()) worker_.join();
  }
  ~PeriodicSender() { stop_and_join(); }
  const TimingRing& timing() const { return timing_; }
 private:
  friend struct PeriodicSenderTestAccess;
  void send(Motion value, TimingRow* timing=nullptr) {
    if (gate_.fault()!=Fault::none) value=Motion();
    if (timing) timing->send_enter_ns=steady_ns();
    try { sender_(value.linear,value.angular); }
    catch (...) {
      gate_.fail(Fault::transport_exception);
      try { sender_(0.0,0.0); } catch (...) {}
    }
    if (timing) { timing->send_exit_ns=steady_ns(); timing->fault_after_send=static_cast<int>(gate_.fault()); }
  }
  void run() {
    // Even if a callback has already supplied a valid command, startup sends zero.
    send(Motion());
    auto deadline=std::chrono::steady_clock::now()+std::chrono::nanoseconds(kPeriodNs);
    while (!stop_.load()) {
      TimingRow row; row.sleep_start_ns=steady_ns();
      row.sleep_requested_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch()).count()-row.sleep_start_ns;
      std::this_thread::sleep_until(deadline);
      row.wake_ns=steady_ns();
      if (stop_.load()) break;
      row.sample_enter_ns=steady_ns();
      Motion value=gate_.sample_live(&row);
      row.sample_exit_ns=steady_ns(); row.fault_after_sample=static_cast<int>(gate_.fault());
      send(value,&row); timing_.append(row);
      deadline=std::chrono::steady_clock::now()+std::chrono::nanoseconds(kPeriodNs);
    }
    send(Motion());
  }
  CommandGate& gate_;
  TimingRing timing_;
  Sender sender_;
  std::atomic<bool> stop_;
  bool started_;
  std::thread worker_;
};
}  // namespace nso_radius
#endif
