from pathlib import Path
p=Path(__file__).parent
s=(p/'original/scout_radius_command_timeout_v1.hpp').read_text()
a=s.index('class CommandGate {');b=s.index('\nclass PeriodicSender',a)
new=r'''class CommandGate {
 public:
  CommandGate(): fault_(Fault::none), middle_(1), back_(2), front_(0),
      producer_armed_(false), producer_seen_(-1), producer_input_(-1),
      consumer_seen_(-1), last_tick_ns_(-1) {
    // No fallback to an implementation that hides a blocking publication lock.
    if (!middle_.is_lock_free() || !fault_.is_lock_free())
      throw std::runtime_error("lock-free gate atomics required");
  }
  CommandGate(const CommandGate&) = delete;
  CommandGate& operator=(const CommandGate&) = delete;
  Fault fault() const { return fault_.load(); }
  void fail(Fault value) {
    if (value == Fault::none) return;
    Fault expected=Fault::none; fault_.compare_exchange_strong(expected,value);
  }
  bool submit(const Command& c,std::int64_t now) {
    return submit_with(c,[now]{return now;},[]{});
  }
  bool submit_live(const Command& c) {
    return submit_with(c,[]{return steady_ns();},[]{});
  }
  Motion sample(std::int64_t now) { return sample_with([now]{return now;},nullptr); }
  Motion sample_live(TimingRow* timing=nullptr) {
    return sample_with([]{return steady_ns();},timing);
  }
 private:
  friend struct CommandGateTestAccess;
  // ROS spinOnce is the sole producer; PeriodicSender is the sole consumer.
  // Flags enforce non-overlap, without waiting. Fault 6 now means ownership
  // violation, not ordinary producer/consumer contention.
  struct Claim {
    std::atomic_flag& flag; bool owned;
    explicit Claim(std::atomic_flag& f):flag(f),owned(!f.test_and_set()) {}
    ~Claim(){if(owned) flag.clear();}
  };
  struct Snapshot { Motion desired; std::int64_t input=-1; bool armed=false; };
  template<class Clock,class BeforePublish>
  bool submit_with(const Command& value,Clock clock,BeforePublish before_publish) {
    Claim claim(producer_busy_);
    if (!claim.owned) { fail(Fault::lock_contention); return false; }
    const std::int64_t now=clock();
    if (now<0 || (producer_seen_>=0 && now<producer_seen_)) {
      fail(Fault::clock_regressed); return false;
    }
    producer_seen_=now;
    if(fault()!=Fault::none) return false;
    if (!std::isfinite(value.linear_x) || !std::isfinite(value.linear_y) ||
        !std::isfinite(value.linear_z) || !std::isfinite(value.angular_x) ||
        !std::isfinite(value.angular_y) || !std::isfinite(value.angular_z) ||
        value.linear_y!=0 || value.linear_z!=0 || value.angular_x!=0 || value.angular_y!=0 ||
        std::abs(value.linear_x)>kMaximumLinear || std::abs(value.angular_z)>kMaximumAngular) {
      fail(Fault::invalid_command); return false;
    }
    Motion requested(value.linear_x,value.angular_z);
    if(!producer_armed_ && !requested.zero()) { fail(Fault::first_command_not_zero); return false; }
    if(producer_armed_ && now-producer_input_>=kTimeoutNs) { fail(Fault::input_expired); return false; }
    producer_armed_=true; producer_input_=now;
    slots_[back_].desired=requested; slots_[back_].input=now; slots_[back_].armed=true;
    before_publish(); // Private deterministic preemption seam; production no-op.
    if(fault()!=Fault::none) return false;
    // Exchange transfers ownership: neither thread accesses the middle slot.
    back_=middle_.exchange(back_|kDirty,std::memory_order_acq_rel)&kIndex;
    return true;
  }
  template<class Clock>
  Motion sample_with(Clock clock,TimingRow* timing) {
    Claim claim(consumer_busy_);
    if(!claim.owned) { fail(Fault::lock_contention); return Motion(); }
    // Field retained for timing schema compatibility: acquisition is ownership
    // claim, not a mutex acquisition. No timed wait remains.
    if(timing) timing->lock_acquired_ns=steady_ns();
    if(middle_.load(std::memory_order_acquire)&kDirty)
      front_=middle_.exchange(front_,std::memory_order_acq_rel)&kIndex;
    const std::int64_t now=clock();
    if(timing) {timing->gate_now_ns=now;timing->gate_prev_tick_ns=last_tick_ns_;}
    if(now<0 || (consumer_seen_>=0 && now<consumer_seen_)) {fail(Fault::clock_regressed);return Motion();}
    consumer_seen_=now;
    if(fault()!=Fault::none) return Motion();
    if(last_tick_ns_>=0 && now-last_tick_ns_>kMaximumTickGapNs) {fail(Fault::sender_late);return Motion();}
    last_tick_ns_=now;
    const Snapshot& s=slots_[front_];
    if(!s.armed) return Motion();
    if(now<s.input) {fail(Fault::clock_regressed);return Motion();}
    if(now-s.input>=kTimeoutNs) {fail(Fault::input_expired);return Motion();}
    return fault()==Fault::none?s.desired:Motion();
  }
  static const unsigned kDirty=4,kIndex=3;
  std::atomic<Fault> fault_;
  std::atomic<unsigned> middle_;
  std::atomic_flag producer_busy_=ATOMIC_FLAG_INIT,consumer_busy_=ATOMIC_FLAG_INIT;
  std::array<Snapshot,3> slots_{};
  unsigned back_,front_;
  bool producer_armed_;
  std::int64_t producer_seen_,producer_input_,consumer_seen_,last_tick_ns_;
};
'''
(p/'scout_radius_command_timeout_v1.hpp').write_text(s[:a]+new+s[b:])
