#include "scout_radius_command_timeout_v1.hpp"
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <thread>
namespace nso_radius {
struct CommandGateTestAccess {
 static Motion sample(CommandGate& g,std::int64_t now,TimingRow* row) { return g.sample_locked(now,row); }
 static Motion lock_failure(CommandGate& g,TimingRow* row) { return g.sample_live_with([](std::unique_lock<std::timed_mutex>&,std::chrono::milliseconds){return false;},[](){return 12;},row); }
};
struct PeriodicSenderTestAccess { static void send(PeriodicSender& s, Motion v, TimingRow* r) { s.send(v,r); } };
}
int main() {
 using namespace nso_radius;
 for (bool instrumented : {false,true}) {
  CommandGate g;TimingRow a,b;assert(g.submit(Command(),1000000000));
  assert(CommandGateTestAccess::sample(g,1000000000,instrumented?&a:nullptr).zero());
  assert(CommandGateTestAccess::sample(g,1040000000,instrumented?&b:nullptr).zero());
  assert(g.fault()==Fault::none);
  assert(CommandGateTestAccess::sample(g,1080000001,instrumented?&b:nullptr).zero());
  assert(g.fault()==Fault::sender_late);
  if(instrumented){assert(b.gate_prev_tick_ns==1040000000);assert(b.gate_now_ns==1080000001);}
 }
 CommandGate regressed;TimingRow bad;assert(regressed.submit(Command(),100));CommandGateTestAccess::sample(regressed,99,&bad);assert(regressed.fault()==Fault::clock_regressed);assert(bad.gate_now_ns==99 && bad.gate_prev_tick_ns==-1);TimingRow unavailable;CommandGate lockgate;CommandGateTestAccess::lock_failure(lockgate,&unavailable);assert(lockgate.fault()==Fault::lock_contention);assert(unavailable.gate_now_ns==-1 && unavailable.gate_prev_tick_ns==-1 && unavailable.lock_acquired_ns==-1);
 TimingRing ring;TimingRow fault;fault.sleep_requested_ns=91;fault.send_enter_ns=123;fault.gate_prev_tick_ns=4;fault.gate_now_ns=40000005;fault.fault_after_sample=5;ring.append(fault);
 for(int i=0;i<600;++i)ring.append(TimingRow());
 std::ostringstream out;ring.write(out);auto first=out.str().find("\"first_fault\":{\"sleep_requested_ns\":91");assert(first!=std::string::npos);assert(out.str().find("\"send_enter_ns\":123",first)!=std::string::npos);
 std::ostringstream broken;broken.setstate(std::ios::badbit);ring.write(broken);
 CommandGate gate;int count=0;PeriodicSender sender(gate,[&](double v,double w){assert(v==0 && w==0);++count;if(count==2)throw std::runtime_error("fake transport");});
 TimingRow row;PeriodicSenderTestAccess::send(sender,Motion(),&row);PeriodicSenderTestAccess::send(sender,Motion(),&row);assert(gate.fault()==Fault::transport_exception);assert(count==3);assert(row.fault_after_send==7);
 return 0;
}
