#include "scout_radius_command_timeout_v1.hpp"
#include <cassert>
#include <future>
#include <iostream>
namespace nso_radius {
struct CommandGateTestAccess {
 static bool paused(CommandGate& g,const Command& c,std::int64_t now,std::promise<void>& entered,std::future<void>& release) {
  return g.submit_with(c,[now]{return now;},[&]{entered.set_value();release.wait();});
 }
 static Motion coherent(CommandGate& g) {
  return g.sample_with([&]{const auto& snap=g.slots_[g.front_];if(snap.armed && snap.input>0){assert(snap.desired.linear==((snap.input%2)?.05:.1));assert(snap.desired.angular==snap.desired.linear*2);}return snap.input<0?0:snap.input;},nullptr);
 }
 static void violate(CommandGate& g) {g.producer_busy_.test_and_set();assert(!g.submit(Command(),0));g.producer_busy_.clear();}
};
}
int main(){using namespace nso_radius;
 // A complete but unpublished newer command never renews the old TTL.
 CommandGate g;assert(g.submit(Command(),0));assert(g.submit(Command(.05,0,0,0,0,.1),10000000));assert(g.sample(10000000).linear==.05);
 std::promise<void> entered,release;auto ready=entered.get_future(),proceed=release.get_future();bool accepted=true;
 std::thread writer([&]{accepted=CommandGateTestAccess::paused(g,Command(.09,0,0,0,0,.2),90000000,entered,proceed);});ready.wait();
 for(long long t=30000000;t<=90000000;t+=20000000){auto m=g.sample(t);assert(m.linear==.05&&m.angular==.1);assert(g.fault()==Fault::none);}
 assert(g.sample(110000000).zero());assert(g.fault()==Fault::input_expired);release.set_value();writer.join();assert(!accepted);
 CommandGate owner;CommandGateTestAccess::violate(owner);assert(owner.fault()==Fault::lock_contention);assert(owner.sample(0).zero());
 for(Fault f:{Fault::shutdown,Fault::external_guard}){CommandGate stop;assert(stop.submit(Command(),0));assert(stop.submit(Command(.1),1));stop.fail(f);assert(stop.sample(1).zero());assert(!stop.submit(Command(),2));assert(stop.fault()==f);}
 CommandGate invalid;assert(invalid.submit(Command(),0));assert(!invalid.submit(Command(.11),1));assert(invalid.sample(1).zero());
 CommandGate first;assert(!first.submit(Command(.01),0));assert(first.fault()==Fault::first_command_not_zero);
 CommandGate skipped;assert(skipped.submit(Command(),0));assert(skipped.submit(Command(.1),1));assert(skipped.sample(1).linear==.1);
 CommandGate expired;assert(expired.submit(Command(),0));assert(!expired.submit(Command(),100000000));assert(expired.fault()==Fault::input_expired);
 SourceContract source;assert(source.accept(1,1000000000,1080000000));assert(!source.accept(2,1000000001,1080000002));
 CommandGate age;SourceContract age_contract;
 assert(age_contract.accept(1,1000000000,1000000000));assert(age.submit(Command(),0));
 assert(age_contract.accept(2,1020000000,1020000000));assert(age.submit(Command(.05),20000000));
 for(long long t=20000000;t<=100000000;t+=20000000)assert(age.sample(t).linear==.05);
 // Driver callback rejects the stale source before calling submit_live, and latches external_guard.
 assert(!age_contract.accept(3,1020000001,1100000002));age.fail(Fault::external_guard);
 assert(age.sample(120000000).zero());assert(age.fault()==Fault::external_guard);
 CommandGate boundary;assert(boundary.submit(Command(),0));assert(boundary.sample(0).zero());assert(boundary.sample(40000000).zero());assert(boundary.fault()==Fault::none);boundary.sample(80000001);assert(boundary.fault()==Fault::sender_late);
 CommandGate backwards;assert(backwards.submit(Command(),100));backwards.sample(100);backwards.sample(99);assert(backwards.fault()==Fault::clock_regressed);
 // Real concurrent publication stress, synthetic monotonic timestamps avoid OS
 // scheduler noise testing a deadline instead of snapshot coherence.
 CommandGate stress;assert(stress.submit(Command(),0));std::atomic<bool> done(false);std::atomic<int> count(0);
 std::thread producer([&]{for(int i=1;i<=200000;++i){double v=(i%2)?.05:.1;assert(stress.submit(Command(v,0,0,0,0,v*2),i));}done=true;});
 while(!done.load()){auto m=CommandGateTestAccess::coherent(stress);assert(m.angular==m.linear*2);assert(stress.fault()==Fault::none);++count;}
 producer.join();assert(stress.fault()==Fault::none);assert(CommandGateTestAccess::coherent(stress).linear==.1);
 std::cout<<"paused publication previous-command/TTL, ownership fault6, stop/invalid/firstzero, boundaries, 200000 coherent publications passed; reads="<<count<<"\n";
}
