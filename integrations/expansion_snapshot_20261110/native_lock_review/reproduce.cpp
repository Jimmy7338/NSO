#include "/home/ouc/nso_robot/staging/ans-live-20261108-native-timing-v2/native/scout_radius_command_timeout_v1.hpp"
#include <cassert>
#include <future>
#include <iostream>
namespace nso_radius {
struct CommandGateTestAccess {
 static void hold(CommandGate& g, std::promise<void>& acquired, std::future<void>& release) {
  std::lock_guard<std::timed_mutex> lock(g.mutex_);
  acquired.set_value();
  release.wait(); // Deterministic stand-in for a descheduled lock owner, not production code.
 }
};
}
int main() {
 using namespace nso_radius;
 CommandGate gate; assert(gate.submit_live(Command()));
 std::promise<void> acquired, release;
 auto ready=acquired.get_future(), proceed=release.get_future();
 std::thread holder([&]{CommandGateTestAccess::hold(gate,acquired,proceed);});
 ready.wait(); TimingRow row; row.sample_enter_ns=steady_ns();
 Motion result=gate.sample_live(&row); row.sample_exit_ns=steady_ns();
 assert(result.zero()); assert(gate.fault()==Fault::lock_contention); assert(row.lock_acquired_ns==-1);
 release.set_value(); holder.join();
 assert(!gate.submit_live(Command())); assert(gate.sample_live().zero());
 std::cout << "fault=" << static_cast<int>(gate.fault()) << " lock_acquired=" << row.lock_acquired_ns
           << " elapsed_ns=" << row.sample_exit_ns-row.sample_enter_ns << " latch_and_zero=pass\n";
}
