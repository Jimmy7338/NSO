#include "scout_radius_command_timeout_v1.hpp"
#include <cassert>
#include <iostream>
using namespace nso_radius;
int main() {
  assert(kMaximumLinear==.10 && kMaximumAngular==.30);
  CommandGate g;
  assert(g.submit(Command(),0));
  assert(g.sample(0).zero());
  assert(g.submit(Command(.1,0,0,0,0,.3),20000000));
  auto active=g.sample(20000000);
  assert(active.linear==.1 && active.angular==.3);
  for(long long t=40000000;t<120000000;t+=20000000) assert(!g.sample(t).zero());
  assert(g.sample(120000000).zero());
  assert(g.fault()==Fault::input_expired);
  assert(!g.submit(Command(),120000001));
  CommandGate first;assert(!first.submit(Command(.01),0));
  CommandGate bounds;assert(bounds.submit(Command(),0));
  assert(!bounds.submit(Command(.101),1));assert(bounds.sample(1).zero());
  CommandGate turn;assert(turn.submit(Command(),0));
  assert(!turn.submit(Command(0,0,0,0,0,.301),1));
  SourceContract s;assert(s.accept(1,1000000000,1080000000));
  assert(!s.accept(2,1000000001,1080000002));
  std::cout<<"attended limits and existing expiry/first-zero contracts passed\n";
}
