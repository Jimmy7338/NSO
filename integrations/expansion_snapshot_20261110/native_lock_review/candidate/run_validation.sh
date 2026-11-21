#!/bin/bash
set -eu
cd "$(dirname "$0")"
for test in attended_gate_test timing_test concurrency_test; do
 g++ -std=c++11 -O2 -pthread "$test.cpp" -o "$test"
 "./$test"
done
g++ -std=c++11 -O2 -pthread ../reproduce.cpp -o old_reproduce
./old_reproduce
diff -u original/scout_radius_command_timeout_v1.hpp scout_radius_command_timeout_v1.hpp > candidate.patch || test "$?" -eq 1
