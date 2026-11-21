#!/bin/bash
set -euo pipefail
candidate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
c++ -O3 -std=c++11 -Wall -Wextra -Werror -fPIC -shared "$candidate_dir/exact_visibility_native.cpp" -o "$candidate_dir/libexact_visibility_native.so"
