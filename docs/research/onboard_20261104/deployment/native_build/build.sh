#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${NSO_STAGE:-$(cd "$ROOT/.." && pwd)}/hardware_sources"
SDK="$ROOT/sdk-build"
NODE="$ROOT/node-build"
mkdir -p "$SDK" "$NODE"
cmake -S "$SRC/ugv_sdk" -B "$SDK" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build "$SDK" --target ugv_sdk --parallel 2
cmake -S "$SRC/nso_scout_radius_driver_v1" -B "$NODE" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_PREFIX_PATH=/opt/ros/noetic -DNSO_UGV_SDK_INCLUDE_DIR="$SRC/ugv_sdk/include" -DNSO_UGV_SDK_LIBRARY="$SDK/lib/libugv_sdk.a"
cmake --build "$NODE" --target nso_scout_attended_node --parallel 2
