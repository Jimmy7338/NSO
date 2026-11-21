#!/usr/bin/env bash
set -euo pipefail
exec /home/ouc/nso_robot/workspaces/nso_hybrid_20261029/run_hybrid.sh --mode nso --joint-configuration /home/ouc/nso_robot/analyses/reboot_recovery_20261109/fresh_C40/configuration.json --execute --duration 600 --nso-correction-reverse --report /home/ouc/nso_robot/reports/nso-hybrid-20261109/native_timing_C40_20261109_01
