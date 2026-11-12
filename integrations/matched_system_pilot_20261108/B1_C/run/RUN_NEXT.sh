#!/usr/bin/env bash
set -euo pipefail
exec /home/ouc/nso_robot/workspaces/nso_hybrid_20261029/run_hybrid.sh --mode nso --joint-configuration /home/ouc/nso_robot/analyses/matched_system_pilot_20261108/B1_C/fresh/configuration.json --execute --duration 600 --nso-correction-reverse --report /home/ouc/nso_robot/reports/nso-hybrid-20261108/matched_system_pilot_20261108_B1_C
