#!/usr/bin/env bash
set -euo pipefail
exec /home/ouc/nso_robot/workspaces/nso_hybrid_20261029/run_hybrid.sh --mode nso --joint-configuration /home/ouc/nso_robot/analyses/resume_20261112/configuration_S96.json --execute --duration 1200 --duration-profile long_budget_1200s_v1 --nso-correction-reverse --report /home/ouc/nso_robot/reports/nso-hybrid-20261112/resume_S96_20261112_01
