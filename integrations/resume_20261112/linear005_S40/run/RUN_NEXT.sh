#!/usr/bin/env bash
set -euo pipefail
exec /home/ouc/nso_robot/workspaces/nso_hybrid_20261029/run_hybrid.sh --mode nso --joint-configuration /home/ouc/nso_robot/analyses/resume_20261112/linear005_S40/configuration.json --execute --duration 600 --nso-correction-reverse --report /home/ouc/nso_robot/reports/nso-hybrid-20261112/linear005_S40_20261112_01
