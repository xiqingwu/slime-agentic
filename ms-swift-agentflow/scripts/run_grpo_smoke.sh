#!/bin/bash
set -euo pipefail
export MAX_STEPS=${MAX_STEPS:-1}
export DATASET=${DATASET:-/data/MAT/mat_swift_smoke.jsonl}
exec "$(dirname -- "${BASH_SOURCE[0]}")/train_grpo_lora.sh"
