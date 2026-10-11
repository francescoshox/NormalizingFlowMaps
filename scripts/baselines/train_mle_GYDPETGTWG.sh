#!/usr/bin/env bash
# Train the standard TARFlow MLE baseline. Append Hydra key=value overrides.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
exec "${PYTHON:-python}" -m transferable_samplers.train \
  experiment=baselines/single_system/train/tarflow_GYDPETGTWG \
  task_name=mle_GYDPETGTWG_train \
  tags='[mle,tarflow,GYDPETGTWG]' \
  seed=47 \
  "$@"
