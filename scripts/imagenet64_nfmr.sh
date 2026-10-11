#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 COUPLING_DATA FID_REFERENCE [WORKDIR]" >&2
    exit 2
fi
NFM_PYTHON=${NFM_PYTHON:-python3}
NFM_RUN_DIR=${3:-runs/imagenet64_nfmr}
"$NFM_PYTHON" -m nfm.train --config configs/imagenet64_nfmr_flow_matching.yaml \
    --data "$1" --workdir "$NFM_RUN_DIR/flow_matching"
NFM_FM_CHECKPOINT=$("$NFM_PYTHON" -c 'from nfm.checkpoint import latest; import sys; print(latest(sys.argv[1]))' "$NFM_RUN_DIR/flow_matching")
"$NFM_PYTHON" -m nfm.train --config configs/imagenet64_nfmr_mean_flow.yaml \
    --data "$1" --workdir "$NFM_RUN_DIR/mean_flow" --init-from "$NFM_FM_CHECKPOINT"
if [[ ${JAX_PROCESS_INDEX:-${SLURM_PROCID:-0}} == 0 ]]; then
    NFM_CHECKPOINT=$("$NFM_PYTHON" -c 'from nfm.checkpoint import latest; import sys; print(latest(sys.argv[1]))' "$NFM_RUN_DIR/mean_flow")
    "$NFM_PYTHON" -m nfm.evaluate --config configs/imagenet64_nfmr_mean_flow.yaml --checkpoint "$NFM_CHECKPOINT" \
        --reference "$2" --output "$NFM_RUN_DIR/evaluation"
fi
