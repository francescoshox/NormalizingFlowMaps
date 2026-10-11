#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 DATA FID_REFERENCE [WORKDIR]" >&2
    exit 2
fi
NFM_PYTHON=${NFM_PYTHON:-python3}
NFM_RUN_DIR=${3:-runs/cifar10_nfmd}
"$NFM_PYTHON" -m nfm.train --config configs/cifar10_nfmd.yaml --data "$1" --workdir "$NFM_RUN_DIR"
if [[ ${JAX_PROCESS_INDEX:-${SLURM_PROCID:-0}} == 0 ]]; then
    NFM_CHECKPOINT=$("$NFM_PYTHON" -c 'from nfm.checkpoint import latest; import sys; print(latest(sys.argv[1]))' "$NFM_RUN_DIR")
    "$NFM_PYTHON" -m nfm.evaluate --config configs/cifar10_nfmd.yaml --checkpoint "$NFM_CHECKPOINT" \
        --reference "$2" --output "$NFM_RUN_DIR/evaluation"
fi
