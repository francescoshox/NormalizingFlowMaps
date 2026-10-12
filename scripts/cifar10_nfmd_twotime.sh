#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 DATA FID_REFERENCE [WORKDIR]" >&2
    exit 2
fi
NFM_PYTHON=${NFM_PYTHON:-python3}
NFM_RUN_DIR=${3:-runs/cifar10_nfmd_twotime}
"$NFM_PYTHON" -m nfm.train --config configs/cifar10_nfmd_twotime.yaml --data "$1" --workdir "$NFM_RUN_DIR" --reference "$2"
