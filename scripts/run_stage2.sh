#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the Ubuntu training server.
variant="${1:-joint}"
if [ "$#" -gt 0 ]; then shift; fi
case "$variant" in
    baseline) a2=false; cdfp=false ;;
    a2dfp)    a2=true;  cdfp=false ;;
    cdfp)     a2=false; cdfp=true ;;
    joint)    a2=true;  cdfp=true ;;
    *) echo "Usage: bash scripts/run_stage2.sh baseline|a2dfp|cdfp|joint [train.py options]" >&2; exit 2 ;;
esac

# Use the same sampler in all controlled ablations. Extra CLI options can set
# server data, BPE, local CLIP weights, output directory, device and batch size.
python train.py --mode baseline --name "stage2_${variant}" \
    --dataset_name RSTPReid --loss_names 'sdm+id' \
    --use_a2dfp "$a2" --use_cdfp "$cdfp" \
    --sampler identity --num_instance 4 "$@"
