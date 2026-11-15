#!/usr/bin/env bash
set -euo pipefail
# The baseline uses SDM+ID. Supply --clip_weights PATH or explicit --allow_download.
python train.py --mode baseline --dataset_name RSTPReid --loss_names 'sdm+id' "$@"
