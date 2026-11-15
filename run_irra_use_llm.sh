#!/usr/bin/env bash
set -euo pipefail
echo "LLM/image augmentation is not enabled in recovery stage 1. Use run_irra.sh for the baseline; augmentation will be restored in stage 4." >&2
exit 2
