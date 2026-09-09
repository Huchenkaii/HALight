#!/usr/bin/env bash
set -euo pipefail

# Activate your Cityflow environment before running this script.
# Place this file at the repository root, alongside Nets/ and flows/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for net in Gudang Xiasha; do
  for demand in high moderate; do
    for drop in 0.0 0.1 0.2 0.3 0.4; do
      python HALight/main.py \
        -net "Nets/${net}/${net}.json" \
        -flow_dir "flows/${net}_${demand}" \
        -obs_drop_prob "$drop"
    done
  done
done
