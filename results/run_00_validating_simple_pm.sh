#!/bin/bash
# 00-validating-simple-pm: Simple PM validation
# 512 cube, box 500 Mpc/h, 10 steps, 10 shells, t0=0.001
# KDK (LPT2, shell-spacing=a) + BF (LPT1, shell-spacing=growth)
# Usage: SLURM_SCRIPT=/path/to/script.sh bash run_00_validating_simple_pm.sh
set -euo pipefail

ACCOUNT=tkc
NODES=4
TIME_LIMIT="00:30:00"
OUTPUT_DIR="results/00-validating-simple-pm"

echo "=== 00-validating-simple-pm: KDK (LPT2, shell-spacing=a) ==="
python -m launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" \
    --nodes "$NODES" \
    --pdim 16 1 \
    --time-limit "$TIME_LIMIT" \
    --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR" \
    --mesh-size 512 512 512 \
    --box-size 500.0 500.0 500.0 \
    --nside 512 \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 10 \
    --shell-spacing a \
    --drift-on-lightcone \
    --solver kdk \
    --lpt-order 2

echo "=== 00-validating-simple-pm: BF (LPT1, shell-spacing=growth) ==="
python -m launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" \
    --nodes "$NODES" \
    --pdim 16 1 \
    --time-limit "$TIME_LIMIT" \
    --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR" \
    --mesh-size 512 512 512 \
    --box-size 500.0 500.0 500.0 \
    --nside 512 \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 10 \
    --shell-spacing growth \
    --drift-on-lightcone \
    --solver bf \
    --lpt-order 1
