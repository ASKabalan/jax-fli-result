#!/bin/bash
# 02-against-cosmo-grid: Validation against CosmoGrid catalog
# 2048 cube, box 6000 Mpc/h, nside 1024, t0=0.001
# 10 shells from CosmoGrid a_near/a_far (max_z=1.5):
#   - shells 0-4  (near z~1.5): a ~ 0.394-0.450
#   - shells 15-19 (near  z~0): a ~ 0.557-0.609
# KDK: 30 steps, LPT2, shell-spacing=a
# BF:  10 steps, LPT1, shell-spacing=growth
# Usage: SLURM_SCRIPT=/path/to/script.sh bash run_02_against_cosmo_grid.sh
set -euo pipefail

ACCOUNT=tkc
NODES=32
GPUS=4
TIME_LIMIT="01:00:00"
OUTPUT_DIR="02-against-cosmo-grid"

MESH_SIZE=2048
NSIDE=1024
SLURM_SCRIPT="02-against-cosmo-grid.slurm.sh"

# 10 selected shells from jax-fli/notebooks/a_near.txt and a_far.txt
TS_NEAR="0.4052 0.4165 0.4276 0.4387 0.5565 0.5669 0.5774 0.5879 0.5983"
TS_FAR="0.4165 0.4276 0.4387 0.4497 0.5669 0.5774 0.5879 0.5983 0.6088"

echo "=== 02-against-cosmo-grid: KDK (30 steps, LPT2, shell-spacing=a) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" --nodes "$NODES" --pdim 128 1 \
    --gpus-per-node $GPUS \
    --tasks-per-node $GPUS \
    --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR" \
    --mesh-size $MESH_SIZE $MESH_SIZE $MESH_SIZE \
    --box-size 6000.0 6000.0 6000.0 \
    --nside $NSIDE \
    --t0 0.001  \
    --shell-spacing a \
    --ts-near $TS_NEAR \
    --ts-far $TS_FAR \
    --solver kdk --lpt-order 2

echo "=== 02-against-cosmo-grid: BF (10 steps, LPT1, shell-spacing=growth) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" --nodes "$NODES" --pdim 128 1 \
    --gpus-per-node $GPUS \
    --tasks-per-node $GPUS \
    --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR" \
    --mesh-size $MESH_SIZE $MESH_SIZE $MESH_SIZE \
    --box-size 6000.0 6000.0 6000.0 \
    --nside $NSIDE \
    --t0 0.001 \
    --shell-spacing growth \
    --ts-near $TS_NEAR \
    --ts-far $TS_FAR \
    --solver bf --lpt-order 1
