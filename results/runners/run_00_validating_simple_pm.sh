#!/bin/bash
# 00-validating-simple-pm: Simple PM validation
# 512 cube, box 500 Mpc/h, 10 steps, 10 shells, t0=0.001
# KDK (LPT2, shell-spacing=a) + BF (LPT1, shell-spacing=growth)
# Usage: SLURM_SCRIPT=/path/to/script.sh bash run_00_validating_simple_pm.sh
set -euo pipefail
SLURM_SCRIPT=${SLURM_SCRIPT:-"scripts/slurm/default.sh"}

ACCOUNT=tkc
NODES=4
TIME_LIMIT="00:30:00"
OUTPUT_DIR="results/00-validating-simple-pm"
MESH_SIZE=512
NSIDE=512
MESH_SIZE_LARGE=1024
NSIDE_LARGE=1024

echo "=== 00-validating-simple-pm: KDK (LPT2, shell-spacing=a) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" \
    --nodes "$NODES" \
    --pdim 16 1 \
    --time-limit "$TIME_LIMIT" \
    --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR/KDK" \
    --mesh-size $MESH_SIZE $MESH_SIZE $MESH_SIZE \
    --box-size 500.0 500.0 500.0 \
    --nside $NSIDE \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 10 \
    --shell-spacing a \
    --min-width 10.0 \
    --drift-on-lightcone \
    --solver kdk \
    --lpt-order 2 \
    --simulation-type lensing \
    --nz-shear 0.05 0.08 \
    --min-z 0.01 \
    --max-z 0.085 \
    --n-integrate 10

echo "=== 00-validating-simple-pm: BF (LPT1, shell-spacing=growth) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" \
    --nodes "$NODES" \
    --pdim 16 1 \
    --time-limit "$TIME_LIMIT" \
    --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR/BF" \
    --mesh-size $MESH_SIZE $MESH_SIZE $MESH_SIZE \
    --box-size 500.0 500.0 500.0 \
    --nside $NSIDE \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 10 \
    --shell-spacing growth \
    --min-width 5.0 \
    --drift-on-lightcone \
    --solver bf \
    --lpt-order 1 \
    --simulation-type lensing \
    --nz-shear 0.05 0.08 \
    --min-z 0.01 \
    --max-z 0.085 \
    --n-integrate 10

echo "=== 00-validating-simple-pm: KDK large (LPT2, shell-spacing=a) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" \
    --nodes "$NODES" \
    --pdim 16 1 \
    --time-limit "$TIME_LIMIT" \
    --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR/KDK" \
    --mesh-size $MESH_SIZE_LARGE $MESH_SIZE_LARGE $MESH_SIZE_LARGE \
    --box-size 1000.0 1000.0 1000.0 \
    --nside $NSIDE_LARGE \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 10 \
    --shell-spacing a \
    --min-width 10.0 \
    --drift-on-lightcone \
    --solver kdk \
    --lpt-order 2 \
    --simulation-type lensing \
    --nz-shear 0.1 0.17 \
    --min-z 0.01 \
    --max-z 0.18 \
    --n-integrate 10

echo "=== 00-validating-simple-pm: BF large (LPT1, shell-spacing=growth) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" \
    --nodes "$NODES" \
    --pdim 16 1 \
    --time-limit "$TIME_LIMIT" \
    --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR/BF" \
    --mesh-size $MESH_SIZE_LARGE $MESH_SIZE_LARGE $MESH_SIZE_LARGE \
    --box-size 1000.0 1000.0 1000.0 \
    --nside $NSIDE_LARGE \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 10 \
    --shell-spacing growth \
    --min-width 5.0 \
    --drift-on-lightcone \
    --solver bf \
    --lpt-order 1 \
    --simulation-type lensing \
    --nz-shear 0.1 0.17 \
    --min-z 0.01 \
    --max-z 0.18 \
    --n-integrate 10
