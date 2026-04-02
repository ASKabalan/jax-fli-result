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

# ---------------------------------------------------------------------------
# CosmoGrid cosmology — 15 shells (no fixed ts-near/ts-far)
# h=0.73  Omega_b=0.045  Omega_c=0.25378877  w0=-1.1665  sigma8=0.9
# n_s=0.97  Omega_nu=0.0012112348
# ---------------------------------------------------------------------------
COSMO_OC=0.25378877
COSMO_S8=0.9
COSMO_H=0.73
COSMO_OB=0.045
COSMO_W0=-1.1665
COSMO_NS=0.97
COSMO_ONU=0.0012112348

echo "=== 02-against-cosmo-grid: CosmoGrid cosmo — KDK (30 steps, LPT2, shell-spacing=a, nb-shells=15) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" --nodes "$NODES" --pdim 128 1 \
    --gpus-per-node $GPUS \
    --tasks-per-node $GPUS \
    --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR/cosmogrid_cosmo" \
    --mesh-size $MESH_SIZE $MESH_SIZE $MESH_SIZE \
    --box-size 6000.0 6000.0 6000.0 \
    --nside $NSIDE \
    --t0 0.001 \
    --nb-steps 30 \
    --nb-shells 15 \
    --shell-spacing a \
    --drift-on-lightcone \
    --omega-c $COSMO_OC \
    --sigma8 $COSMO_S8 \
    --h $COSMO_H \
    --omega-b $COSMO_OB \
    --w0 $COSMO_W0 \
    --n-s $COSMO_NS \
    --omega-nu $COSMO_ONU \
    --solver kdk --lpt-order 2

echo "=== 02-against-cosmo-grid: CosmoGrid cosmo — BF (10 steps, LPT1, shell-spacing=growth, nb-shells=15) ==="
fli-launcher simulate \
    --mode sbatch \
    --account "$ACCOUNT" --nodes "$NODES" --pdim 128 1 \
    --gpus-per-node $GPUS \
    --tasks-per-node $GPUS \
    --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
    --output-dir "$OUTPUT_DIR/cosmogrid_cosmo" \
    --mesh-size $MESH_SIZE $MESH_SIZE $MESH_SIZE \
    --box-size 6000.0 6000.0 6000.0 \
    --nside $NSIDE \
    --t0 0.001 \
    --nb-steps 10 \
    --nb-shells 15 \
    --shell-spacing growth \
    --drift-on-lightcone \
    --omega-c $COSMO_OC \
    --sigma8 $COSMO_S8 \
    --h $COSMO_H \
    --omega-b $COSMO_OB \
    --w0 $COSMO_W0 \
    --n-s $COSMO_NS \
    --omega-nu $COSMO_ONU \
    --solver bf --lpt-order 1
