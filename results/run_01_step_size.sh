#!/bin/bash
# 01-step_size: Step-size convergence study
# Meshes: 512, 1024, 1536, 2048, 3072, 4096 — box 6000 Mpc/h cube
# Steps: 5 6 8 10 15 20 30 50
# nb-shells: equals nb-steps for steps<=8, else 10
# KDK (LPT2, shell-spacing=a) + BF (LPT1, shell-spacing=growth)
# Painting: 512->nside512; 1024-3072->nside1024; 4096->nside1024,paint-nside2048
# Usage: SLURM_SCRIPT=/path/to/script.sh bash run_01_step_size.sh
set -euo pipefail

ACCOUNT=tkc
NODES=64
TIME_LIMIT="01:00:00"
OUTPUT_DIR="results/01-step_size"

for NB_STEPS in 5 6 8 10 15 20 30 50; do
    if [ "$NB_STEPS" -le 8 ]; then
        NB_SHELLS=$NB_STEPS
    else
        NB_SHELLS=10
    fi

    echo "=== NB_STEPS=${NB_STEPS} NB_SHELLS=${NB_SHELLS} ==="

    # ---- KDK: LPT2, shell-spacing=a ----

    # 512 (nside 512)
    fli-launcher simulate \
        --mode sbatch \
        --account "$ACCOUNT" --nodes "$NODES" --pdim 256 1 \
        --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --mesh-size 512 512 512 \
        --box-size 6000.0 6000.0 6000.0 \
        --nside 512 \
        --t0 0.001 --nb-steps "$NB_STEPS" \
        --nb-shells "$NB_SHELLS" --shell-spacing a \
        --drift-on-lightcone \
        --solver kdk --lpt-order 2

    # 1024 / 1536 / 2048 / 3072 (nside 1024)
    fli-launcher simulate \
        --mode sbatch \
        --account "$ACCOUNT" --nodes "$NODES" --pdim 256 1 \
        --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --mesh-size 1024 1024 1024  1536 1536 1536  2048 2048 2048  3072 3072 3072 \
        --box-size 6000.0 6000.0 6000.0 \
        --nside 1024 \
        --t0 0.001 --nb-steps "$NB_STEPS" \
        --nb-shells "$NB_SHELLS" --shell-spacing a \
        --drift-on-lightcone \
        --solver kdk --lpt-order 2

    # 4096 (nside 1024, paint-nside 2048)
    fli-launcher simulate \
        --mode sbatch \
        --account "$ACCOUNT" --nodes "$NODES" --pdim 256 1 \
        --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --mesh-size 4096 4096 4096 \
        --box-size 6000.0 6000.0 6000.0 \
        --nside 1024 --paint-nside 2048 \
        --t0 0.001 --nb-steps "$NB_STEPS" \
        --nb-shells "$NB_SHELLS" --shell-spacing a \
        --drift-on-lightcone \
        --solver kdk --lpt-order 2

    # ---- BF: LPT1, shell-spacing=growth ----

    # 512 (nside 512)
    fli-launcher simulate \
        --mode sbatch \
        --account "$ACCOUNT" --nodes "$NODES" --pdim 256 1 \
        --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --mesh-size 512 512 512 \
        --box-size 6000.0 6000.0 6000.0 \
        --nside 512 \
        --t0 0.001 --nb-steps "$NB_STEPS" \
        --nb-shells "$NB_SHELLS" --shell-spacing growth \
        --drift-on-lightcone \
        --solver bf --lpt-order 1

    # 1024 / 1536 / 2048 / 3072 (nside 1024)
    fli-launcher simulate \
        --mode sbatch \
        --account "$ACCOUNT" --nodes "$NODES" --pdim 256 1 \
        --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --mesh-size 1024 1024 1024  1536 1536 1536  2048 2048 2048  3072 3072 3072 \
        --box-size 6000.0 6000.0 6000.0 \
        --nside 1024 \
        --t0 0.001 --nb-steps "$NB_STEPS" \
        --nb-shells "$NB_SHELLS" --shell-spacing growth \
        --drift-on-lightcone \
        --solver bf --lpt-order 1

    # 4096 (nside 1024, paint-nside 2048)
    fli-launcher simulate \
        --mode sbatch \
        --account "$ACCOUNT" --nodes "$NODES" --pdim 256 1 \
        --time-limit "$TIME_LIMIT" --slurm-script "$SLURM_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --mesh-size 4096 4096 4096 \
        --box-size 6000.0 6000.0 6000.0 \
        --nside 1024 --paint-nside 2048 \
        --t0 0.001 --nb-steps "$NB_STEPS" \
        --nb-shells "$NB_SHELLS" --shell-spacing growth \
        --drift-on-lightcone \
        --solver bf --lpt-order 1

done
