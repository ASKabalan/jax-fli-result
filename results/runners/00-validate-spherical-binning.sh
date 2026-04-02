#!/bin/bash
# 00-validate-spherical-binning: Spherical painting method validation
#
# Tests all painting schemes (NGP, BILINEAR, RBF×4) across 7 mesh/box/nside configs:
#
#   Group A — varied scale (nside matches mesh scale):
#     512³  / box 500  / nside 512
#     1024³ / box 1000 / nside 1024
#     2048³ / box 2000 / nside 2048
#     4096³ / box 4000 / nside 4096
#
#   Group B — fixed box 2000, fixed nside 1024 (3 meshes, 1 fli-launcher call each):
#     512³  / box 2000 / nside 1024
#     1024³ / box 2000 / nside 1024
#     4096³ / box 2000 / nside 1024
#
# Each group × 6 painting schemes × 2 upgrade options (with/without paint-nside)
# × 2 solvers (KDK 30 steps / BF 15 steps) = 168 SLURM jobs.
#
# Internal gridding is used for Group B: all 3 meshes are submitted in a single
# fli-launcher call (producing 3 separate SLURM jobs), reducing the total number
# of bash invocations from 168 to 120.
#
# Usage: SLURM_SCRIPT=/path/to/script.sh bash 00-validate-spherical-binning.sh

set -euo pipefail
SLURM_SCRIPT=${SLURM_SCRIPT:-"scripts/slurm/default.sh"}

ACCOUNT=tkc@h100
TIME_LIMIT="01:00:00"
OUTPUT_DIR="results/00-validate-spherical-binning"

# ---------------------------------------------------------------------------
# Group A: per-config arrays (indices 0-3)
# ---------------------------------------------------------------------------
MESHES_A=(512  1024 2048 4096)
BOXES_A=( 500  1000 2000 4000)
NSIDES_A=(512  1024 2048 4096)
NODES_A=( 4    8    32   64  )
PDIMX_A=( 16   32   128  256 )

# ---------------------------------------------------------------------------
# Painting schemes (indices 0-5)
# name tag used in --output-dir suffix
# ---------------------------------------------------------------------------
SCHEMES=(  ngp      bilinear rbf_neighbor rbf_neighbor rbf_neighbor rbf_neighbor)
KERNELS=(  ""       ""       ""           "0.1"        "15.0"       "20.0"      )
STAGS=(    ngp      bilinear rbf_none     rbf_0.1      rbf_15.0     rbf_20.0    )

# ---------------------------------------------------------------------------
# Helper: emit one fli-launcher simulate call
#   _run <label> <nodes> <pdimx> <mesh_args> <box_args> <nside_args> <paint_args>
#        <scheme_args> <solver> <steps> <lpt> <spacing>
# We pass extra args as positional words; quoting via arrays is handled inline.
# ---------------------------------------------------------------------------

for si in 0 1 2 3 4 5; do
    SCHEME="${SCHEMES[$si]}"
    KW="${KERNELS[$si]}"
    STAG="${STAGS[$si]}"

    # Build kernel-width arg array (empty when no kernel)
    if [[ -n "$KW" ]]; then
        KW_ARGS=(--kernel-width-arcmin "$KW")
    else
        KW_ARGS=()
    fi

    for SOLVER_IDX in 0 1; do
        if [[ $SOLVER_IDX -eq 0 ]]; then
            SOLVER=kdk; STEPS=30; LPT=2; SPACING=a;      T0=0.001; STAG_SOL=KDK
        else
            SOLVER=bf;  STEPS=15; LPT=1; SPACING=growth; T0=0.001; STAG_SOL=BF
        fi

        SOL_DIR="$OUTPUT_DIR/$STAG_SOL"

        # ----------------------------------------------------------------
        # Group A — without upgrade (no --paint-nside)
        # ----------------------------------------------------------------
        for ci in 0 1 2 3; do
            M="${MESHES_A[$ci]}"
            B="${BOXES_A[$ci]}"
            N="${NSIDES_A[$ci]}"
            ND="${NODES_A[$ci]}"
            PX="${PDIMX_A[$ci]}"

            echo "=== A-no-upgrade  M=${M} B=${B} N=${N} scheme=${STAG} solver=${STAG_SOL} ==="
            fli-launcher simulate \
                --mode sbatch \
                --account "$ACCOUNT" \
                --nodes "$ND" --pdim "$PX" 1 \
                --time-limit "$TIME_LIMIT" \
                --slurm-script "$SLURM_SCRIPT" \
                --output-dir "${SOL_DIR}/${STAG}" \
                --mesh-size "$M" "$M" "$M" \
                --box-size "${B}.0" "${B}.0" "${B}.0" \
                --nside "$N" \
                --scheme "$SCHEME" \
                "${KW_ARGS[@]}" \
                --t0 "$T0" \
                --nb-steps "$STEPS" \
                --nb-shells 5 \
                --shell-spacing "$SPACING" \
                --min-width 10.0 \
                --drift-on-lightcone \
                --solver "$SOLVER" \
                --lpt-order "$LPT" \
                --simulation-type nbody
        done

        # ----------------------------------------------------------------
        # Group A — with upgrade (--paint-nside = 2 × nside)
        # ----------------------------------------------------------------
        for ci in 0 1 2 3; do
            M="${MESHES_A[$ci]}"
            B="${BOXES_A[$ci]}"
            N="${NSIDES_A[$ci]}"
            PN=$(( N * 2 ))
            ND="${NODES_A[$ci]}"
            PX="${PDIMX_A[$ci]}"

            echo "=== A-upgrade     M=${M} B=${B} N=${N} PN=${PN} scheme=${STAG} solver=${STAG_SOL} ==="
            fli-launcher simulate \
                --mode sbatch \
                --account "$ACCOUNT" \
                --nodes "$ND" --pdim "$PX" 1 \
                --time-limit "$TIME_LIMIT" \
                --slurm-script "$SLURM_SCRIPT" \
                --output-dir "${SOL_DIR}/${STAG}_upg" \
                --mesh-size "$M" "$M" "$M" \
                --box-size "${B}.0" "${B}.0" "${B}.0" \
                --nside "$N" --paint-nside "$PN" \
                --scheme "$SCHEME" \
                "${KW_ARGS[@]}" \
                --t0 "$T0" \
                --nb-steps "$STEPS" \
                --nb-shells 5 \
                --shell-spacing "$SPACING" \
                --min-width 10.0 \
                --drift-on-lightcone \
                --solver "$SOLVER" \
                --lpt-order "$LPT" \
                --simulation-type nbody
        done

        # ----------------------------------------------------------------
        # Group B — without upgrade (3 meshes, fixed box=2000, nside=1024)
        # Internal gridding: 1 fli-launcher call → 3 SLURM jobs
        # ----------------------------------------------------------------
        echo "=== B-no-upgrade  M=512/1024/4096 B=2000 N=1024 scheme=${STAG} solver=${STAG_SOL} ==="
        fli-launcher simulate \
            --mode sbatch \
            --account "$ACCOUNT" \
            --nodes 64 --pdim 256 1 \
            --time-limit "$TIME_LIMIT" \
            --slurm-script "$SLURM_SCRIPT" \
            --output-dir "${SOL_DIR}/${STAG}_B2000" \
            --mesh-size 512  512  512  1024 1024 1024  4096 4096 4096 \
            --box-size  2000.0 2000.0 2000.0 \
            --nside 1024 \
            --scheme "$SCHEME" \
            "${KW_ARGS[@]}" \
            --t0 "$T0" \
            --nb-steps "$STEPS" \
            --nb-shells 5 \
            --shell-spacing "$SPACING" \
            --min-width 10.0 \
            --drift-on-lightcone \
            --solver "$SOLVER" \
            --lpt-order "$LPT" \
            --simulation-type nbody

        # ----------------------------------------------------------------
        # Group B — with upgrade (paint-nside=2048, target nside=1024)
        # Internal gridding: 1 fli-launcher call → 3 SLURM jobs
        # ----------------------------------------------------------------
        echo "=== B-upgrade     M=512/1024/4096 B=2000 N=1024 PN=2048 scheme=${STAG} solver=${STAG_SOL} ==="
        fli-launcher simulate \
            --mode sbatch \
            --account "$ACCOUNT" \
            --nodes 64 --pdim 256 1 \
            --time-limit "$TIME_LIMIT" \
            --slurm-script "$SLURM_SCRIPT" \
            --output-dir "${SOL_DIR}/${STAG}_B2000_upg" \
            --mesh-size 512  512  512  1024 1024 1024  4096 4096 4096 \
            --box-size  2000.0 2000.0 2000.0 \
            --nside 1024 --paint-nside 2048 \
            --scheme "$SCHEME" \
            "${KW_ARGS[@]}" \
            --t0 "$T0" \
            --nb-steps "$STEPS" \
            --nb-shells 5 \
            --shell-spacing "$SPACING" \
            --min-width 10.0 \
            --drift-on-lightcone \
            --solver "$SOLVER" \
            --lpt-order "$LPT" \
            --simulation-type nbody

    done  # solver loop
done  # scheme loop
