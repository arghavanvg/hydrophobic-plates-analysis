#!/bin/bash
#
# makedx.bash
# -----------
# For each plate–plate distance (5.4–15.0 Å, step 0.2 Å), convert the
# edge-eliminated GIST .dat table into .dx maps for selected columns via
# gistpp makedx (column const indices match the GIST output header).
#
# Input dir per distance:
#   /gibbs/arghavan/gist_hydrophobic_plates/gist_results/<Å>/
#
# Steps per distance:
#   1. Rename d-<Å>-output-edge-eliminated.dat -> .out (gistpp input)
#   2. Write edge-removed .dx for:
#        gO (const 05), dTStrans-dens (08), dTSorient-dens (10),
#        dTSsix-dens (12), neighbor-dens (24), neighbor-norm (25)
#
# Outputs (in each distance directory):
#   d-<Å>-*-edge-removed.dx
#

module load gistpp

base_path='/gibbs/arghavan/gist_hydrophobic_plates/gist_results/'

# ints = range(54, 151, 2) -> distances 5.4, 5.6, ..., 15.0 Å
for i in $(seq 54 2 150); do
    distance_angstrom=$(awk -v i="$i" 'BEGIN {printf "%.1f", i/10}')
    work_dir="${base_path}${distance_angstrom}"

    if [[ ! -d "$work_dir" ]]; then
        echo "Skipping missing directory: $work_dir"
        continue
    fi

    echo "=== Processing d=${distance_angstrom} Å ==="
    cd "$work_dir" || continue

    mv "d-${distance_angstrom}-output-edge-eliminated.dat" "d-${distance_angstrom}-output-edge-eliminated.out"

    gistpp -i "d-${distance_angstrom}-output-edge-eliminated.out" \
           -i2 "d-${distance_angstrom}-gO.dx" \
           -op makedx -opt const 05 \
           -o "d-${distance_angstrom}-gO-edge-removed.dx"

    gistpp -i "d-${distance_angstrom}-output-edge-eliminated.out" \
           -i2 "d-${distance_angstrom}-dTStrans-dens.dx" \
           -op makedx -opt const 08 \
           -o "d-${distance_angstrom}-dTStrans-dens-edge-removed.dx"

    gistpp -i "d-${distance_angstrom}-output-edge-eliminated.out" \
           -i2 "d-${distance_angstrom}-dTSorient-dens.dx" \
           -op makedx -opt const 10 \
           -o "d-${distance_angstrom}-dTSorient-dens-edge-removed.dx"

    gistpp -i "d-${distance_angstrom}-output-edge-eliminated.out" \
           -i2 "d-${distance_angstrom}-dTSsix-dens.dx" \
           -op makedx -opt const 12 \
           -o "d-${distance_angstrom}-dTSsix-dens-edge-removed.dx"

    gistpp -i "d-${distance_angstrom}-output-edge-eliminated.out" \
           -i2 "d-${distance_angstrom}-neighbor-norm.dx" \
           -op makedx -opt const 24 \
           -o "d-${distance_angstrom}-neighbor-dens-edge-removed.dx"

    gistpp -i "d-${distance_angstrom}-output-edge-eliminated.out" \
           -i2 "d-${distance_angstrom}-neighbor-norm.dx" \
           -op makedx -opt const 25 \
           -o "d-${distance_angstrom}-neighbor-norm-edge-removed.dx"
done
