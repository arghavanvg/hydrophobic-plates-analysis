#!/bin/bash

module load gistpp

base_path='/gibbs/arghavan/gist_hydrophobic_plates/gist_results/'
results_file="${base_path}total_entropy_results.dat"

echo -e "#Dist(A)\tTotalEntropy(kcal/mol)" > "$results_file"

for i in $(seq 54 2 150); do
    distance_angstrom=$(awk -v i="$i" 'BEGIN {printf "%.1f", i/10}')
    work_dir="${base_path}${distance_angstrom}"

    if [[ ! -d "$work_dir" ]]; then
        echo "Skipping missing directory: $work_dir"
        continue
    fi

    echo "=== Processing d=${distance_angstrom} Å ==="
    cd "$work_dir" || continue

    gistpp -i d-${distance_angstrom}-dTStrans-dens-edge-removed.dx \
           -i2 d-${distance_angstrom}-dTSorient-dens-edge-removed.dx \
           -op add -o d-${distance_angstrom}-dTStot-dens-edge-removed.dx

    gistpp -i d-${distance_angstrom}-dTStot-dens-edge-removed.dx \
           -op multconst -opt const 0.125 \
           -o d-${distance_angstrom}-dTStotXVolume.dx

    sum_out=$(gistpp -i d-${distance_angstrom}-dTStotXVolume.dx -op sum)
    # typical line: "sum of: file.dx is: -12.345"
    total=$(echo "$sum_out" | awk '/sum of:/ {print $NF}')

    echo -e "${distance_angstrom}\t${total}" >> "$results_file"
    echo "$sum_out"
done

echo "Wrote results to $results_file"