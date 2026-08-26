#!/bin/bash

base_path='/gibbs/arghavan/gist_hydrophobic_plates/gist_results/'

# Distances: 5.4, 5.6, ..., 15.0 Å
for i in $(seq 54 2 150); do
    distance_angstrom=$(awk -v i="$i" 'BEGIN {printf "%.1f", i/10}')
    work_dir="${base_path}${distance_angstrom}"

    if [[ ! -d "$work_dir" ]]; then
        echo "Skipping missing directory: $work_dir"
        continue
    fi

    echo "=== Cleaning d=${distance_angstrom} Å ==="

    find "$work_dir" -maxdepth 1 -type f \( \
        -name '*-edge-removed.dx' -o \
        -name '*-edge-eliminated.out' \
    \) -delete
done