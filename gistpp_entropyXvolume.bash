#!/bin/bash
#
# gistpp_entropyXvolume.bash
# --------------------------
# For each plate–plate distance (5.4–15.0 Å, step 0.2 Å), compute the
# total-box first-order GIST entropy with gistpp.
#
# Uses the original cpptraj dens maps (full grid). Do NOT use the
# edge-removed makedx outputs here: makedx expects a complete voxel
# table in grid order, so row-deleted edge-eliminated files misalign
# the dens maps and produce nonsense (~1e10) sums.
#
# Input dir per distance:
#   /gibbs/arghavan/gist_hydrophobic_plates/gist_results/<Å>/
#
# Steps per distance:
#   1. add dTStrans-dens + dTSorient-dens -> dTStot-dens
#   2. multconst by voxel volume (0.5^3 = 0.125) -> kcal/mol per voxel
#   3. sum over all voxels -> total TS (kcal/mol)
#
# After the loop, rewrite the results file with values referenced to
# the 15.0 Å total (TS_ref = TS(d) - TS(15.0)).
#
# Outputs:
#   per distance dir:
#     d-<Å>-dTStot-dens.dx
#     d-<Å>-dTStotXVolume.dx
#   results:
#     /gibbs/arghavan/gist_hydrophobic_plates/gist_results/total_entropy_results.dat
#

module load gistpp

base_path='/gibbs/arghavan/gist_hydrophobic_plates/gist_results/'
results_file="${base_path}total_entropy_results.dat"
voxel_volume='0.125'   # 0.5 Å grid spacing

echo -e "#Dist(A)\tTotalEntropy(kcal/mol)" > "$results_file"

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

    trans_dx="d-${distance_angstrom}-dTStrans-dens.dx"
    orient_dx="d-${distance_angstrom}-dTSorient-dens.dx"
    tot_dens_dx="d-${distance_angstrom}-dTStot-dens.dx"
    tot_dx="d-${distance_angstrom}-dTStotXVolume.dx"

    if [[ ! -f "$trans_dx" || ! -f "$orient_dx" ]]; then
        echo "Missing dens maps in $work_dir; skipping"
        continue
    fi

    gistpp -i "$trans_dx" \
           -i2 "$orient_dx" \
           -op add \
           -o "$tot_dens_dx"

    gistpp -i "$tot_dens_dx" \
           -op multconst -opt const "$voxel_volume" \
           -o "$tot_dx"

    sum_out=$(gistpp -i "$tot_dx" -op sum)
    # typical line: "sum of: file.dx is: -12.345"
    total=$(echo "$sum_out" | awk '/sum of:/ {print $NF}')

    echo -e "${distance_angstrom}\t${total}" >> "$results_file"
    echo "$sum_out"
done

# Reference totals to the 15.0 Å value: TS_ref(d) = TS(d) - TS(15.0)
awk '
BEGIN { OFS="\t"; n = 0 }
/^#/ { next }
NF >= 2 {
    n++
    dist[n] = $1
    val[n] = $2 + 0
    if ($1 + 0 == 15.0) ref = $2 + 0
}
END {
    print "#Dist(A)", "TotalEntropy(kcal/mol)", "TotalEntropy_ref15(kcal/mol)"
    for (i = 1; i <= n; i++) {
        print dist[i], val[i], val[i] - ref
    }
}
' "$results_file" > "${results_file}.tmp" && mv "${results_file}.tmp" "$results_file"

echo "Wrote results to $results_file"
