module load gistpp

base_path='/gibbs/arghavan/gist_hydrophobic_plates/gist_results/'
output_file="${base_path}average_nbrs.dat"

# Create the output file and write the header
echo -e "distance(A)\tNum_nbrs" > "$output_file"

# Distances: 5.4, 5.6, ..., 15.0 Å
for i in $(seq 54 2 150); do
    distance_angstrom=$(awk -v i="$i" 'BEGIN {printf "%.1f", i/10}')
    work_dir="${base_path}${distance_angstrom}"

    if [[ ! -d "$work_dir" ]]; then
        echo "Skipping missing directory: $work_dir"
        continue
    fi

    echo "=== Processing d=${distance_angstrom} Å ==="
    cd "$work_dir" || continue

    # Run gistpp and capture its output
    result=$(gistpp -i "d-${distance_angstrom}-neighbor-norm.dx" -op sum)

    # Extract the average value
    avg_nbrs=$(echo "$result" | awk '/avg of:/ {print $NF}')

    # Save distance and average number of neighbors
    echo -e "${distance_angstrom}\t${avg_nbrs}" >> "$output_file"

    echo "Average number of neighbors: $avg_nbrs"
done

echo "Results saved to: $output_file"