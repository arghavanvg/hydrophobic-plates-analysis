cd /gibbs/arghavan/gist_hydrophobic_plates/gist_results/
for dir in arghavan.omp_*_0.5.*; do
    newname=$(echo "$dir" | sed -E 's/^arghavan\.omp_([0-9.]+)_0\.5\..*/\1/')
    # echo mv "$dir" "$newname"
    mv "$dir" "$newname"
done