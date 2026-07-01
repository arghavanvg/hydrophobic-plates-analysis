import numpy as np
import pandas as pd

ints = list(range(54, 151, 2))
distances_in_A = [f"{d/10:.1f}" for d in ints]


base_path = '/gibbs/arghavan/gist_hydrophobic_plates/gist_results/'

for d in distances_in_A:


    input_file = f'{base_path}{d}/d-{d}-output.dat'
    output_file = f'{base_path}{d}/d-{d}-output-edge-eliminated.dat'

    with open(input_file) as f:
        lines = f.readlines()

    first_header = lines[0].rstrip()
    second_header = lines[1].rstrip()

    cols = second_header.split()

    df = pd.read_csv(
        input_file,
        sep=r'\s+',
        names=cols,
        skiprows=2,
        comment='#'
    )

    # -----------------------------------

    x_coords = (df["xcoord"] > 2.5) & (df["xcoord"] < 47.5)
    y_coords = (df["ycoord"] > 2.5) & (df["ycoord"] < 47.5)
    z_coords = (df["zcoord"] > 2.5) & (df["zcoord"] < 77.5)

    subset = df[x_coords & y_coords & z_coords]

    with open(output_file, "w") as f:
        f.write(first_header + " | Edge voxels within 2.5Å of each box boundary were excluded.\n")
        f.write(second_header + "\n")
        subset.to_csv(
            f,
            sep="\t",
            index=False,
            header=False,
            float_format="%.8f"
        )

