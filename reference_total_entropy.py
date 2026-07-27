#!/usr/bin/env python3
"""
Reference total-box GIST entropy totals to the 15.0 Å value.

Reads:
  /gibbs/arghavan/gist_hydrophobic_plates/gist_results/total_entropy_results.dat

Writes:
  /gibbs/arghavan/gist_hydrophobic_plates/gist_results/total_entropy_results_ref15.dat

TS_ref(d) = TS(d) - TS(15.0)
"""

from pathlib import Path

import pandas as pd

BASE_PATH = Path("/gibbs/arghavan/gist_hydrophobic_plates/gist_results/")
INPUT_FILE = BASE_PATH / "total_entropy_results.dat"
OUTPUT_FILE = BASE_PATH / "total_entropy_results_ref15.dat"
REFERENCE_DISTANCE = 15.0


def main() -> None:
    df = pd.read_csv(
        INPUT_FILE,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["Dist(A)", "TotalEntropy(kcal/mol)"],
    )

    ref_rows = df.loc[df["Dist(A)"] == REFERENCE_DISTANCE, "TotalEntropy(kcal/mol)"]
    if ref_rows.empty:
        raise ValueError(f"Reference distance {REFERENCE_DISTANCE} Å not found in {INPUT_FILE}")

    ref_value = float(ref_rows.iloc[0])
    df["TotalEntropy_ref15(kcal/mol)"] = df["TotalEntropy(kcal/mol)"] - ref_value

    with open(OUTPUT_FILE, "w") as f:
        f.write(
            "#Dist(A)\tTotalEntropy(kcal/mol)\tTotalEntropy_ref15(kcal/mol)\n"
        )
        df.to_csv(f, sep="\t", index=False, header=False, float_format="%.8f")

    print(f"Reference TS(15.0) = {ref_value:.8f} kcal/mol")
    print(f"Wrote referenced results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
