"""Composition features (src/analysis/composition_features.py) for every
structure in an MP metadata CSV, from its formula and its structural site
assignment (a_sites/b_sites/x_site, a_site_cn/b_site_cn).

This replaces data/mp_composition_clusters.csv as the source of tier-C inputs
for descriptor_selection.ipynb. That file holds only the structures
cluster_compositions.py could build a full descriptor vector for, which left
the 1190 structures added to the MLIP sweep without any composition inputs at
all (no masses even when only an electronegativity was missing). Values agree
with it wherever it has a row.

Rows whose structural check left a_sites or b_sites empty (404 of 3949, e.g.
Ca2CrWO6, LiOsO3) get their sites from design_descriptors.assign_sublattices
on the MP structure -- the ECoN rule extract_design_descriptors.py also falls
back on -- with the default site CNs (site_source = "econ").

--compositions takes a generated set's compositions.csv instead
(scripts/dataset/generate_mixed_halides.py): sites from its A / B / X1 / X2
columns, the X-site means weighted by the formula's halide counts, and
bond valence on the generated cell. Rows are keyed by the "structures"
column, the MLIP run_name.

Output: one row per metadata row, keyed by name = "<material_id>_<formula>"
(the MLIP sweep's run_name), with FEATURE_NAMES and the site/oxi/radius/
en_source columns.

Usage:
    uv run python3 scripts/analysis/extract_composition_features.py
    # -> data/mp_perovskites/composition_features.csv
    uv run python3 scripts/analysis/extract_composition_features.py \\
        --compositions data/mixed_halides/compositions.csv --out-csv data/mixed_halides/analysis/composition_features.csv
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.composition_features import generated_feature_rows, mp_feature_rows
from src.paths import MP_METADATA_STABLE



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=MP_METADATA_STABLE)
    parser.add_argument("--compositions", type=Path, default=None,
                        help="A generated set's compositions.csv (generate_mixed_halides.py), instead of --metadata")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Default: composition_features.csv next to --metadata / --compositions")
    args = parser.parse_args()
    source = args.compositions or args.metadata

    rows = generated_feature_rows(args.compositions) if args.compositions else mp_feature_rows(args.metadata)
    out = pd.DataFrame(rows)

    out_csv = args.out_csv or source.with_name("composition_features.csv")
    out.to_csv(out_csv, index=False)
    print(f"{len(out)} structures -> {out_csv}")
    print("missing fraction:", out.drop(columns="name").isna().mean().loc[lambda s: s > 0].round(3).to_dict())
    for col in ["site_source", "oxi_source", "radius_source", "en_source"]:
        values = out[col].str.replace(r"^guess[1-9]\d*$", "guess>0", regex=True) if col == "oxi_source" else out[col]
        print(f"{col}: {values.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
