"""Pick the production mixed-halide series that the SQS disorder-noise check
repeats with several SQS seeds (generate_mixed_halides.py --mode noise-check
--source mp-clusters --series-csv ...).

A series is one cation framework and halide pair, x = 1/6 ... 5/6. Eligible:
every mixture of the series is a kept, dynamically stable production
composition, and both end members are stable MP-sweep references. That way
each realization has the production run (its sqs0) and the linear end-member
baseline to compare with, and the noise ceiling on R2(series) uses the same
rows descriptor_selection.ipynb scores. Picked at random within each stratum
(halide pair x single / double perovskite), plus the series the notebook's
series figures show. Frameworks whose parent structure has changed since the
production run are left out (EXCLUDE_FRAMEWORKS).

Output: <out>/series.csv (framework, X1, X2, pair, double, n_atoms).

Usage:
    uv run python3 scripts/dataset/select_noise_check_series.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MIXED_HALIDES_DIR, MIXED_HALIDES_NOISE_CHECK_MP_DIR
# (pair, double perovskite) -> series to pick; the stratum's size if it has fewer
QUOTA = {("Cl/Br", False): 4, ("Cl/Br", True): 8, ("Cl/I", False): 2, ("Cl/I", True): 4,
         ("Br/I", False): 1, ("Br/I", True): 5}
ALWAYS = ["Cs | Ag;Bi Cl/Br", "Cs | Na;In Cl/Br"]  # descriptor_selection.ipynb's example series
# The MP sweep's KCaCl3 pick changed after the production set was generated, so
# generate_mixed_halides.py now builds the Cl-majority K|Ca mixtures on another
# parent: their seed 0 would not be the production structure.
EXCLUDE_FRAMEWORKS = ["K | Ca"]


def eligible_series(mixed: Path) -> pd.DataFrame:
    comp = pd.read_csv(mixed / "compositions.csv")
    comp = comp[comp["status"] == "generated"].copy()
    sel = pd.read_csv(mixed / "analysis" / "selection" / "stable_polymorphs.csv")
    stable = sel.groupby("run_name")["dynamically_stable"].any()
    comp["stable"] = comp["structures"].map(stable).astype("boolean").fillna(False).astype(bool)
    comp["series"] = comp["framework"] + " " + comp["X1"] + "/" + comp["X2"]
    g = comp.groupby("series").agg(framework=("framework", "first"), X1=("X1", "first"), X2=("X2", "first"),
                                   n=("stable", "size"), n_stable=("stable", "sum"), n_atoms=("n_atoms", "max"),
                                   ref_X1=("ref_X1", "first"), ref_X2=("ref_X2", "first"))
    g["pair"] = g["X1"] + "/" + g["X2"]
    g["double"] = g["framework"].str.split(" | ", regex=False).str[1].str.contains(";")
    return g[(g["n_stable"] == g["n"]) & (g["ref_X1"] == "stable") & (g["ref_X2"] == "stable")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixed", type=Path, default=MIXED_HALIDES_DIR, help="The production set")
    parser.add_argument("--out", type=Path, default=MIXED_HALIDES_NOISE_CHECK_MP_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ok = eligible_series(args.mixed)
    ok = ok[~ok["framework"].isin(EXCLUDE_FRAMEWORKS)]
    picked = [s for s in ALWAYS if s in ok.index]
    for (pair, double), n in QUOTA.items():
        stratum = ok[(ok["pair"] == pair) & (ok["double"] == double) & ~ok.index.isin(picked)]
        have = sum(ok.loc[s, "pair"] == pair and ok.loc[s, "double"] == double for s in picked)
        k = max(0, min(n - have, len(stratum)))
        picked += list(stratum.sample(k, random_state=args.seed).index)
    out = ok.loc[picked, ["framework", "X1", "X2", "pair", "double", "n_atoms"]].sort_values(["pair", "double", "framework"])
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "series.csv", index_label="series")
    print(f"{len(ok)} eligible series, {len(out)} picked -> {args.out / 'series.csv'}")
    print(out.groupby(["pair", "double"]).size().to_string())


if __name__ == "__main__":
    main()
