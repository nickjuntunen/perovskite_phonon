"""Ensemble statistics over the SQS realizations of each mixed composition:
join a noise-check set's descriptor table to its selection and manifest
weights, then average per composition (structure_gen.disorder.ensemble_average).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.structure_gen.disorder import ensemble_average

# identifiers and counts, not physical descriptors: never averaged
SKIP = {"n_A", "n_X", "sublattice_unassigned", "spacegroup_number"}


def realizations(noise_dir: Path) -> pd.DataFrame:
    """One row per selected realization of a noise-check set directory:
    descriptors (analysis/design_descriptors.csv), stability
    (analysis/selection/stable_polymorphs.csv) and manifest weight
    (manifest.csv). ValueError if a row has no weight."""
    design = pd.read_csv(noise_dir / "analysis" / "design_descriptors.csv")
    sel = pd.read_csv(noise_dir / "analysis" / "selection" / "stable_polymorphs.csv")
    manifest = pd.read_csv(noise_dir / "manifest.csv")[["name", "formula", "weight", "seed"]]
    sel["key"] = sel["run_name"] + "/" + sel["state"]
    d = design.merge(sel[["key", "dynamically_stable"]], on="key", how="inner")
    d = d.merge(manifest.rename(columns={"name": "run_name", "formula": "manifest_formula"}), on="run_name", how="left")
    if d["weight"].isna().any():
        raise ValueError(f"{d['weight'].isna().sum()} rows have no manifest weight: is --noise-dir the set the runs came from?")
    if "error" in d:
        d = d[d["error"].isna()]
    return d.reset_index(drop=True)


def average(d: pd.DataFrame, quantiles: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Weighted mean/std/sem of every numeric descriptor over each
    composition's dynamically stable realizations (rows of realizations()),
    plus the weighted-mean quantile function where every member has one.
    Returns (table, mean quantile functions, their composition keys)."""
    numeric = [c for c in d.select_dtypes("number").columns if c not in SKIP | {"weight", "seed"}]
    rows, q_mean, keys = [], [], []
    for formula, g in d.groupby("manifest_formula", sort=True):
        ok = g[g["dynamically_stable"].astype(bool)]
        row = {"formula": formula, "n_realizations": len(g), "n_stable": len(ok), "n_unstable": len(g) - len(ok),
               "weight_stable": ok["weight"].sum(), "runs": ";".join(ok["run_name"])}
        if len(ok):
            for c in numeric:
                v = ok[c].to_numpy(float)
                fin = np.isfinite(v)
                if fin.any():
                    stats = ensemble_average(v[fin], ok["weight"].to_numpy(float)[fin])
                    row |= {f"{c}_mean": stats["mean"], f"{c}_std": stats["std"], f"{c}_sem": stats["sem"]}
            row["n_eff"] = ensemble_average(np.zeros(len(ok)), ok["weight"])["n_eff"]
            if all(k in quantiles for k in ok["key"]):
                w = ok["weight"].to_numpy(float)
                q_mean.append(np.average(np.stack([quantiles[k] for k in ok["key"]]), axis=0, weights=w))
                keys.append(formula)
        rows.append(row)
    return pd.DataFrame(rows), np.array(q_mean), keys
