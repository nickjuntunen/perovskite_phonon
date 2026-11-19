"""How many SQS cells per composition make the disorder noise negligible?

Data: the mixed-halide noise check (data/mixed_halides_noise_check_mp, 115
compositions x 5 SQS seeds, the production supercell; see
sqs_ensemble_average.py), read exactly as descriptor_selection.ipynb's noise
section does (dynamically stable realizations, symmetry-equivalent cells
pooled into counts, shape PCs in the MP basis).

Model. Each seed is an independent draw of an SQS cell, so the mean over N
cells has disorder error sigma_seed / sqrt(N), with sigma_seed^2 the seed
variance of a target at one composition (count-weighted, unbiased). Seeds that
came out symmetry-equivalent count as repeated draws, which is what they are.

Two criteria for "negligible", each solved for the N it needs:

    MLIP   sigma_seed / sqrt(N) <= --mlip-frac x the gauntlet MLIP noise
           (default 1/2: the disorder then adds <= 12% to the model noise in
           quadrature). MLIP noise = dataset std / gauntlet SNR from
           descriptor_scorecard.csv.
    series R2(series) ceiling 1 - (sigma_seed^2 / N) / V >= --ceiling
           (default 0.95), with V the within-series variance of the production
           mixed-halide values (series with > 1 composition), per halide pair
           and overall.

Each is given for the pooled variance (the mean over compositions: what the
notebook's ceiling uses) and for the 90th-percentile composition, overall and
per halide pair. A per-composition variance from 5 seeds has 4 degrees of
freedom, so its p90 overstates the true p90 (chi^2 scatter): read p90 as a
conservative bound, the pooled value as the typical one.

Not measured here: the finite-cell bias. Every seed uses the same supercell,
so averaging converges to that cell size's ensemble mean, not to the
infinite-disorder limit. Testing that needs a second, larger supercell.

Outputs, in --out (default data/mixed_halides_noise_check_mp/analysis):
    sqs_convergence.csv          per target: sigma_seed (pooled, p90), MLIP noise, V,
                                 ceiling and sigma/MLIP at N = 1, 2, 3, 5, 10, N needed
    sqs_convergence_by_pair.csv  N needed per target x halide pair

Usage:
    uv run python3 scripts/analysis/sqs_convergence.py
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

NOISE_DIR = PROJECT_ROOT / "data" / "mixed_halides_noise_check_mp"
N_GRID = (1, 2, 3, 5, 10)


def notebook_tables() -> dict:
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        g: dict = {}
        exec(open(PROJECT_ROOT / "gp_vs_tree" / "gp_setup.py").read(), g)
    finally:
        os.chdir(cwd)
    return g


def n_needed(var: float, limit_var: float) -> float:
    """Smallest N with var / N <= limit_var."""
    if not np.isfinite(var) or not np.isfinite(limit_var) or limit_var <= 0:
        return np.nan
    return float(max(1, int(np.ceil(var / limit_var - 1e-9))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-dir", type=Path, default=NOISE_DIR)
    parser.add_argument("--mlip-frac", type=float, default=0.5)
    parser.add_argument("--ceiling", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.noise_dir / "analysis"

    from scripts.analysis.sqs_ensemble_average import realizations
    g = notebook_tables()
    dm, Tm, phonon_targets = g["dm"], g["Tm"], g["phonon_targets"]
    shape_pca, n_shape, halide_targets = g["shape_pca"], g["n_shape"], g["HALIDE_TARGETS"]

    # --- the noise-check realizations, as the notebook reads them
    dn = realizations(args.noise_dir)
    n_runs = len(dn)
    dn = dn[dn["dynamically_stable"].astype(bool) & (dn["sublattice_unassigned"] == 0)].reset_index(drop=True)
    n_sqs = int(pd.read_csv(args.noise_dir / "manifest.csv")["seed"].max()) + 1
    dn["count"] = (dn["weight"] * n_sqs).round()
    Tn = phonon_targets(dn)
    nq = np.load(args.noise_dir / "analysis" / "design_descriptors_quantiles.npz")
    qmap = dict(zip(nq["keys"], nq["quantiles_cm1"]))
    Q = np.array([qmap[k] for k in dn["key"]])
    S = shape_pca.transform(Q / Q.mean(axis=1, keepdims=True))
    for k in range(n_shape):
        Tn[f"shape_pc{k + 1}"] = S[:, k]
    targets = [t for t in halide_targets if t in Tn]

    comp = pd.read_csv(args.noise_dir / "compositions.csv")
    pair_of = dict(zip(comp["formula"], comp["X1"] + "/" + comp["X2"]))
    dn["pair"] = dn["manifest_formula"].map(pair_of)
    per = dn.groupby("manifest_formula").agg(cells=("count", "size"), seeds=("count", "sum"))
    print(f"{len(dn)}/{n_runs} realizations stable, {len(per)} compositions; distinct cells per composition "
          f"(of {n_sqs} seeds, symmetry-equivalent pooled): {per['cells'].value_counts().sort_index().to_dict()}")

    def seed_var(v: pd.Series) -> pd.DataFrame:
        d = pd.DataFrame({"v": v, "c": dn["count"], "f": dn["manifest_formula"], "p": dn["pair"]}).dropna()
        n = d.groupby("f")["c"].sum()
        mean = (d["v"] * d["c"]).groupby(d["f"]).sum() / n
        ss = (d["c"] * (d["v"] - d["f"].map(mean)) ** 2).groupby(d["f"]).sum()
        res = pd.DataFrame({"var": (ss / (n - 1)).where(n > 1), "pair": d.groupby("f")["p"].first()})
        return res.dropna()

    # --- within-series variance of the production values, per pair and overall
    multi = dm.groupby("series")["formula"].transform("size") > 1
    dev = Tm.loc[multi, targets] - Tm.loc[multi, targets].groupby(dm.loc[multi, "series"]).transform("mean")
    V_all = (dev ** 2).mean()
    V_pair = (dev ** 2).groupby(dm.loc[multi, "pair"]).mean()

    sc = pd.read_csv(PROJECT_ROOT / "data" / "mlip_stable_polymorphs" / "descriptor_scorecard.csv", index_col=0)
    snr_col = [c for c in sc.columns if c.startswith("SNR gauntlet")][0]
    mlip = (sc["dataset_std"] / sc[snr_col]).reindex(targets)

    rows, prow = {}, []
    for t in targets:
        sv = seed_var(Tn[t])
        pooled, p90 = sv["var"].mean(), sv["var"].quantile(0.9)
        lim_m, lim_s = (args.mlip_frac * mlip[t]) ** 2, (1 - args.ceiling) * V_all[t]
        row = {"sigma_seed": np.sqrt(pooled), "sigma_seed_p90": np.sqrt(p90), "mlip_noise": mlip[t],
               "V_series_std": np.sqrt(V_all[t])}
        for n in N_GRID:
            row[f"ceiling_N{n}"] = 1 - pooled / n / V_all[t]
        for n in N_GRID:
            row[f"sigma/mlip_N{n}"] = np.sqrt(pooled / n) / mlip[t]
        row["N_mlip"], row["N_mlip_p90"] = n_needed(pooled, lim_m), n_needed(p90, lim_m)
        row["N_series"], row["N_series_p90"] = n_needed(pooled, lim_s), n_needed(p90, lim_s)
        row["N_needed"] = np.nanmax([row["N_mlip"], row["N_series"]])
        rows[t] = row
        for pair, grp in sv.groupby("pair"):
            pv = grp["var"].mean()
            prow.append({"target": t, "pair": pair, "n_comp": len(grp), "sigma_seed": np.sqrt(pv),
                         "N_mlip": n_needed(pv, lim_m),
                         "N_series": n_needed(pv, (1 - args.ceiling) * V_pair.loc[pair, t])
                         if pair in V_pair.index else np.nan})
    res = pd.DataFrame(rows).T.sort_values("N_needed")
    by_pair = pd.DataFrame(prow)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "sqs_convergence.csv")
    by_pair.to_csv(out / "sqs_convergence_by_pair.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["sigma_seed", "mlip_noise"] + [f"ceiling_N{n}" for n in N_GRID] + \
               ["N_mlip", "N_mlip_p90", "N_series", "N_series_p90", "N_needed"]
        print(f"\nN to make disorder negligible: sigma/sqrt(N) <= {args.mlip_frac} x MLIP noise (N_mlip), "
              f"R2(series) ceiling >= {args.ceiling} (N_series); pooled and p90 composition")
        print(res[cols].round(3).to_string())
        print("\nN_needed (pooled, max of both criteria) per halide pair:")
        piv = by_pair.assign(N=by_pair[["N_mlip", "N_series"]].max(axis=1)).pivot(index="target", columns="pair",
                                                                                   values="N")
        print(piv.reindex(res.index).to_string())
        n_comp = by_pair.drop_duplicates("pair").set_index("pair")["n_comp"].to_dict()
        print(f"compositions per pair: {n_comp}")
    print(f"-> {out}/sqs_convergence.csv, sqs_convergence_by_pair.csv")


if __name__ == "__main__":
    main()
