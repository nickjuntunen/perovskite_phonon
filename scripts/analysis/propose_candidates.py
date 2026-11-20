"""Proposal set for the prospective (inverse-design) validation: given target
values on the core phonon axes, which never-computed compositions does the
forward model expect to reach them? The proposals then go through the MLIP
phonon pipeline, and the measured descriptors are compared with the targets.

Core targets (descriptor_selection.md section 6), in MLIP-noise units
(data/target_clusters/noise_scale.csv, the gauntlet spread across 8 MLIPs):

    log_omega_log, rel_centroid_A, rel_width_A, shape_pc1, rel_centroid_B

log_distortion_energy is left out: it needs a relaxation (C-tier R2w 0.59).

Candidate pool (default). Structurally verified MP perovskites
(metadata_verified.csv) that the MP sweep never ran: above its 0.05 eV/atom
hull cut but within --max-e-above-hull (default 0.1), with a formula the sweep
never computed (otherwise the C-tier inputs equal a training row's and the
"prediction" is a lookup). X site one of O S Se Te N F Cl Br I, and oxidation
states resolved (no ionic picture, no meaningful C-tier input). They are
real structures, so they can go straight into run_mlip_phonons.py.

Forward model. Candidates have no relaxed geometry, so the inputs are the
notebook's C tier (composition + site assignment; input_features() of
descriptor_selection.ipynb, through gp_vs_tree/gp_setup.py). The pool's
site-heterogeneity inputs come from the same design_descriptors functions
on the MP structure. One GP per target, as gp_vs_tree/gp_try.py: constant x
ARD-RBF + white kernel, hyperparameters optimized on --n-opt training rows,
then conditioned on all of them. Training: every kept MP polymorph and
mixed-halide composition.

Targets. --n-targets observed target vectors, so every target is attainable:
the medoids of KMeans on the noise-scaled targets of the MP training rows
with an anion on the X site, the pool's chemistry (spread over the populated
part of target space), or --targets-csv
(columns = the five targets, one row per target, optional "target_id").

Ranking. For candidate c and target t, the GP predictive N(mu, sigma^2) on each
axis gives p_j = P(|y_j - t_j| <= --tol-rank * noise_j); the score is
sum_j log p_j (independent axes), so a confident near-miss and an uncertain
near-hit are both weighed. Each target takes its --n-per-target best
candidates with distinct formulas, and a formula already proposed for another
target is skipped. --n-controls random pool members (stratified by X anion)
are added as the baseline: the same comparison on them gives the hit rate
without the model, and all proposals double as a forward-model test on new
chemistry.

A hit (scored after the MLIP runs) is |y_j - t_j| <= tol * noise_j on every
axis (Chebyshev, as cluster_targets.py), reported at several tol.

--retrospective runs the same procedure with known answers before spending
GPU time. For each chemical cluster k (data/mp_perovskites/composition_clusters.csv,
via polymorph_clusters.csv's chem_cluster), train on the MP polymorphs
outside k, take --n-targets random members of k as targets (each target's own
formula removed from the pool), and propose from the rest of k. It reports
hit rates of the top --n-per-target vs the pool's base rate, i.e. the
enrichment the model buys, and the median Chebyshev miss.

Outputs (prospective), in --out (default data/proposals/<name>):
    targets.csv            target vectors (+ the medoid's run_name)
    pool.csv               the candidate pool with predictions (mu, sigma per target)
    proposals.csv          one row per proposed structure: role (proposal / control),
                           target_id, rank, predicted mu/sigma, per-target score
    structures/            the proposals' MP CIFs
    mlip_structures.txt    for run_mlip_phonons.py --structure-list (largest cells first)
Outputs (--retrospective): <out>/retrospective.csv (per cluster x target x tol) and
    retrospective_summary.csv.

Usage:
    uv run python3 scripts/analysis/propose_candidates.py --retrospective --jobs 8
    uv run python3 scripts/analysis/propose_candidates.py --jobs 8     # -> data/proposals/mp_pool_v1/
"""

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm

CORE_TARGETS = ["log_omega_log", "rel_centroid_A", "rel_width_A", "shape_pc1", "rel_centroid_B"]
ANIONS = {"O", "S", "Se", "Te", "N", "F", "Cl", "Br", "I"}
HIT_TOLS = (2.0, 3.0, 5.0)
SEED = 0


# ------------------------------------------------------------------ data

def notebook_tables() -> dict:
    """df / T / X_all (MP) and dm / Tm / Xm (mixed halides), C_FEATURES and
    input_features, from the notebook cells gp_setup.py executes."""
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        g: dict = {}
        exec(open(PROJECT_ROOT / "gp_vs_tree" / "gp_setup.py").read(), g)
    finally:
        os.chdir(cwd)
    return g


def build_pool(out_dir: Path, max_e_hull: float, swept_formulas: set[str]) -> pd.DataFrame:
    """The uncomputed MP pool with the notebook's raw input columns (before
    input_features): composition features + site heterogeneity on the MP
    structure. Cached as <out_dir>/pool_raw.csv."""
    cache = out_dir / "pool_raw.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import ase.io
    from src.analysis.composition_features import mp_feature_rows
    from src.analysis.design_descriptors import assign_sublattices, oxidation_states, site_heterogeneity
    from src.paths import MP_DIR, MP_METADATA_STABLE, MP_METADATA_VERIFIED
    from pymatgen.core import Composition

    v = pd.read_csv(MP_METADATA_VERIFIED)
    swept = pd.read_csv(MP_METADATA_STABLE)
    v = v[(v["verified_perovskite"].astype(str) == "True") & ~v["material_id"].isin(swept["material_id"])
          & (v["energy_above_hull"] <= max_e_hull) & v["x_site"].isin(ANIONS)]
    v = v[~v["formula_pretty"].map(lambda f: Composition(f).reduced_formula).isin(swept_formulas)]
    v = v.assign(cif_path=[str(MP_DIR / c) for c in v["cif_path"]])  # relative to MP_DIR; absolute here
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "pool_metadata.csv"
    v.to_csv(meta_path, index=False)

    comp = pd.DataFrame(mp_feature_rows(meta_path))
    rows = []
    for _, r in v.iterrows():
        name = f"{r['material_id']}_{r['formula_pretty']}"
        row = {"name": name, "material_id": r["material_id"], "formula_pretty": r["formula_pretty"],
               "nsites": r["nsites"], "energy_above_hull": r["energy_above_hull"], "x_site": r["x_site"],
               "cif": r["cif_path"]}
        try:
            atoms = ase.io.read(row["cif"])
            oxi = oxidation_states(Composition(atoms.get_chemical_formula()).reduced_formula)
            split = lambda s: set(str(s).split(";")) - {"", "nan"} if isinstance(s, str) else set()
            roles = assign_sublattices(atoms, oxi, split(r["a_sites"]), split(r["b_sites"]), r["x_site"])
            row.update(site_heterogeneity(atoms, roles, oxi))
            row.update({"n_A": int((roles == "A").sum()), "n_B": int((roles == "B").sum()),
                        "sublattice_unassigned": int((roles == "?").sum())})
        except Exception as exc:  # noqa: BLE001 -- recorded, filtered below
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    pool = pd.DataFrame(rows).merge(comp, on="name", how="left")
    pool.to_csv(cache, index=False)
    return pool


# ------------------------------------------------------------------ model

def fit_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, n_opt: int):
    """GP (gp_try.py's kernel) -> (mu, sigma) on X_te, in target units."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = y_tr.notna().to_numpy()
    X_tr, y_tr = X_tr[keep], y_tr[keep]
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_tr)
    Z_tr, Z_te = prep.transform(X_tr), prep.transform(X_te)
    idx = np.random.default_rng(SEED).choice(len(y_tr), min(n_opt, len(y_tr)), replace=False)
    kernel = ConstantKernel(1.0) * RBF(np.full(Z_tr.shape[1], 3.0), (1e-2, 1e3)) + WhiteKernel(0.1, (1e-4, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_fit = GaussianProcessRegressor(kernel, normalize_y=True, random_state=SEED).fit(
            Z_tr[idx], y_tr.iloc[idx]).kernel_
        gp = GaussianProcessRegressor(k_fit, normalize_y=True, optimizer=None).fit(Z_tr, y_tr)
        mu, sd = gp.predict(Z_te, return_std=True)
    return mu, sd


def predict_all(X_tr, Y_tr, X_te, n_opt, jobs) -> tuple[pd.DataFrame, pd.DataFrame]:
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=jobs)(delayed(fit_predict)(X_tr, Y_tr[t], X_te, n_opt) for t in CORE_TARGETS)
    mu = pd.DataFrame({t: r[0] for t, r in zip(CORE_TARGETS, res)}, index=X_te.index)
    sd = pd.DataFrame({t: r[1] for t, r in zip(CORE_TARGETS, res)}, index=X_te.index)
    return mu, sd


def scores(mu: pd.DataFrame, sd: pd.DataFrame, target: pd.Series, noise: pd.Series, tol: float) -> pd.Series:
    """sum_j log P(|y_j - t_j| <= tol * noise_j), y_j ~ N(mu_j, sd_j^2)."""
    half = tol * noise
    p = norm.cdf((target + half - mu) / sd) - norm.cdf((target - half - mu) / sd)
    return pd.Series(np.log(np.clip(np.asarray(p, float), 1e-300, None)).sum(axis=1), index=mu.index)


def chebyshev(Y: pd.DataFrame, target: pd.Series, noise: pd.Series) -> pd.Series:
    return ((Y[CORE_TARGETS] - target) / noise).abs().max(axis=1)


def medoid_targets(Y: pd.DataFrame, names: pd.Series, noise: pd.Series, k: int) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    Z = (Y[CORE_TARGETS] / noise).to_numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    rows = []
    for c in range(k):
        m = np.flatnonzero(km.labels_ == c)
        i = m[np.argmin(((Z[m] - km.cluster_centers_[c]) ** 2).sum(axis=1))]
        rows.append({"target_id": f"t{c}", "medoid_run_name": names.iloc[i], "cluster_size": len(m),
                     **Y[CORE_TARGETS].iloc[i].to_dict()})
    return pd.DataFrame(rows)


def pick(score: pd.Series, formulas: pd.Series, n: int, taken: set[str]) -> list:
    out = []
    for i in score.sort_values(ascending=False).index:
        f = formulas.loc[i]
        if f in taken:
            continue
        out.append(i)
        taken.add(f)
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------ modes

def retrospective(g: dict, noise: pd.Series, args) -> None:
    df, T = g["df"], g["T"]
    X = g["X_all"][g["C_FEATURES"]]
    ok = T[CORE_TARGETS].notna().all(axis=1) & df["chem_cluster"].notna()
    rng = np.random.default_rng(SEED)
    rows = []
    for k in sorted(df.loc[ok, "chem_cluster"].astype(int).unique()):
        held = ok & (df["chem_cluster"] == k)
        if held.sum() < 3 * args.n_per_target:
            print(f"cluster {k}: {held.sum()} members, too few -- skipped")
            continue
        t0 = time.time()
        tr = ~(df["chem_cluster"] == k)
        mu, sd = predict_all(X[tr], T.loc[tr, CORE_TARGETS], X[held], args.n_opt, args.jobs)
        Yk = T.loc[held, CORE_TARGETS]
        fk = df.loc[held, "formula"]
        for j, ti in enumerate(rng.choice(Yk.index, min(args.n_targets, len(Yk)), replace=False)):
            target = Yk.loc[ti]
            cand = fk.index[fk != fk.loc[ti]]  # another composition must reach it
            s = scores(mu.loc[cand], sd.loc[cand], target, noise, args.tol_rank)
            top = pick(s, fk, args.n_per_target, set())
            d_all = chebyshev(Yk.loc[cand], target, noise)
            d_top = d_all.loc[top]
            for tol in HIT_TOLS:
                base = float((d_all <= tol).mean())
                rate = float((d_top <= tol).mean())
                rows.append({"cluster": k, "n_cluster": int(held.sum()), "target": ti, "tol": tol,
                             "base_rate": base, "hit_rate_top": rate,
                             "median_miss_top": float(d_top.median()), "median_miss_pool": float(d_all.median())})
        print(f"cluster {k}: {held.sum()} members, {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "retrospective.csv", index=False)
    summ = res.groupby("tol").agg(base_rate=("base_rate", "mean"), hit_rate_top=("hit_rate_top", "mean"),
                                  median_miss_top=("median_miss_top", "median"),
                                  median_miss_pool=("median_miss_pool", "median"),
                                  n=("target", "size"))
    summ["enrichment"] = summ["hit_rate_top"] / summ["base_rate"]
    summ.to_csv(args.out / "retrospective_summary.csv")
    with pd.option_context("display.width", 200):
        print(f"\nheld-out chemical cluster, top {args.n_per_target} per target (Chebyshev, noise units):")
        print(summ.round(3).to_string())
        per = res[res["tol"] == 3.0].groupby("cluster")[["base_rate", "hit_rate_top", "median_miss_top",
                                                          "median_miss_pool"]].mean()
        print("\nper cluster, tol 3:")
        print(per.round(3).to_string())
    print(f"-> {args.out}")


def prospective(g: dict, noise: pd.Series, args) -> None:
    df, T, dm, Tm = g["df"], g["T"], g["dm"], g["Tm"]
    C = g["C_FEATURES"]
    swept = set(df["formula"]) | set(pd.read_csv(PROJECT_ROOT / "data" / "mlip_stable_polymorphs" /
                                                 "candidates.csv", usecols=["formula"])["formula"])
    pool = build_pool(args.out, args.max_e_above_hull, swept)
    n0 = len(pool)
    if "error" in pool:
        pool = pool[pool["error"].isna()]
    pool = pool[(pool["sublattice_unassigned"] == 0) & (pool["n_A"] > 0) & (pool["n_B"] > 0)
                & pool["oxidation_A"].notna()].reset_index(drop=True)
    print(f"pool: {len(pool)}/{n0} structures, {pool['formula_pretty'].nunique()} formulas; "
          f"X: {pool['x_site'].value_counts().to_dict()}")
    for c in g["G_FEATURES"] + g["I_FEATURES"]:
        pool[c] = np.nan
    Xp = g["input_features"](pool)[C]

    X_tr = pd.concat([g["X_all"][C], g["Xm"][C]], ignore_index=True)
    Y_tr = pd.concat([T[CORE_TARGETS], Tm[CORE_TARGETS]], ignore_index=True)
    t0 = time.time()
    mu, sd = predict_all(X_tr, Y_tr, Xp, args.n_opt, args.jobs)
    print(f"GP fits: {time.time() - t0:.0f} s; median predictive sigma in noise units: "
          f"{(sd / noise).median().round(2).to_dict()}")

    if args.targets_csv:
        targets = pd.read_csv(args.targets_csv)
        if "target_id" not in targets:
            targets["target_id"] = [f"t{i}" for i in range(len(targets))]
    else:
        ionic = df["mp_x_site"].isin(ANIONS) & T[CORE_TARGETS].notna().all(axis=1)  # the pool's chemistry
        targets = medoid_targets(T[ionic], df.loc[ionic, "run_name"], noise, args.n_targets)
    pool_out = pool[["name", "material_id", "formula_pretty", "x_site", "nsites", "energy_above_hull", "cif"]].copy()
    for t in CORE_TARGETS:
        pool_out[f"mu_{t}"], pool_out[f"sigma_{t}"] = mu[t], sd[t]

    taken: set[str] = set()
    picked = []
    for _, tr in targets.iterrows():
        target = tr[CORE_TARGETS].astype(float)
        s = scores(mu, sd, target, noise, args.tol_rank)
        pool_out[f"score_{tr['target_id']}"] = s
        pred_miss = chebyshev(mu, target, noise)
        for rank, i in enumerate(pick(s, pool["formula_pretty"], args.n_per_target, taken)):
            picked.append({"name": pool.loc[i, "name"], "role": "proposal", "target_id": tr["target_id"],
                           "rank": rank, "score": s.loc[i], "predicted_miss": pred_miss.loc[i]})
    rng = np.random.default_rng(SEED)
    rest = pool[~pool["formula_pretty"].isin(taken)]
    by_x = rest.groupby("x_site").size()
    quota = (by_x / by_x.sum() * args.n_controls).round().astype(int)
    for x, q in quota.items():
        idx = rest.index[rest["x_site"] == x]
        for i in rng.choice(idx, min(q, len(idx)), replace=False):
            picked.append({"name": pool.loc[i, "name"], "role": "control", "target_id": "", "rank": -1,
                           "score": np.nan, "predicted_miss": np.nan})
    prop = pd.DataFrame(picked).merge(pool_out, on="name", how="left")

    args.out.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out / "targets.csv", index=False)
    pool_out.to_csv(args.out / "pool.csv", index=False)
    prop.to_csv(args.out / "proposals.csv", index=False)
    sdir = args.out / "structures"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir()
    uniq = prop.drop_duplicates("name").sort_values("nsites", ascending=False)
    for _, r in uniq.iterrows():
        shutil.copy(r["cif"], sdir / f"{r['name']}.cif")
    (args.out / "mlip_structures.txt").write_text("".join(f"structures/{n}.cif\n" for n in uniq["name"]))

    with pd.option_context("display.width", 220):
        print("\ntargets:")
        print(targets.round(3).to_string(index=False))
        print(f"\n{(prop['role'] == 'proposal').sum()} proposals + {(prop['role'] == 'control').sum()} controls "
              f"= {len(uniq)} structures; X: {uniq['x_site'].value_counts().to_dict()}; "
              f"nsites median {uniq['nsites'].median():.0f}, max {uniq['nsites'].max()}")
        p = prop[prop["role"] == "proposal"]
        print("\npredicted Chebyshev miss of the proposals (noise units), per target:")
        print(p.groupby("target_id")["predicted_miss"].describe()[["min", "50%", "max"]].round(2).to_string())
    print(f"-> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrospective", action="store_true", help="Held-out chemical cluster dry run")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: data/proposals/mp_pool_v1 (or data/proposals/retrospective)")
    parser.add_argument("--max-e-above-hull", type=float, default=0.1, help="Pool hull cut, eV/atom (default 0.1)")
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--targets-csv", type=Path, default=None)
    parser.add_argument("--n-per-target", type=int, default=8)
    parser.add_argument("--n-controls", type=int, default=16)
    parser.add_argument("--tol-rank", type=float, default=3.0, help="Noise units for the ranking score (default 3)")
    parser.add_argument("--n-opt", type=int, default=1000, help="Rows the GP hyperparameters are fitted on")
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    if args.out is None:
        args.out = PROJECT_ROOT / "data" / "proposals" / ("retrospective" if args.retrospective else "mp_pool_v1")

    noise = pd.read_csv(PROJECT_ROOT / "data" / "target_clusters" / "noise_scale.csv",
                        index_col=0)["noise"].reindex(CORE_TARGETS)
    g = notebook_tables()
    (retrospective if args.retrospective else prospective)(g, noise, args)


if __name__ == "__main__":
    main()
