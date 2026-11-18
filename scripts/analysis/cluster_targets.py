"""Redundant compositions in phonon-target space: groups of different
compositions whose core design targets agree to within a tolerance, i.e.
interchangeable answers to the same design question.

Target space: the core targets of descriptor_selection.md section 6, one per
independent direction --
    log_omega_log, rel_centroid_A, rel_width_A, shape_pc1, rel_centroid_B,
    log_distortion_energy
computed as in descriptor_selection.ipynb (phonon_targets, the MP vDOS shape
PCA) for the dynamically stable MP polymorphs (data/mlip_stable_polymorphs)
and the mixed-halide production set (data/mixed_halides). MP antiperovskites
(a metal on the X site, e.g. Mn3GaN, LaNi3B, Ca3BiN) are left out.

Units: each target is divided by its MLIP noise, the median model-to-model std
on the gauntlet set (Pnma CsPbBr3, eight MLIPs; the notebook's Check 1), so a
distance of 1 is "one MLIP noise". log_distortion_energy has no gauntlet value
(the gauntlet runs have no symmetric parent); its noise is set to
std / median(SNR of the other core targets), flagged in the output.

Redundancy: complete-linkage clustering under the Chebyshev (max-over-targets)
metric, cut at --tol noise units. Complete linkage bounds the cluster
diameter, so every pair of members is within tol on every target at once --
the same "inside a window on each target" definition the inverse search uses.
Single linkage would chain dissimilar compositions through intermediates.
Run for each tolerance in --tols.

Chemical spread of a cluster: the median pairwise distance between its
distinct formulas in the standardized composition (C-tier) descriptors,
divided by the median over random pairs (descriptor_selection.md's rho_inv,
per cluster). ~0: same chemistry (polymorphs, neighbouring mixing fractions);
~1: as different as random compositions.

Outputs, in --out-dir (default data/target_clusters):
    members_tol<t>.csv    one row per polymorph: formula, source, anion, cluster, cluster size, targets
    clusters_tol<t>.csv   one row per cluster with >= 2 distinct formulas: sizes, chemical spread,
                          A/B/X element sets, example formulas, target centroid
    summary.csv           per tolerance: fraction of formulas with a redundant partner, cluster counts
    noise_scale.csv       the noise unit per target

Usage:
    uv run python3 scripts/analysis/cluster_targets.py
    uv run python3 scripts/analysis/cluster_targets.py --tols 1 2 3 --no-mixed
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

DATA = PROJECT_ROOT / "data"
MP = DATA / "mlip_stable_polymorphs"
MIXED = DATA / "mixed_halides"
GAUNTLET = DATA / "phonon_benchmark" / "design_descriptors_gauntlet.csv"
CORE = ["log_omega_log", "rel_centroid_A", "rel_width_A", "shape_pc1", "rel_centroid_B", "log_distortion_energy"]
C_FEATURES = ["mass_A", "mass_B", "mass_X", "r_A", "r_B", "r_X", "tolerance_factor", "octahedral_factor",
              "en_A", "en_B", "en_X", "oxidation_A", "oxidation_B", "oxidation_X",
              "x_frac_F", "x_frac_Cl", "x_frac_Br", "x_frac_I"]
# X sites kept: anions only. A metal on the structural check's X site means an antiperovskite (A3BX with
# metal-cornered octahedra: Mn3GaN, Y3AlC, LaNi3B, but also ionic ones like Ca3BiN, Sr3PbO), dropped.
X_ANIONS = {"O", "S", "Se", "Te", "N", "F", "Cl", "Br", "I"}
SEED = 0


def phonon_targets(d: pd.DataFrame) -> pd.DataFrame:
    # The core subset of descriptor_selection.ipynb's phonon_targets (cell 5), same definitions.
    t = pd.DataFrame(index=d.index)
    t["log_omega_log"] = np.log(d["omega_log_cm1"])
    for s in "AB":
        t[f"rel_centroid_{s}"] = d[f"omega_mean_{s}_cm1"] / d["omega_mean_cm1"]
    t["rel_width_A"] = d["omega_std_A_cm1"] / d["omega_mean_cm1"]
    if "distortion_energy_meV_per_atom" in d:
        t["log_distortion_energy"] = np.log1p(d["distortion_energy_meV_per_atom"].clip(lower=0))
    return t


def quantiles(path: Path, keys: pd.Series) -> np.ndarray:
    q = np.load(path)
    qmap = dict(zip(q["keys"], q["quantiles_cm1"]))
    Q = np.array([qmap[k] for k in keys])
    return Q / Q.mean(axis=1, keepdims=True)


def stable(d: pd.DataFrame) -> pd.DataFrame:
    if "error" in d:
        d = d[d["error"].isna()]
    return d[d["dynamically_stable"].astype(bool) & (d["sublattice_unassigned"] == 0)
             & (d["n_A"] > 0) & (d["n_B"] > 0)].reset_index(drop=True)


def load_mp() -> pd.DataFrame:
    # as descriptor_selection.ipynb cell 3
    design = pd.read_csv(MP / "design_descriptors.csv")
    poly = pd.read_csv(MP / "polymorph_clusters.csv")
    poly["key"] = poly["run_name"] + "/" + poly["state"]
    comp = pd.read_csv(DATA / "mp_perovskites" / "composition_features.csv").rename(columns={"name": "run_name"})
    d = design.merge(poly[["key", "dynamically_stable", "mp_x_site"]], on="key", how="inner").merge(
        comp, on="run_name", how="left")
    d = stable(d)
    n = d["formula"].nunique()
    d = d[d["mp_x_site"].isin(X_ANIONS)].reset_index(drop=True)
    print(f"MP: dropped {n - d['formula'].nunique()} antiperovskite formulas (metal on the X site)")
    d["source"], d["anion"] = "MP", d["mp_x_site"]
    return d


def load_mixed() -> pd.DataFrame:
    # as descriptor_selection.ipynb cell 27
    design = pd.read_csv(MIXED / "analysis" / "design_descriptors.csv")
    sel = pd.read_csv(MIXED / "analysis" / "selection" / "stable_polymorphs.csv")
    sel["key"] = sel["run_name"] + "/" + sel["state"]
    comp = pd.read_csv(MIXED / "analysis" / "composition_features.csv").rename(columns={"name": "run_name"})
    series = pd.read_csv(MIXED / "compositions.csv").rename(columns={"structures": "run_name"})[
        ["run_name", "framework", "X1", "X2"]]
    d = (design.merge(sel[["key", "dynamically_stable"]], on="key", how="inner")
         .merge(comp, on="run_name", how="left").merge(series, on="run_name", how="inner"))
    d = stable(d)
    d["source"], d["anion"] = "mixed", d["X1"] + "/" + d["X2"]
    return d


def elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula))


def site_set(d: pd.DataFrame, col: str) -> str:
    if col not in d:
        return ""
    vals = sorted({e for v in d[col].dropna().astype(str) for e in re.findall(r"[A-Z][a-z]?", v)})
    return " ".join(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tols", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                        help="Tolerances, in MLIP-noise units per target (default: 1 2 3)")
    parser.add_argument("--no-mixed", action="store_true", help="MP polymorphs only")
    parser.add_argument("--out-dir", type=Path, default=DATA / "target_clusters")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mp = load_mp()
    Qmp = quantiles(MP / "design_descriptors_quantiles.npz", mp["key"])
    shape_pca = PCA(n_components=6, random_state=SEED).fit(Qmp)  # the notebook's MP shape basis
    frames = [mp]
    if not args.no_mixed:
        frames.append(load_mixed())
    d = pd.concat(frames, ignore_index=True)
    T = phonon_targets(d)
    Q = np.vstack([Qmp] + ([quantiles(MIXED / "analysis" / "design_descriptors_quantiles.npz", frames[1]["key"])]
                           if len(frames) > 1 else []))
    T["shape_pc1"] = shape_pca.transform(Q)[:, 0]
    T = T[CORE]

    # noise unit per target: median model-to-model std on the gauntlet (notebook cell 11)
    g = pd.read_csv(GAUNTLET)
    g = g[g["error"].isna()] if "error" in g else g
    tg = phonon_targets(g)
    tg["shape_pc1"] = shape_pca.transform(quantiles(GAUNTLET.with_name(GAUNTLET.stem + "_quantiles.npz"), g["key"]))[:, 0]
    noise = tg.groupby(g["structure"]).std().median().reindex(CORE)
    measured = noise.notna() & (noise > 0)
    snr = T.std()[measured] / noise[measured]
    noise[~measured] = T.std()[~measured] / snr.median()
    pd.DataFrame({"noise": noise, "dataset_std": T.std(), "snr": T.std() / noise,
                  "from_gauntlet": measured}).to_csv(args.out_dir / "noise_scale.csv")
    print("noise unit per target (SNR = dataset std / noise):")
    print(pd.DataFrame({"noise": noise, "SNR": T.std() / noise, "measured": measured}).round(4).to_string())

    ok = T.notna().all(axis=1)
    d, T = d[ok].reset_index(drop=True), T[ok].reset_index(drop=True)
    Z = (T / noise).to_numpy()
    print(f"\n{len(d)} polymorphs ({d['formula'].nunique()} formulas) with all core targets: "
          + ", ".join(f"{k} {v}" for k, v in d["source"].value_counts().items()))

    # chemical spread reference: random pairs of formulas in standardized C descriptors
    C = d.groupby("formula")[[c for c in C_FEATURES if c in d]].first()
    C = ((C - C.mean()) / C.std()).fillna(0.0)
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, len(C), (2, 20000))
    keep = i != j
    ref = np.median(np.linalg.norm(C.to_numpy()[i[keep]] - C.to_numpy()[j[keep]], axis=1))

    link = linkage(pdist(Z, metric="chebyshev"), method="complete")
    summary = []
    for tol in args.tols:
        lab = fcluster(link, t=tol, criterion="distance")
        m = d[["formula", "source", "anion", "run_name", "key"]].copy()
        m["cluster"] = lab
        m = pd.concat([m, T], axis=1)
        nform = m.groupby("cluster")["formula"].nunique()
        m["n_formulas_in_cluster"] = m["cluster"].map(nform)
        m.to_csv(args.out_dir / f"members_tol{tol:g}.csv", index=False)

        rows = []
        for c, grp in m[m["n_formulas_in_cluster"] >= 2].groupby("cluster"):
            forms = grp["formula"].unique()
            x = C.loc[forms].to_numpy()
            spread = np.median(pdist(x)) / ref
            src = d.loc[grp.index]
            rows.append({
                "cluster": c, "n_formulas": len(forms), "n_polymorphs": len(grp),
                "chem_spread": spread, "anions": " ".join(sorted(grp["anion"].astype(str).unique())),
                "sources": " ".join(sorted(grp["source"].unique())),
                "A_elements": site_set(src, "a_sites"), "B_elements": site_set(src, "b_sites"),
                "elements": " ".join(sorted(set().union(*map(elements, forms)))),
                "examples": "; ".join(sorted(forms)[:8]),
                **{f"{t}_mean": grp[t].mean() for t in CORE},
            })
        cl = pd.DataFrame(rows).sort_values(["n_formulas", "chem_spread"], ascending=False)
        cl.to_csv(args.out_dir / f"clusters_tol{tol:g}.csv", index=False)

        per_form = m.groupby("formula")["n_formulas_in_cluster"].max()
        multi_anion = (cl["anions"].str.split().str.len() > 1).sum() if len(cl) else 0
        summary.append({
            "tol_noise_units": tol, "n_formulas": len(per_form),
            "frac_formulas_with_partner": float((per_form >= 2).mean()),
            "n_clusters": int(m["cluster"].nunique()), "n_redundant_clusters": len(cl),
            "median_formulas_per_redundant_cluster": float(cl["n_formulas"].median()) if len(cl) else 0,
            "max_formulas_per_cluster": int(cl["n_formulas"].max()) if len(cl) else 1,
            "redundant_clusters_chem_spread_gt_0.5": int((cl["chem_spread"] > 0.5).sum()) if len(cl) else 0,
            "redundant_clusters_multi_anion": int(multi_anion),
        })
        print(f"\n== tol {tol:g} noise units: {summary[-1]['frac_formulas_with_partner']:.0%} of formulas have a "
              f"redundant partner; {len(cl)} redundant clusters "
              f"(median {summary[-1]['median_formulas_per_redundant_cluster']:.0f} formulas, "
              f"max {summary[-1]['max_formulas_per_cluster']}); "
              f"{summary[-1]['redundant_clusters_chem_spread_gt_0.5']} with chemical spread > 0.5, "
              f"{multi_anion} spanning anion families/pairs")
        if len(cl):
            with pd.option_context("display.width", 250, "display.max_colwidth", 90):
                print(cl.head(8)[["n_formulas", "chem_spread", "anions", "examples"]].round(2).to_string(index=False))
    pd.DataFrame(summary).to_csv(args.out_dir / "summary.csv", index=False)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
