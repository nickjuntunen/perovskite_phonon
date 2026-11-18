"""Mixed-halide series: do X-site disorder features and a bowing-only target
fix the GP / tree R2(series)?

Three variants, each fit with the GP (gp_try.py's ARD RBF + white kernel,
hyperparameters refit per training set) and the gradient-boosted trees:

    base       C+G features (descriptor_selection.ipynb's FEATS), raw target
    dis        C+G + X-site disorder (composition_features.X_DISORDER_FEATURES:
               x_mix, mass_cv_X, r_spread_X, en_spread_X), raw target
    dis_delta  C+G + disorder, target = value - linear end-member baseline
               (the bowing); prediction = baseline + predicted bowing. MP rows
               train it with bowing 0; mixed rows without both end members
               can't be expressed and are left out of its training

Two hold-out schemes; MP polymorphs are always in training:

    pair    one halide pair (Cl/Br, Br/I, Cl/I) held out, the others' mixtures
            trained on -- transfer to a pair never seen (gp_try.py's test)
    series  GroupKFold(5) over series, every pair seen -- learnability of the
            x-dependence itself

Scored on the same rows for every variant: mixed compositions with both end
members (the linear baseline exists), R2(series) pooled over each scheme's
held-out rows, against the per-series mean as in the notebook. The linear
baseline's own R2(series) on those rows is the reference line.

One target per call (one Slurm array task, see run_gp_disorder.sbatch);
every (variant, scheme, fold) is checkpointed to <out>/parts/<target>/, so a
requeued task resumes. Run from anywhere:

    uv run python3 gp_vs_tree/gp_disorder.py --target shape_pc1 --jobs 12
    uv run python3 gp_vs_tree/gp_disorder.py --summarize      # after all targets
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # gp_setup / the notebook cells resolve data paths from the cwd
sys.path.insert(0, str(ROOT))

TARGETS = ["rel_centroid_A", "log_omega_log", "rel_centroid_B", "shape_pc1", "rel_width_A",
           "log_omega_max", "rel_width_X", "log_msd_A", "log_k_A"]
VARIANTS = ["base", "dis", "dis_delta"]
N_OPT = 1000  # rows the kernel hyperparameters are optimized on, as gp_try.py
OUT = ROOT / "gp_vs_tree" / "disorder"


def summarize() -> None:
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("result_*.json"))]
    if not rows:
        raise SystemExit(f"no result_*.json in {OUT}")
    res = pd.DataFrame(rows).set_index("target")
    res.to_csv(OUT / "gp_disorder.csv")
    with pd.option_context("display.width", 250):
        print(res.drop(columns=["sec"], errors="ignore").round(2).to_string())
    print(f"-> {OUT / 'gp_disorder.csv'}")


def main() -> None:
    global OUT, N_OPT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--task-index", type=int, default=None, help="Pick TARGETS[i] (Slurm array index)")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel (variant, scheme, fold) fits")
    parser.add_argument("--summarize", action="store_true", help="Collect every target's result into one CSV")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: optimize the kernel on 100 rows instead of 1000 (use with a scratch --out)")
    args = parser.parse_args()
    OUT = args.out
    if args.quick:
        N_OPT = 100
    if args.summarize:
        summarize()
        return
    target = args.target or (TARGETS[args.task_index] if args.task_index is not None else None)
    if target is None:
        parser.error("give --target, --task-index or --summarize")

    g = {}
    exec(open(ROOT / "gp_vs_tree" / "gp_setup.py").read(), g)
    run(target, g, args.jobs)


def run(tgt: str, g: dict, n_jobs: int) -> None:
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.analysis.composition_features import X_DISORDER_FEATURES, x_site_disorder

    t0 = time.time()
    df, dm, T, Tm, end_value, model, SEED = (g[k] for k in ("df", "dm", "T", "Tm", "end_value", "model", "SEED"))
    base_feats = list(g["FEATS"])
    feats = {"base": base_feats, "dis": base_feats + X_DISORDER_FEATURES, "dis_delta": base_feats + X_DISORDER_FEATURES}
    X_all = pd.concat([g["X_all"], x_site_disorder(df)], axis=1)
    Xm = pd.concat([g["Xm"], x_site_disorder(dm)], axis=1)

    y = Tm[tgt]
    lin = pd.Series(((1 - dm["x2"]) * end_value[tgt].reindex(dm["end_X1"]).to_numpy()
                     + dm["x2"] * end_value[tgt].reindex(dm["end_X2"]).to_numpy()).to_numpy(), index=dm.index)
    ok = y.notna()
    has_lin = ok & lin.notna()
    series_mean = y.where(ok).groupby(dm["series"]).transform("mean")  # as the notebook's R2(series)
    y_mp = T[tgt]

    folds = [("pair", p.replace("/", "-"), (dm["pair"] == p).to_numpy()) for p in sorted(dm["pair"].unique())]
    mixed_idx = np.where(ok.to_numpy())[0]
    for k, (_, te) in enumerate(GroupKFold(5).split(mixed_idx, groups=dm["series"].to_numpy()[mixed_idx])):
        mask = np.zeros(len(dm), bool)
        mask[mixed_idx[te]] = True
        folds.append(("series", f"fold{k}", mask))

    parts = OUT / "parts" / tgt
    parts.mkdir(parents=True, exist_ok=True)

    def prep():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    def one(variant: str, scheme: str, fold: str, te_mask: np.ndarray) -> None:
        path = parts / f"{variant}__{scheme}__{fold}.csv"
        if path.exists():
            return
        F = feats[variant]
        tr = ~te_mask & (has_lin if variant == "dis_delta" else ok).to_numpy()
        te = te_mask & has_lin.to_numpy()  # scored rows: the same for every variant
        if variant == "dis_delta":
            y_tr = pd.concat([pd.Series(0.0, index=y_mp.index).where(y_mp.notna()), (y - lin)[tr]])
        else:
            y_tr = pd.concat([y_mp, y[tr]])
        X_tr = pd.concat([X_all, Xm[tr]])[F]
        keep = y_tr.notna().to_numpy()
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        X_te = Xm[te][F]

        p = prep().fit(X_tr)
        Z_tr, Z_te = p.transform(X_tr), p.transform(X_te)
        idx = np.random.default_rng(SEED).choice(len(y_tr), min(N_OPT, len(y_tr)), replace=False)
        kernel = ConstantKernel(1.0) * RBF(np.full(len(F), 3.0), (1e-2, 1e3)) + WhiteKernel(0.1, (1e-4, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k_fit = GaussianProcessRegressor(kernel, normalize_y=True, random_state=SEED).fit(
                Z_tr[idx], y_tr.iloc[idx]).kernel_
            gp = GaussianProcessRegressor(k_fit, normalize_y=True, optimizer=None).fit(Z_tr, y_tr)
            pred_gp = gp.predict(Z_te)
        pred_tree = model().fit(X_tr, y_tr).predict(X_te)
        if variant == "dis_delta":
            pred_gp, pred_tree = pred_gp + lin[te].to_numpy(), pred_tree + lin[te].to_numpy()
        pd.DataFrame({"run_name": dm.loc[te, "run_name"], "GP": pred_gp, "tree": pred_tree},
                     index=dm.index[te]).to_csv(path)
        (parts / f"{variant}__{scheme}__{fold}.kernel.txt").write_text(str(k_fit))

    jobs = [(v, s, f, m) for v in VARIANTS for s, f, m in folds]
    Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs)

    # pooled R2(series) per (variant, scheme, model), on the rows with a linear baseline
    def r2(idx, pred):
        yy, sm = y.loc[idx], series_mean.loc[idx]
        return float(1 - ((yy - pred) ** 2).sum() / ((yy - sm) ** 2).sum())

    row = {"target": tgt}
    for scheme in ("pair", "series"):
        te_all = np.zeros(len(dm), bool)
        for s, _, m in folds:
            if s == scheme:
                te_all |= m
        idx = dm.index[te_all & has_lin.to_numpy()]
        row[f"{scheme} linear"] = r2(idx, lin.loc[idx])
        for v in VARIANTS:
            pr = pd.concat([pd.read_csv(parts / f"{v}__{scheme}__{f}.csv", index_col=0)
                            for s, f, _ in folds if s == scheme])
            for name in ("GP", "tree"):
                row[f"{scheme} {v} {name}"] = r2(pr.index, pr[name])
    row["n_scored"] = int(has_lin.sum())
    row["sec"] = time.time() - t0
    (OUT / f"result_{tgt}.json").write_text(json.dumps(row, indent=1))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
