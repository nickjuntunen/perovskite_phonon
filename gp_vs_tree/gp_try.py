import os, sys, time, warnings
exec(open(sys.argv[1]).read())
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

N_OPT = 1000
OUTP = os.path.dirname(sys.argv[3]) if len(sys.argv) > 3 else "."
def prep():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

def fit_kernel(X, y, seed=SEED):
    ok = y.notna().to_numpy()
    X, y = X.loc[ok, FEATS], y[ok]
    idx = np.random.default_rng(seed).choice(len(y), min(N_OPT, len(y)), replace=False)
    p = prep().fit(X)
    k = ConstantKernel(1.0) * RBF(np.full(len(FEATS), 3.0), (1e-2, 1e3)) + WhiteKernel(0.1, (1e-4, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(k, normalize_y=True, random_state=seed).fit(p.transform(X.iloc[idx]), y.iloc[idx])
    return gp.kernel_

def gp_fit_predict(kernel, X_tr, y_tr, X_te):
    ok = y_tr.notna().to_numpy()
    p = prep().fit(X_tr.loc[ok, FEATS])
    gp = GaussianProcessRegressor(kernel, normalize_y=True, optimizer=None).fit(p.transform(X_tr.loc[ok, FEATS]), y_tr[ok])
    return gp.predict(p.transform(X_te[FEATS]))

TARGETS = [t for t in (sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else HALIDE_TARGETS)]

def run(tgt):
    t0 = time.time()
    y = Tm[tgt]; ok = y.notna().to_numpy()
    series_mean = y.groupby(dm["series"]).transform("mean")
    k_mp = fit_kernel(X_all, T[tgt])
    row, preds = {}, {}
    # MP grouped CV within-family R2, GP vs tree
    okm = T[tgt].notna().to_numpy()
    Xo, yo, gr = X_all[okm], T.loc[okm, tgt], df.loc[okm, "formula"]
    fam_mean = yo.groupby(df.loc[okm, "anion_family"]).transform("mean")
    pg, pt = np.empty(len(yo)), np.empty(len(yo))
    for tr, te in GroupKFold(5).split(Xo, yo, gr):
        pg[te] = gp_fit_predict(k_mp, Xo.iloc[tr], yo.iloc[tr], Xo.iloc[te])
        pt[te] = model().fit(Xo.iloc[tr][FEATS], yo.iloc[tr]).predict(Xo.iloc[te][FEATS])
    row["MP R2w tree"], row["MP R2w GP"] = r2_vs(yo, pt, fam_mean), r2_vs(yo, pg, fam_mean)
    # 1. MP -> mixed
    for name, f in [("tree", lambda *a: fit_predict(tgt, *a)), ("GP", lambda *a: gp_fit_predict(k_mp, *a))]:
        pred = pd.Series(f(X_all, T[tgt], Xm), index=dm.index)
        preds[f"mp {name}"] = pred
        preds[f"pair {name}"] = pd.Series(np.nan, index=dm.index)
        row[f"R2series MP->mixed {name}"] = r2_vs(y[ok], pred[ok], series_mean[ok])
        # 2. held-out pair
        res = []
        for pair in dm["pair"].unique():
            te = (dm["pair"] == pair).to_numpy() & ok; tr = ~te & ok
            X_tr, y_tr = pd.concat([X_all, Xm[tr]]), pd.concat([T[tgt], y[tr]])
            if name == "GP":
                k_p = fit_kernel(X_tr, y_tr)
                f = lambda *a: gp_fit_predict(k_p, *a)
            pr = f(X_tr, y_tr, Xm[te])
            preds[f"pair {name}"].loc[te] = pr
            res.append(r2_vs(y[te], pr, series_mean[te]))
        row[f"R2series held-out pair {name}"] = np.median(res)
    # 3. held-out pure family + mixed (kernel refit without the family)
    for fam in ["chloride", "bromide/iodide"]:
        te = (df["anion_family"] == fam) & T[tgt].notna(); tr = ~te
        y_te = T.loc[te, tgt]
        X_tr, y_tr = pd.concat([X_all[tr], Xm]), pd.concat([T.loc[tr, tgt], y])
        k_f = fit_kernel(X_tr, y_tr)
        for name, f in [("tree", lambda *a: fit_predict(tgt, *a)), ("GP", lambda *a: gp_fit_predict(k_f, *a))]:
            pr = f(X_tr, y_tr, X_all[te])
            row[f"held-out {fam.split('/')[0]} +mixed {name}"] = r2_vs(y_te, pr, y_te.mean())
    row["sec"] = time.time() - t0
    pd.DataFrame(preds).assign(run_name=dm["run_name"]).to_csv(f"{OUTP}/pred_{tgt}.csv", index=False)
    return tgt, row, str(k_mp)

out = Parallel(n_jobs=min(12, len(TARGETS)))(delayed(run)(t) for t in TARGETS)
res = pd.DataFrame({t: r for t, r, _ in out}).T
for t, _, k in out:
    print(t, k[:300])
with pd.option_context("display.width", 250):
    print(res.round(2).to_string())
res.to_csv(sys.argv[3] if len(sys.argv) > 3 else "gp_try.csv")
