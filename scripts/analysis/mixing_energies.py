"""Mixing energies and free energies of the mixed-halide SQS sets
(scripts/dataset/generate_mixed_halides.py) against their MP end members.

For a mixture AB(X1_{1-x} X2_x)3 -- and the double-perovskite analogues; every
mixture has the same atoms per formula unit as its end members, so per-atom
energies compare directly -- with x the X2 share of the X site, taken from the
SQS cell's actual formula:

    dE_mix   = E(mix) - (1-x) E(ABX1_3) - x E(ABX2_3)                  static, meV/atom
    dS_conf  = -f_X [x ln x + (1-x) ln(1-x)]                            ideal random alloy, kB/atom
    dS_vib   = S_vib(mix) - (1-x) S_vib(X1 end) - x S_vib(X2 end)        harmonic, 300 K, kB/atom
    dF_vib   = the same for the harmonic vibrational free energy (incl. zero point), at --temperature
    dF_mix   = dE_mix + dF_vib - T dS_conf

E is the MLIP (MACE-matpes-r2scan-omat-ft) relaxed energy per atom; f_X = X
sites per atom (3/5 in every perovskite here). dE_mix > 0: the alloy is above
the end members at 0 K, i.e. it would rather phase-separate; dF_mix < 0 means
configurational and vibrational entropy stabilize it at T.

States.
- Mixture: the run's selected polymorph (select_stable_polymorphs.py on the set),
  its lowest-energy dynamically stable state. Rows without one are kept,
  flagged mix_stable = False.
- End member, the reference (`ref_*`): the lowest-energy dynamically stable
  state over *all* MP entries of the end-member formula, i.e. rank 0 of
  data/mlip_stable_polymorphs/stable_polymorphs.csv -- the MLIP ground state
  this sweep knows of.
- Also given, `dE_mix_entry`: against the lowest stable state of the specific
  MP entry the mixture was built from (compositions.csv end_X1 / end_X2). Where
  the two references differ, part of dE_mix is a polymorph difference, not mixing.

Series hull. Along each series (one framework, one halide pair) the end members
sit at x = 0, 1 with dE = 0. `dE_above_hull` (and `dF_above_hull`) is each
mixture's height above the lower convex hull of the series' points: > 0 means
it would decompose into its neighbours (other x, or the end members), even
where dE_mix < 0. Mixtures whose run is unstable or not computed are left out of
the hull.

Vibrational terms. dS_vib uses the exact harmonic entropy of
extract_design_descriptors.py (s_kB_per_atom, 300 K, from the force constants
on the q-mesh). dF_vib has no stored counterpart and is integrated over the
saved 64-level vDOS quantile function, F = 3 <f(Q(p))>_p. Checked: the same
quadrature reproduces s_kB_per_atom to -0.005 kB/atom median (T*S ~0.13
meV/atom at 300 K) on the MP set, and most of that cancels in differences;
`dS_vib_quantile` is written alongside so the cancellation can be checked per row.

Caveats.
- dE_mix < 0 mostly means the reference is not the end member's true ground
  state, not favourable mixing. The end members were relaxed in their MP cells
  (often 10 atoms), the mixtures in >= 40-atom SQS cells and then rattled, so a
  mixture can find a distortion its end member's cell can't hold (e.g.
  Rb2AgAu(Br5Cl): P1 rattled vs the 10-atom Fm-3m Rb2AgAuBr6 at -0.98 cm-1, just
  inside the 1 cm-1 tolerance; dE_mix = -11 meV/atom). Compare mix_n_atoms /
  ref*_n_atoms and mix_state; the series hull inherits the same bias.
- One SQS cell per composition in production: the noise-check set
  (--set-dir data/mixed_halides_noise_check_mp) gives the seed spread.
- The ideal dS_conf is an upper bound (short-range order lowers it).
- All energies are without DFT-D3, like the phonon data.

Output: <set-dir>/analysis/mixing_energies.csv, one row per SQS run.

Usage:
    uv run python3 scripts/analysis/mixing_energies.py
    uv run python3 scripts/analysis/mixing_energies.py --set-dir data/mixed_halides_noise_check_mp
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.constants import c as C_LIGHT, e as E_CHARGE, h as H_PLANCK, k as K_B

from src.paths import MIXED_HALIDES_DIR, STABLE_POLYMORPHS_DIR

KB_MEV = K_B / E_CHARGE * 1e3  # meV/K
HALIDES = ("F", "Cl", "Br", "I")


def vib_from_quantiles(q_cm1: np.ndarray, temperature: float) -> tuple[float, float]:
    """Harmonic (F meV/atom incl. zero point, S kB/atom) from a unit-area
    vDOS quantile function at midpoint levels: 3 modes per atom."""
    x = H_PLANCK * C_LIGHT * 100 * np.clip(q_cm1, 1e-3, None) / (K_B * temperature)
    f = 3 * np.mean(x / 2 + np.log1p(-np.exp(-x))) * KB_MEV * temperature
    s = 3 * np.mean(x / np.expm1(x) - np.log1p(-np.exp(-x)))
    return float(f), float(s)


def lower_hull_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Height of each (x, y) above the lower convex hull of the points plus
    (0, 0) and (1, 0)."""
    px = np.concatenate([[0.0, 1.0], x])
    py = np.concatenate([[0.0, 0.0], y])
    order = np.lexsort((py, px))
    hull: list[int] = []
    for i in order:  # monotone chain, lower half
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = (px[hull[-2]], py[hull[-2]]), (px[hull[-1]], py[hull[-1]])
            if (x2 - x1) * (py[i] - y1) - (y2 - y1) * (px[i] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    hx, hy = px[hull], py[hull]
    return y - np.interp(x, hx, hy)


def load_states(analysis_dir: Path) -> tuple[pd.DataFrame, dict]:
    """design_descriptors.csv's entropy by key, and the quantile functions."""
    d = pd.read_csv(analysis_dir / "design_descriptors.csv", usecols=["key", "s_kB_per_atom"])
    q = np.load(analysis_dir / "design_descriptors_quantiles.npz")
    return d.set_index("key"), dict(zip(q["keys"], q["quantiles_cm1"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-dir", type=Path, default=MIXED_HALIDES_DIR,
                        help="A generate_mixed_halides.py set (compositions.csv, analysis/selection/, analysis/)")
    parser.add_argument("--mp-dir", type=Path, default=STABLE_POLYMORPHS_DIR,
                        help="The MP polymorph selection and its design descriptors")
    parser.add_argument("--temperature", type=float, default=300.0, help="K, for dS_conf and dF_vib (default 300)")
    parser.add_argument("--out", type=Path, default=None, help="Default: <set-dir>/analysis/mixing_energies.csv")
    args = parser.parse_args()
    T = args.temperature
    analysis = args.set_dir / "analysis"
    out_path = args.out or analysis / "mixing_energies.csv"

    # --- mixtures: one row per SQS run
    comp = pd.read_csv(args.set_dir / "compositions.csv")
    comp = comp[comp["structures"].notna()].assign(run_name=lambda d: d["structures"].str.split(";")).explode("run_name")
    sel = pd.read_csv(analysis / "selection" / "stable_polymorphs.csv", float_precision="round_trip")
    sel = sel.sort_values("rank").drop_duplicates("run_name")  # one state per run: the lowest (rank 0)
    mix = comp[["run_name", "composition", "framework", "X1", "X2", "x2", "end_X1", "end_X2"]].merge(
        sel[["run_name", "formula", "state", "energy_eV_per_atom", "dynamically_stable", "unstable_state_lower",
             "omega_min_cm1", "n_atoms"]].rename(columns={"n_atoms": "mix_n_atoms"}), on="run_name", how="left")
    n_missing = mix["energy_eV_per_atom"].isna().sum()

    # actual x and X share per atom, from the SQS cell's formula
    def x_actual(row):
        el = Composition(row["formula"]).get_el_amt_dict()
        n1, n2 = el.get(row["X1"], 0.0), el.get(row["X2"], 0.0)
        return n2 / (n1 + n2), (n1 + n2) / sum(el.values())
    have = mix["formula"].notna()
    mix.loc[have, ["x", "f_X"]] = np.array([x_actual(r) for _, r in mix[have].iterrows()])

    # --- MP end members
    mp_sel = pd.read_csv(args.mp_dir / "stable_polymorphs.csv", float_precision="round_trip")
    mp_cand = pd.read_csv(args.mp_dir / "candidates.csv", float_precision="round_trip")
    formula_of_run = mp_cand.drop_duplicates("run_name").set_index("run_name")["formula"]
    ground = (mp_sel[mp_sel["dynamically_stable"].astype(bool)].sort_values("rank")
              .drop_duplicates("formula").set_index("formula"))
    entry = (mp_cand[mp_cand["dynamically_stable"].astype(bool)].sort_values("energy_eV_per_atom")
             .drop_duplicates("run_name").set_index("run_name"))
    mp_s, mp_q = load_states(args.mp_dir)
    mix_s, mix_q = load_states(analysis)

    for side in ("1", "2"):
        end = mix[f"end_X{side}"]
        f = end.map(formula_of_run)
        g = ground.reindex(f)
        mix[f"ref{side}_formula"] = f.to_numpy()
        mix[f"ref{side}_run"] = g["run_name"].to_numpy()
        mix[f"ref{side}_state"] = g["state"].to_numpy()
        mix[f"ref{side}_n_atoms"] = g["n_atoms"].to_numpy()
        mix[f"E_ref{side}"] = g["energy_eV_per_atom"].to_numpy()
        mix[f"E_entry{side}"] = entry["energy_eV_per_atom"].reindex(end).to_numpy()
        mix[f"ref{side}_is_entry"] = (g["run_name"].to_numpy() == end.to_numpy())

    x, fX = mix["x"], mix["f_X"]
    lin = lambda a, b: (1 - x) * mix[a] + x * mix[b]
    mix["dE_mix_meV_per_atom"] = 1e3 * (mix["energy_eV_per_atom"] - lin("E_ref1", "E_ref2"))
    mix["dE_mix_entry_meV_per_atom"] = 1e3 * (mix["energy_eV_per_atom"] - lin("E_entry1", "E_entry2"))
    with np.errstate(divide="ignore", invalid="ignore"):
        s_conf = -fX * (x * np.log(x) + (1 - x) * np.log(1 - x))
    mix["dS_conf_kB_per_atom"] = s_conf.fillna(0.0).where(x.notna())

    # vibrational terms, against the ground-state references
    def vib(key_mix, keys_ref):
        out = {"s": np.nan, "sq": np.nan, "f": np.nan}
        k1, k2 = keys_ref
        if not (isinstance(key_mix, str) and key_mix in mix_q and k1 in mp_q and k2 in mp_q):
            return out
        (fm, sm), (f1, s1), (f2, s2) = (vib_from_quantiles(mix_q[key_mix], T),
                                        vib_from_quantiles(mp_q[k1], T), vib_from_quantiles(mp_q[k2], T))
        return {"fq": (fm, f1, f2), "sq": (sm, s1, s2),
                "s": (mix_s["s_kB_per_atom"].get(key_mix, np.nan), mp_s["s_kB_per_atom"].get(k1, np.nan),
                      mp_s["s_kB_per_atom"].get(k2, np.nan))}
    dS, dSq, dF = [], [], []
    for _, r in mix.iterrows():
        v = vib(f"{r['run_name']}/{r['state']}" if isinstance(r["state"], str) else None,
                (f"{r['ref1_run']}/{r['ref1_state']}", f"{r['ref2_run']}/{r['ref2_state']}"))
        if "fq" not in v:
            dS.append(np.nan), dSq.append(np.nan), dF.append(np.nan)
            continue
        d = lambda t: t[0] - (1 - r["x"]) * t[1] - r["x"] * t[2]
        dS.append(d(v["s"])), dSq.append(d(v["sq"])), dF.append(d(v["fq"]))
    mix["dS_vib_300K_kB_per_atom"] = dS
    mix["dS_vib_quantile_kB_per_atom"] = dSq
    mix[f"dF_vib_{T:g}K_meV_per_atom"] = dF
    mix[f"dF_mix_{T:g}K_meV_per_atom"] = (mix["dE_mix_meV_per_atom"] + mix[f"dF_vib_{T:g}K_meV_per_atom"]
                                          - KB_MEV * T * mix["dS_conf_kB_per_atom"])

    # --- heights above each series' lower hull (stable, computed mixtures only)
    mix["series"] = mix["framework"] + " " + mix["X1"] + "/" + mix["X2"]
    ok = mix["dynamically_stable"].astype("boolean").fillna(False).astype(bool)
    for col, hcol in [("dE_mix_meV_per_atom", "dE_above_hull_meV_per_atom"),
                      (f"dF_mix_{T:g}K_meV_per_atom", f"dF_{T:g}K_above_hull_meV_per_atom")]:
        mix[hcol] = np.nan
        for _, grp in mix[ok & mix[col].notna()].groupby("series"):
            mix.loc[grp.index, hcol] = lower_hull_height(grp["x"].to_numpy(), grp[col].to_numpy())

    mix = mix.rename(columns={"energy_eV_per_atom": "E_mix", "dynamically_stable": "mix_stable", "state": "mix_state"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mix.to_csv(out_path, index=False)

    # --- summary
    good = mix[ok & mix["dE_mix_meV_per_atom"].notna()]
    print(f"{len(mix)} SQS runs, {mix['composition'].nunique()} compositions in {args.set_dir}; "
          f"{n_missing} without a selected state, {(~ok).sum()} not dynamically stable, "
          f"{mix['dE_mix_meV_per_atom'].isna().sum()} without both end-member references")
    print(f"dE_mix < -2 meV/atom (reference likely not the ground state, see docstring): "
          f"{(good['dE_mix_meV_per_atom'] < -2).sum()}/{len(good)}")
    print(f"ground-state reference is the mixture's own MP entry on both sides for "
          f"{(good['ref1_is_entry'] & good['ref2_is_entry']).mean():.0%} of the {len(good)} stable rows")
    fcol, hcol = f"dF_mix_{T:g}K_meV_per_atom", f"dF_{T:g}K_above_hull_meV_per_atom"
    summary = good.groupby(good["X1"] + "/" + good["X2"]).agg(
        n=("dE_mix_meV_per_atom", "size"),
        dE_median=("dE_mix_meV_per_atom", "median"),
        dE_p90=("dE_mix_meV_per_atom", lambda s: s.quantile(0.9)),
        frac_dE_pos=("dE_mix_meV_per_atom", lambda s: (s > 0).mean()),
        TdS_conf_median=("dS_conf_kB_per_atom", lambda s: (KB_MEV * T * s).median()),
        dF_vib_median=(f"dF_vib_{T:g}K_meV_per_atom", "median"),
        frac_dF_neg=(fcol, lambda s: (s < 0).mean()),
        frac_on_F_hull=(hcol, lambda s: (s < 0.1).mean()),
    )
    with pd.option_context("display.width", 200):
        print(f"\nper halide pair (meV/atom; T = {T:g} K; on hull = within 0.1 meV/atom):")
        print(summary.round(2).to_string())
    chk = good[["dS_vib_300K_kB_per_atom", "dS_vib_quantile_kB_per_atom"]].dropna()
    if len(chk):
        err = chk.iloc[:, 1] - chk.iloc[:, 0]
        print(f"\ndS_vib quadrature check: quantile - exact, median {err.median():+.4f}, "
              f"|p95| {err.abs().quantile(0.95):.4f} kB/atom ({len(chk)} rows)")
    if mix.groupby("composition")["run_name"].nunique().max() > 1:
        spread = good.groupby("composition")["dE_mix_meV_per_atom"].agg(["std", "size"])
        spread = spread[spread["size"] > 1]
        print(f"\nSQS seed spread of dE_mix: median std {spread['std'].median():.2f} meV/atom, "
              f"p90 {spread['std'].quantile(0.9):.2f} ({len(spread)} compositions with > 1 stable run)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
