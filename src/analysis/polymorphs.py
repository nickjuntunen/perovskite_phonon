"""Polymorph bookkeeping on top of an MLIP phonon sweep (src/mlip/
phonon_pipeline.py output): describe every state each run produced
(symmetric / rattled / soft-mode), pick the stable low-energy ones per
formula, flag non-framework compounds, and the between/within-group spread
statistics used to compare clusterings of their vDOS.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from src.mlip.phonon_results import is_finished, path_min_q, run_states
from src.utils.symmetry import symmetry


def describe_state(state: dict, symprec: float, root: Path) -> dict:
    atoms = ase.io.read(state["dir"] / "structure.cif")
    comp = Composition(atoms.get_chemical_formula())
    reduced, n_fu = comp.get_reduced_formula_and_factor()
    return {
        **state,
        "dir": str(state["dir"].relative_to(root)),
        "structure_path": str((state["dir"] / "structure.cif").relative_to(root)),
        "formula": reduced,
        "n_atoms": len(atoms),
        "n_formula_units": int(n_fu),
        "volume_A3_per_atom": atoms.get_volume() / len(atoms),
        "cell_abc_A": " ".join(f"{x:.4f}" for x in atoms.cell.cellpar()[:3]),
        "cell_angles_deg": " ".join(f"{x:.2f}" for x in atoms.cell.cellpar()[3:]),
        **symmetry(atoms, symprec),
        "omega_min_q_path": path_min_q(state["dir"] / "band.npz"),
    }


def _describe(job):
    state, symprec, root = job
    try:
        return describe_state(state, symprec, root)
    except Exception as exc:  # unreadable CIF etc.: keep the row, flag it
        return {**state, "dir": str(state["dir"].relative_to(root)), "read_error": f"{type(exc).__name__}: {exc}"}


def collect(root: Path, symprec: float, workers: int) -> pd.DataFrame:
    states, incomplete = [], []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not is_finished(run_dir):
            incomplete.append(run_dir.name)
            continue
        states += run_states(run_dir)
    if incomplete:
        print(f"{len(incomplete)} run(s) without meta.json (unfinished), skipped: {', '.join(incomplete)}")

    jobs = [(s, symprec, root) for s in states]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_describe, jobs, chunksize=16))
    df = pd.DataFrame(rows)
    if "read_error" in df:
        bad = df[df["read_error"].notna()]
        print(f"{len(bad)} state(s) with unreadable structures, skipped")
        df = df[df["read_error"].isna()].drop(columns="read_error")
    return df


def select(df: pd.DataFrame, imag_tol: float, window_meV: float, root: Path,
           unstable_margin_meV: float, group_by: str = "formula") -> pd.DataFrame:
    matcher = StructureMatcher()
    df = df.copy()
    df["dynamically_stable"] = df["omega_min_cm1"] >= -imag_tol
    key = {"formula": "formula", "run": "run_name"}[group_by]
    formula_min = df.groupby(key)["energy_eV_per_atom"].transform("min")
    df["e_above_formula_min_meV_per_atom"] = 1000 * (df["energy_eV_per_atom"] - formula_min)

    picked = []
    for _, group in df.groupby(key, sort=True):
        stable = group[group["dynamically_stable"]].sort_values("energy_eV_per_atom")
        n_entries = group["material_id"].nunique()
        if stable.empty:
            best = group.sort_values("omega_min_cm1", ascending=False).iloc[[0]]
            reps = [(best.iloc[0], [])]
        else:
            e0 = stable["energy_eV_per_atom"].iloc[0]
            window = stable[1000 * (stable["energy_eV_per_atom"] - e0) <= window_meV]
            reps = []  # (representative row, [equivalent "mp-id/state", ...])
            for _, row in window.iterrows():
                s = AseAtomsAdaptor.get_structure(ase.io.read(root / row["structure_path"]))
                for rep, rep_struct, equivalents in reps:
                    if matcher.fit(rep_struct, s):
                        equivalents.append(f"{row['material_id']}/{row['state']}")
                        break
                else:
                    reps.append((row, s, []))
            reps = [(rep, eq) for rep, _, eq in reps]

        e_best_stable = stable["energy_eV_per_atom"].iloc[0] if not stable.empty else np.nan
        unstable = group[~group["dynamically_stable"]].sort_values("energy_eV_per_atom")
        for rank, (row, equivalents) in enumerate(reps):
            # How far the lowest unstable state lies below this one: > 0 means a lower phase the
            # sweep found but never stabilized. Ties (same structure, different cell/phonon
            # supercell, e.g. BaTiO3 R3m) show up as ~0, hence the margin on the flag.
            gap = 1000 * (row["energy_eV_per_atom"] - unstable["energy_eV_per_atom"].iloc[0]) if len(unstable) else np.nan
            picked.append({
                **row.to_dict(),
                "rank": rank,
                "n_selected": len(reps),
                "e_above_lowest_stable_meV_per_atom": 1000 * (row["energy_eV_per_atom"] - e_best_stable),
                "equivalent_states": ";".join(equivalents),
                "n_mp_entries_for_formula": n_entries,
                "n_states_for_formula": len(group),
                "n_stable_states_for_formula": len(stable),
                "lowest_unstable_state": (f"{unstable.iloc[0]['material_id']}/{unstable.iloc[0]['state']}"
                                          if len(unstable) else ""),
                "unstable_below_meV_per_atom": gap,
                "unstable_state_lower": bool(gap > unstable_margin_meV),
            })
    return pd.DataFrame(picked)


# Molecular oxyanions: a B/C/N/P/S centre with >= 2 O within MOLECULAR_BOND_A
# (C-O 1.28, N-O 1.25, B-O 1.37-1.48, S-O 1.47, P-O 1.54 A). Framework B-O
# bonds are >= 1.8 A, so oxynitrides (LaTiO2N) and SiO6 octahedra are kept.
MOLECULAR_CENTRES = {"B", "C", "N", "P", "S"}
MOLECULAR_BOND_A = 1.6


def non_framework_reason(structure_path: Path) -> str | None:
    """'H' or 'oxyanion' for a polymorph whose top band is internal molecular /
    hydrogen modes rather than framework phonons (O-H ~3600, C-O / N-O
    ~1000-1500 cm-1), else None."""
    atoms = ase.io.read(structure_path)
    symbols = np.array(atoms.get_chemical_symbols())
    if "H" in symbols:
        return "H"
    centres = np.nonzero(np.isin(symbols, list(MOLECULAR_CENTRES)))[0]
    oxygens = np.nonzero(symbols == "O")[0]
    if len(centres) and len(oxygens):
        d = atoms.get_all_distances(mic=True)[np.ix_(centres, oxygens)]
        if ((d < MOLECULAR_BOND_A).sum(axis=1) >= 2).any():
            return "oxyanion"
    return None


def eta_squared(X: np.ndarray, labels: np.ndarray) -> float:
    total = ((X - X.mean(0)) ** 2).sum()
    within = sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels))
    return 1 - within / total


def rms_spread(X: np.ndarray) -> float:
    """RMS distance to the centroid -- the W2 spread around the cluster's barycentre."""
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))
