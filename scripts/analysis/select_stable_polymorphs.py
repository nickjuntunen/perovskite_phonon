"""Pick the lowest-energy dynamically stable polymorph(s) of each formula from
an MLIP phonon sweep (scripts/mlip/run_mlip_phonons.py output).

Every <root>/<mp-id>_<formula>/ run holds up to three candidate states, all
on the same MLIP PES, so their energies per atom are directly comparable --
across states, and across MP entries of the same formula:

    symmetric   symmetry-constrained relax of the MP structure
                (pre_rattle/ if the run went on to retry, else the top level)
    rattled     rattle + unconstrained relax (top level, if a retry happened)
    soft_mode   soft modes frozen into a supercell, relaxed (soft_mode/)

A state is dynamically stable if its min frequency (band path and vDOS mesh,
meta.json's min_freq_cm1) is >= -IMAG_TOL. For each reduced formula, the
selected polymorphs are the stable states within --energy-window of the
lowest stable one, with duplicates (the same structure reached from different
MP entries or states, e.g. a soft-mode supercell of an existing Pnma entry)
merged by pymatgen's StructureMatcher. If a formula has no stable state at
all, its least unstable one (highest min frequency) is written instead,
flagged dynamically_stable=False.

--group-by run does all of the above per run instead of per formula: each run
keeps its own best state, however many runs share a formula. That's the
reduction for the mixed-halide noise check (scripts/dataset/
generate_mixed_halides.py --mode noise-check), where several SQS realizations
of one composition are separate runs, and keeping only the lowest-energy one
would discard the very spread the check is meant to measure. The *_for_formula
columns then count per run.

Outputs, in --out-dir:
    candidates.csv          every state of every run, with symmetry + MP metadata
    stable_polymorphs.csv   the selected polymorph(s), one or more per formula
    stable_polymorphs/      their structure.cif files, <formula>_<rank>_<mp-id>_<state>.cif
                            (--copy-structures)

Usage:
    uv run python3 scripts/analysis/select_stable_polymorphs.py
    uv run python3 scripts/analysis/select_stable_polymorphs.py \\
        --root mnt/mlip_phonons_mp/MACE-matpes-r2scan-omat-ft --imag-tol 5 --energy-window 2
    # new chemical clusters into the existing outputs, without rereading the sweep
    uv run python3 scripts/analysis/select_stable_polymorphs.py --reattach-clusters
    # mixed-halide noise check: one selected state per SQS realization
    uv run python3 scripts/analysis/select_stable_polymorphs.py --group-by run \
        --root <noise-check MLIP out-root>/<model> --out-dir data/mixed_halides_noise_check/selection
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MP_CHEM_CLUSTERS_CSV, MP_COMPOSITION_CLUSTERS_CSV, MP_METADATA_STABLE, MP_SWEEP_DIR, STABLE_POLYMORPHS_DIR
import pandas as pd

from src.analysis.polymorphs import collect, select


MP_COLUMNS = [
    "spacegroup_symbol", "spacegroup_number", "crystal_system", "energy_above_hull", "is_stable",
    "theoretical", "band_gap", "is_metal", "a_sites", "b_sites", "x_site",
]
CLUSTER_COLUMNS = ["cluster", "tolerance_factor", "octahedral_factor"]


def clusters_csv() -> Path | None:
    """The chemical clusters to attach: cluster_compositions.py --features-csv's
    table if it exists, else the older --mp-metadata one."""
    for path in (MP_CHEM_CLUSTERS_CSV, MP_COMPOSITION_CLUSTERS_CSV):
        if path.exists():
            return path
    return None


def attach_clusters(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    path = clusters_csv()
    if path is None:
        return df
    cl = pd.read_csv(path)[["name", *CLUSTER_COLUMNS]].rename(columns={"name": "run_name", "cluster": cluster_col})
    if cl[cluster_col].notna().all():
        cl[cluster_col] = cl[cluster_col].astype(int)
    else:
        cl[cluster_col] = cl[cluster_col].astype("Int64")
    return df.merge(cl, on="run_name", how="left")


def reattach_clusters(out_dir: Path) -> None:
    """--reattach-clusters: swap the chemical-cluster columns of an existing
    selection (candidates.csv, stable_polymorphs.csv) and of cluster_polymorphs.py's
    polymorph_clusters.csv (where the column is chem_cluster) for clusters_csv()'s,
    in place. Nothing else is recomputed, so it needs neither the sweep nor the vDOS."""
    print(f"clusters from {clusters_csv()}")
    for name, col in [("candidates.csv", "cluster"), ("stable_polymorphs.csv", "cluster"),
                      ("polymorph_clusters.csv", "chem_cluster")]:
        path = out_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path, float_precision="round_trip")  # rewrite every other value bit-for-bit
        order = list(df.columns)
        df = attach_clusters(df.drop(columns=[c for c in [col, *CLUSTER_COLUMNS[1:]] if c in df]), col)
        df[[c for c in order if c in df] + [c for c in df if c not in order]].to_csv(path, index=False)
        print(f"  {name}: {df[col].notna().sum()}/{len(df)} rows labelled")


def attach_mp_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if MP_METADATA_STABLE.exists():
        mp = pd.read_csv(MP_METADATA_STABLE)[["material_id", *MP_COLUMNS]]
        df = df.merge(mp.rename(columns={c: f"mp_{c}" for c in MP_COLUMNS}), on="material_id", how="left")
    return attach_clusters(df)


COLUMN_ORDER = [
    "formula", "rank", "n_selected", "material_id", "state", "dynamically_stable", "omega_min_cm1",
    "omega_min_path_cm1", "omega_min_q_path", "energy_eV_per_atom", "e_above_lowest_stable_meV_per_atom",
    "e_above_formula_min_meV_per_atom", "spacegroup", "spacegroup_number", "point_group", "crystal_system",
    "n_atoms", "n_atoms_primitive", "n_formula_units", "volume_A3_per_atom", "cell_abc_A", "cell_angles_deg",
    "equivalent_states", "n_mp_entries_for_formula", "n_states_for_formula", "n_stable_states_for_formula",
    "lowest_unstable_state", "unstable_below_meV_per_atom", "unstable_state_lower",
]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in COLUMN_ORDER if c in df]
    return df[first + [c for c in df if c not in first]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=MP_SWEEP_DIR, help="One model's run_mlip_phonons.py output dir")
    parser.add_argument("--out-dir", type=Path, default=STABLE_POLYMORPHS_DIR)
    parser.add_argument(
        "--imag-tol", type=float, default=1.0,
        help="States with min frequency >= -IMAG_TOL cm-1 count as dynamically stable "
             "(default: 1.0, the sweep's own --imag-tol)",
    )
    parser.add_argument(
        "--energy-window", type=float, default=1.0,
        help="Also select distinct stable polymorphs within this many meV/atom of the lowest stable one "
             "(default: 1.0; 0 = only the lowest)",
    )
    parser.add_argument(
        "--symprec", type=float, default=1e-2,
        help="spglib tolerance, Angstrom (default: 1e-2, the sweep's soft-mode symprec; rattled states "
             "relaxed without constraints read as P1 at 1e-5)",
    )
    parser.add_argument(
        "--unstable-margin", type=float, default=0.5,
        help="Flag a selected polymorph (unstable_state_lower) if an unstable state of its formula lies more than "
             "this many meV/atom below it (default: 0.5)",
    )
    parser.add_argument("--group-by", choices=["formula", "run"], default="formula",
                        help="Select per reduced formula (default) or per run (mixed-halide noise check)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--copy-structures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reattach-clusters", action="store_true",
                        help="Only replace the chemical-cluster columns of --out-dir's existing outputs "
                             "(after cluster_compositions.py --features-csv); no sweep access")
    args = parser.parse_args()
    if args.reattach_clusters:
        reattach_clusters(args.out_dir)
        return

    root = args.root.resolve()
    df = attach_mp_metadata(collect(root, args.symprec, args.workers))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    picked = select(df, args.imag_tol, args.energy_window, root, args.unstable_margin, args.group_by)
    df["dynamically_stable"] = df["omega_min_cm1"] >= -args.imag_tol
    ordered(df.sort_values(["formula", "energy_eV_per_atom"])).to_csv(
        args.out_dir / "candidates.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    ordered(picked).to_csv(args.out_dir / "stable_polymorphs.csv", index=False)

    if args.copy_structures:
        cif_dir = args.out_dir / "stable_polymorphs"
        cif_dir.mkdir(exist_ok=True)
        for _, row in picked.iterrows():
            shutil.copy(root / row["structure_path"],
                        cif_dir / f"{row['formula']}_{row['rank']}_{row['material_id']}_{row['state']}.cif")

    n_formulas = picked["formula"].nunique()
    n_stable = picked.loc[picked["dynamically_stable"], "formula"].nunique()
    print(f"{len(df)} states from {df['run_name'].nunique()} runs, {n_formulas} formulas")
    print(f"{n_stable}/{n_formulas} formulas have a dynamically stable state (omega_min >= -{args.imag_tol} cm-1)")
    print(f"{len(picked)} polymorph(s) selected; by state: {picked['state'].value_counts().to_dict()}")
    print(f"{int(picked['unstable_state_lower'].sum())} selected row(s) sit > {args.unstable_margin} meV/atom above "
          f"an unstable state of the same formula")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
