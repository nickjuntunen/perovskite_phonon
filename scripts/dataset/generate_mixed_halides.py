"""Generate mixed-halide perovskite structures A B (X1_{1-x} X2_x)3 for the
MLIP phonon sweep (src/structure_gen), as one of two separate subsets:

    production   (--mode production, default) one SQS per composition, in
                 data/mixed_halides/. Each composition is then a single run that
                 goes through the same steps as an MP entry -- symmetric relax,
                 rattle and soft-mode retries, select_stable_polymorphs.py's
                 lowest-energy stable state -- since the decoration is fixed
                 and only the tilt pattern varies between those states.
    noise check  (--mode noise-check) --n-sqs SQS seeds per composition
                 (default 5), optionally plus random / enumerated cells, in
                 data/mixed_halides_noise_check/, names prefixed "nc-". Their
                 phonon descriptors measure the disorder spread one SQS leaves
                 unresolved: the same role the gauntlet's model-to-model spread
                 plays for MLIP error. Select per realization
                 (select_stable_polymorphs.py --group-by run), never per formula,
                 and give the MLIP runs their own --out-root.

Seeds start at --seed in both modes, so with the same --seed and --supercell
the noise check's nc-sqs0 is the production structure itself, one member of
its own disorder ensemble. If the spread turns out large compared to the
dataset's, production should move to an ensemble average instead
(structure_gen.ensemble_average over the manifest weights).

Compositions come from one of two sources (--source):

    mp-clusters  every cation framework -- fixed A and B site species, from the
                 structure check -- with at least two halide end members in the
                 clustering list (data/mp_composition_clusters.csv), each halide
                 pair present mixed at every --fractions x. 176 frameworks (155
                 double perovskites A2BB'X6, 21 ABX3), so B-site ordering and
                 mixed-valence B sites (CsInX3 = In+/In3+) come with the
                 structure. Every end member is in the finished MLIP sweep, so
                 every mixture has a mixing-energy reference. The parent is the
                 MP structure of the majority-halide end member (the entry the
                 sweep selected for its formula), as its conventional cell,
                 rescaled to the Vegard volume between the two end members'
                 MLIP-relaxed volumes; the supercell is the smallest diagonal one
                 with >= 24 halide sites (structure_gen.supercell_for_sites). No
                 formability screen: these end members are verified perovskites.
    enumerated   the --a x --b x --halides grid on the cubic aristotype (Vegard
                 lattice parameter) in --supercell, with the formability screen
                 below. Structures start cubic; tilts develop when
                 run_mlip_phonons.py relaxes them -- the default 2x2x2 cell is
                 commensurate with every cubic tilt q-point.

x is always the fraction of the second halide of a pair, in F, Cl, Br, I order
for mp-clusters, as given for enumerated. A composition already present in a
production directory under another name -- the 15 cubic-parent CsPb mixtures,
computed before mp-clusters existed -- is kept, not duplicated.

Screening (enumerated only). A mixture is generated only if both of its pure end members pass
the perovskite formability window, Goldschmidt t in [--t-min, --t-max] and
octahedral factor mu >= --mu-min (structure_gen.formability; same radii as
the Vegard lattice parameter). The end members, not the alloy: averaged radii
hide size mismatch. This drops Ge (mu ~ 0.33-0.40: real CsGeX3 are distorted,
off-centered rhombohedral) and Na (t ~ 0.75-0.79). --no-screen generates
everything.

compositions.csv records every composition considered, generated or not, with
what's needed to choose among them later rather than here: t and mu of both end
members and of the alloy, the halide radius mismatch |r1 - r2| / min(r1, r2)
(F/I is 0.71, Cl/Br 0.08; large mismatches are unlikely to form solid
solutions), and whether each end member has a reference in the MP MLIP sweep
("stable" / "unstable" / "queued" / "none") -- without one there is no
mixing energy.

Output, in --out:
    structures/<prefix><strategy><i>-<supercell>_<formula>.cif
    manifest.csv          subset, ensemble, weight, multiplicity, correlation error per structure
    compositions.csv      screening and reference status per composition
    mlip_structures.txt   for run_mlip_phonons.py --structure-list
    mlip_structures_new.txt  (--exclude-done) only structures without a finished run

Usage:
    # full production set from the clustering list, plus what the existing sweep hasn't run
    uv run python3 scripts/dataset/generate_mixed_halides.py --source mp-clusters \\
        --fractions 1/6 1/3 1/2 2/3 5/6 --exclude-done mnt/mlip_phonons_mixed/MACE-matpes-r2scan-omat-ft
    # an enumerated grid on the cubic aristotype instead
    uv run python3 scripts/dataset/generate_mixed_halides.py \\
        --a Cs Rb K --b Pb Sn --halides F Cl Br I --fractions 1/6 1/3 1/2 2/3 5/6
    # disorder noise check on a representative handful
    uv run python3 scripts/dataset/generate_mixed_halides.py --mode noise-check \\
        --a Cs --b Pb Sn --halides Cl Br I --fractions 1/3 1/2
    # noise check on a stratified subset of the production series (mp-clusters parents)
    uv run python3 scripts/dataset/select_noise_check_series.py
    uv run python3 scripts/dataset/generate_mixed_halides.py --mode noise-check --source mp-clusters \\
        --series-csv data/mixed_halides_noise_check_mp/series.csv --fractions 1/6 1/3 1/2 2/3 5/6 \\
        --out data/mixed_halides_noise_check_mp
"""

import argparse
import itertools
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
from pymatgen.core import Composition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MIXED_HALIDES_DIR, MIXED_HALIDES_NOISE_CHECK_DIR, MP_COMPOSITION_CLUSTERS_CSV, MP_DIR, MP_METADATA_STABLE, STABLE_POLYMORPHS_DIR
from src.structure_gen import (
    ParentLattice,
    commensurate_tilt_points,
    conventional_cell,
    cubic_parent,
    enumerate_orderings,
    formability,
    n_decorations,
    random_configurations,
    scaled_to_volume,
    sqs,
    supercell_for_sites,
    write_realizations,
)
from src.structure_gen.io import realization_name, reference_status, same_structure, write_todo_list
from src.structure_gen.parent import expected_formula
from src.data.perovskite_builder import lookup_ion
from src.structure_gen.disorder import MAX_DECORATIONS

MODES = {
    "production": {"out": MIXED_HALIDES_DIR, "subset": "production", "prefix": "", "n_sqs": 1},
    "noise-check": {"out": MIXED_HALIDES_NOISE_CHECK_DIR, "subset": "noise_check",
                    "prefix": "nc-", "n_sqs": 5},
}


def realizations_for(parent, label, args, n_sqs, supercell=None):
    supercell = args.supercell if supercell is None else supercell
    matrix = supercell if isinstance(supercell, str) else np.diag(supercell)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # icet's self-interaction notes for small cells
        if "sqs" in args.strategy:
            out += sqs(parent, matrix, n=n_sqs, seed=args.seed)
        if "random" in args.strategy:
            out += random_configurations(parent, matrix, n=args.n_random, seed=args.seed)
        if "enumerate" in args.strategy:
            if n_decorations(parent, args.enumerate_supercell) > MAX_DECORATIONS:
                print(f"{label}: too many decorations to enumerate in {args.enumerate_supercell}")
            else:
                classes = enumerate_orderings(parent, args.enumerate_supercell)
                if len(classes) > args.enumerate_max_classes:
                    print(f"{label}: {len(classes)} enumerated classes > --enumerate-max-classes "
                          f"{args.enumerate_max_classes}, skipped")
                else:
                    out += classes
    return out


def supercell_label(supercell) -> str:
    return supercell if isinstance(supercell, str) else "x".join(str(int(n)) for n in supercell)


def enumerated_jobs(fractions, polymorphs, queued, args):
    """(row, make_parent) per composition from the --a / --b / --halides grid,
    with the end-member formability screen; make_parent is None if screened out."""
    pairs = [(x1, x2) for i, x1 in enumerate(args.halides) for x2 in args.halides[i + 1:]]
    for a in args.a:
        for b in args.b:
            for x1, x2 in pairs:
                end = {h: formability(a, b, h) for h in (x1, x2)}
                forms = {h: args.t_min <= f["t"] <= args.t_max and f["mu"] >= args.mu_min for h, f in end.items()}
                r1, r2 = (lookup_ion("X", h).ionic_radius_angstrom for h in (x1, x2))
                refs = {h: reference_status(Composition(f"{a}{b}{h}3").reduced_formula, polymorphs, queued)
                        for h in (x1, x2)}
                for x in fractions:
                    alloy = formability(a, b, {x1: 1 - x, x2: x})
                    row = {"composition": f"{a}{b}({x1}{1 - x:.3g}{x2}{x:.3g})3", "source": "enumerated",
                           "A": a, "B": b, "X1": x1, "X2": x2, "x2": round(x, 4), "t": alloy["t"], "mu": alloy["mu"],
                           "t_X1": end[x1]["t"], "mu_X1": end[x1]["mu"], "t_X2": end[x2]["t"], "mu_X2": end[x2]["mu"],
                           "halide_mismatch": abs(r1 - r2) / min(r1, r2),
                           "ref_X1": refs[x1], "ref_X2": refs[x2], "parent": "cubic aristotype", "structures": ""}
                    if not args.no_screen and not all(forms.values()):
                        failing = [f"{a}{b}{h}3 (t={end[h]['t']:.2f}, mu={end[h]['mu']:.2f})" for h in forms if not forms[h]]
                        row["status"] = "screened out: " + ", ".join(failing)
                        yield row, None
                        continue
                    make = (lambda a=a, b=b, x1=x1, x2=x2, x=x:
                            (cubic_parent(a, b, {x1: 1 - x, x2: x}), args.supercell))
                    yield row, make


def mp_cluster_jobs(fractions, polymorphs, args):
    """(row, make_parent) per composition between two halide end members of one
    cation framework (same A and B site species, from the structure check) in
    the clustering list. The parent is the MP structure of the majority-halide
    end member -- the entry the sweep selected for that formula -- in its
    conventional cell, rescaled to the Vegard volume between the two end
    members' MLIP-relaxed volumes; the supercell is supercell_for_sites'."""
    order = ["F", "Cl", "Br", "I"]
    clusters = pd.read_csv(args.clusters_csv)
    meta = pd.read_csv(args.mp_metadata)
    meta["name"] = meta["cif_path"].str.split("/").str[-1].str.rsplit(".", n=1).str[0]
    entries = clusters[["name"]].merge(meta, on="name")
    entries = entries[entries["x_site"].isin(order)].copy()
    entries["reduced"] = entries["formula_pretty"].map(lambda f: Composition(f).reduced_formula)
    entries["framework"] = entries["a_sites"] + " | " + entries["b_sites"]
    selected = polymorphs.sort_values("rank").groupby("formula").first() if len(polymorphs) else pd.DataFrame()
    mp_dir = args.mp_metadata.parent

    def end_member(group):
        """The entry the sweep selected for this formula, else the lowest-hull one."""
        formula = group["reduced"].iloc[0]
        if formula in selected.index:
            hit = group[group["material_id"] == selected.loc[formula, "material_id"]]
            if len(hit):
                return hit.iloc[0], selected.loc[formula]
        return group.sort_values("energy_above_hull").iloc[0], (selected.loc[formula] if formula in selected.index else None)

    wanted = None
    if args.series_csv is not None:  # (framework, X1, X2) triples, e.g. a noise-check subset
        wanted = set(pd.read_csv(args.series_csv)[["framework", "X1", "X2"]].itertuples(index=False, name=None))
    for framework, fam in entries.groupby("framework", sort=True):
        halides = sorted(set(fam["x_site"]), key=order.index)
        halides = [h for h in halides if h in args.halides]
        for x1, x2 in itertools.combinations(halides, 2):
            if wanted is not None and (framework, x1, x2) not in wanted:
                continue
            (e1, s1), (e2, s2) = (end_member(fam[fam["x_site"] == h]) for h in (x1, x2))
            r1, r2 = (lookup_ion("X", h).ionic_radius_angstrom for h in (x1, x2))
            v1 = s1["volume_A3_per_atom"] if s1 is not None else e1["volume"] / e1["nsites"]
            v2 = s2["volume_A3_per_atom"] if s2 is not None else e2["volume"] / e2["nsites"]
            for x in fractions:
                majority, (entry, sel) = (x2, (e2, s2)) if x > 0.5 else (x1, (e1, s1))
                row = {"composition": f"{framework.replace(' | ', '/')}:({x1}{1 - x:.3g}{x2}{x:.3g})", "source": "mp-clusters",
                       "framework": framework, "structure_type": entry["formula_anonymous"],
                       "A": entry["a_sites"], "B": entry["b_sites"], "X1": x1, "X2": x2, "x2": round(x, 4),
                       "end_X1": e1["material_id"] + "_" + e1["reduced"], "end_X2": e2["material_id"] + "_" + e2["reduced"],
                       "parent": f"{entry['material_id']} ({entry['spacegroup_symbol']})", "parent_halide": majority,
                       "halide_mismatch": abs(r1 - r2) / min(r1, r2),
                       "ref_X1": "stable" if s1 is not None and s1["dynamically_stable"] else ("unstable" if s1 is not None else "none"),
                       "ref_X2": "stable" if s2 is not None and s2["dynamically_stable"] else ("unstable" if s2 is not None else "none"),
                       "volume_per_atom_vegard": (1 - x) * v1 + x * v2, "structures": ""}

                def make(entry=entry, majority=majority, x1=x1, x2=x2, x=x, volume=row["volume_per_atom_vegard"]):
                    atoms = scaled_to_volume(conventional_cell(ase.io.read(mp_dir / entry["cif_path"])), volume)
                    parent = ParentLattice(atoms, {majority: {x1: 1 - x, x2: x}})
                    return parent, supercell_for_sites(parent)
                yield row, make


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=list(MODES), default="production")
    parser.add_argument("--source", choices=["enumerated", "mp-clusters"], default="enumerated",
                        help="Compositions from the --a/--b/--halides grid (cubic aristotype parents), or from every "
                             "cation framework with >= 2 halide end members in the clustering list (MP parents)")
    parser.add_argument("--clusters-csv", type=Path, default=MP_COMPOSITION_CLUSTERS_CSV)
    parser.add_argument("--mp-metadata", type=Path, default=MP_METADATA_STABLE)
    parser.add_argument("--series-csv", type=Path, default=None,
                        help="mp-clusters only: generate just the series listed (columns framework, X1, X2), "
                             "e.g. scripts/dataset/select_noise_check_series.py's output")
    parser.add_argument("--a", nargs="+", default=["Cs"], help="A-site cations")
    parser.add_argument("--b", nargs="+", default=["Pb"], help="B-site cations")
    parser.add_argument("--halides", nargs="+", default=["F", "Cl", "Br", "I"],
                        help="Halides; every pair is mixed (mp-clusters: every pair present in a framework)")
    parser.add_argument("--fractions", nargs="+", default=["1/3", "1/2", "2/3"],
                        help="Fractions x of the second halide of each pair, e.g. 1/3 (must fit the supercell)")
    parser.add_argument("--strategy", nargs="+", choices=["sqs", "random", "enumerate"], default=["sqs"],
                        help="noise-check only; production is always one SQS")
    parser.add_argument("--n-sqs", type=int, default=None,
                        help="SQS seeds per composition (noise-check only; default 5)")
    parser.add_argument("--supercell", default="2x2x2", help="SUPERCELL_PRESETS key for sqs/random (default: 2x2x2)")
    parser.add_argument("--enumerate-supercell", default="sqrt2xsqrt2x2",
                        help="SUPERCELL_PRESETS key for enumerate (default: sqrt2xsqrt2x2, 12 X sites)")
    parser.add_argument("--enumerate-max-classes", type=int, default=50)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Default: data/mixed_halides[_noise_check]")
    parser.add_argument("--t-min", type=float, default=0.8)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--mu-min", type=float, default=0.414)
    parser.add_argument("--no-screen", action="store_true", help="Generate every composition, formable end members or not")
    parser.add_argument("--reference-polymorphs", type=Path,
                        default=STABLE_POLYMORPHS_DIR / "stable_polymorphs.csv",
                        help="MP sweep selection, for end-member reference status")
    parser.add_argument("--reference-queue", type=Path,
                        default=MP_DIR / "mlip_structures_new.txt",
                        help="MP structures submitted but not yet in --reference-polymorphs")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many compositions (for a quick test)")
    parser.add_argument("--exclude-done", type=Path, default=None, metavar="SWEEP_ROOT",
                        help="Also write mlip_structures_new.txt: structures without <SWEEP_ROOT>/<name>/meta.json")
    args = parser.parse_args()

    mode = MODES[args.mode]
    if args.mode == "production":
        if args.strategy != ["sqs"] or args.n_sqs not in (None, 1):
            parser.error("production is one SQS per composition; use --mode noise-check for more realizations")
    n_sqs = args.n_sqs if args.n_sqs is not None else mode["n_sqs"]
    out_dir = args.out or mode["out"]

    for cell in {args.supercell} | ({args.enumerate_supercell} if "enumerate" in args.strategy else set()):
        missing = [q for q, ok in commensurate_tilt_points(cell).items() if not ok]
        if missing:
            print(f"note: {cell} is not commensurate with tilt q-point(s) {', '.join(missing)}")

    polymorphs = pd.read_csv(args.reference_polymorphs) if args.reference_polymorphs.exists() else pd.DataFrame(
        columns=["formula", "dynamically_stable"])
    queued = ({Composition(Path(line).stem.split("_", 1)[1]).reduced_formula
               for line in args.reference_queue.read_text().split()} if args.reference_queue.exists() else set())

    fractions = [float(Fraction(f)) for f in args.fractions]
    if args.source == "mp-clusters":
        jobs = mp_cluster_jobs(fractions, polymorphs, args)
    else:
        jobs = enumerated_jobs(fractions, polymorphs, queued, args)

    existing = pd.read_csv(out_dir / "manifest.csv") if (out_dir / "manifest.csv").exists() else pd.DataFrame(
        columns=["name", "formula"])
    rows, n_written = [], 0
    for row, make_parent in itertools.islice(jobs, args.limit):
        rows.append(row)
        if make_parent is None:  # screened out; row["status"] says why
            continue
        label = row["composition"]
        try:
            parent, supercell = make_parent()
            formula = expected_formula(parent, supercell)
        except ValueError as exc:  # e.g. a fraction the supercell can't represent
            row["status"] = f"skipped: {exc}"
            print(f"{label}: skipped ({exc})")
            continue
        row["formula"] = formula
        try:
            realizations = realizations_for(parent, label, args, n_sqs, supercell)
        except ValueError as exc:
            row["status"] = f"skipped: {exc}"
            print(f"{label}: skipped ({exc})")
            continue
        if not realizations:
            row["status"] = "no realizations"
            continue
        if mode["subset"] == "production":
            # A composition already in this directory as a different structure -- another parent,
            # lattice parameter or supercell, e.g. the 15 cubic-parent CsPb mixtures -- keeps it,
            # and its finished MLIP run. Only an identical regeneration passes through.
            names = [realization_name(r, mode["prefix"]) for r in realizations]
            other = existing[existing["formula"] == formula]
            kept = [n for n in other["name"] if n not in names or not same_structure(
                ase.io.read(out_dir / "structures" / f"{n}.cif"), realizations[names.index(n)].atoms)]
            if kept:
                row["status"], row["structures"] = "existing", ";".join(kept)
                print(f"{label}: already present as {';'.join(kept)} (a different structure), kept")
                continue
        written = write_realizations(realizations, out_dir, mixtures=parent.mixtures,
                                     subset=mode["subset"], prefix=mode["prefix"])
        existing = written
        names = written["name"].iloc[-len(realizations):]
        row["status"], row["structures"] = "generated", ";".join(names)
        row["supercell"], row["n_atoms"] = supercell_label(supercell), len(realizations[0].atoms)
        row["correlation_error"] = realizations[0].correlation_error
        n_written += len(realizations)
        errors = ", ".join(f"{r.correlation_error:.3f}" for r in realizations[:6])
        print(f"{label}: {len(realizations)} structure(s), {row['n_atoms']} atoms, correlation error {errors}"
              + (" ..." if len(realizations) > 6 else ""), flush=True)

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compositions.csv"
    if path.exists():  # keep compositions from earlier calls with other --a/--b/--halides
        old = pd.read_csv(path)
        table = pd.concat([old[~old["composition"].isin(table["composition"])], table], ignore_index=True)
    table.to_csv(path, index=False)
    screened = table["status"].str.startswith("screened").sum()
    print(f"{n_written} structures ({mode['subset']}) -> {out_dir}; "
          f"{screened} composition(s) screened out, see {path.name}")
    if args.exclude_done is not None:
        try:
            n_todo, n_all = write_todo_list(out_dir, args.exclude_done)
        except NotADirectoryError as exc:
            raise SystemExit(str(exc))
        print(f"--exclude-done: {n_all - n_todo} of {n_all} already finished; {n_todo} -> mlip_structures_new.txt")


if __name__ == "__main__":
    main()
