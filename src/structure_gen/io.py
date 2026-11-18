"""Write disorder ensembles (disorder.Realization lists) as inputs for the MLIP
phonon sweep.

Each realization becomes <out_dir>/structures/<name>.cif with
name = "<strategy><index>-<supercell>_<formula>" -- the "<id>_<formula>" shape
of the MP sweep's run names (select_stable_polymorphs.py splits on the first
"_"), with no "." anywhere, since run_mlip_phonons.py takes a structure's name
up to its first dot.

manifest.csv carries what the CIFs can't: each realization's ensemble, weight,
multiplicity and correlation error, which is what configurational averages of
the resulting phonon descriptors need (disorder.ensemble_average). An
"ensemble" is one (composition, supercell, strategy): average within it, and
compare across strategies or supercells to see the finite-cell error.
mlip_structures.txt lists the CIFs, largest first, in the format
run_mlip_phonons.py --structure-list reads.

Subsets. A directory holds one subset, recorded in the manifest's "subset"
column, and write_realizations refuses to mix them:

    production   one realization per formula, so the MP sweep's downstream
                 steps (select_stable_polymorphs.py picks the best state per
                 formula) treat a mixed halide exactly like an MP entry
    noise_check  several realizations per formula (SQS seeds, or random /
                 enumerated cells), to measure the disorder spread of the
                 phonon descriptors; names carry a prefix (e.g. "nc-") so their
                 MLIP runs can't collide with production runs of the same
                 decoration in a shared --out-root
"""

from __future__ import annotations

import json
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from src.mlip.phonon_results import unfinished
from src.structure_gen.disorder import Realization


SUBSETS = ("production", "noise_check")


def same_structure(a: ase.Atoms, b: ase.Atoms, tol: float = 1e-4) -> bool:
    """Same species on the same sites (fractional, to CIF precision)."""
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        return False
    diff = a.get_scaled_positions() - b.get_scaled_positions()
    return bool(np.allclose(a.cell.cellpar(), b.cell.cellpar(), atol=1e-3) and np.all(np.abs(diff - np.round(diff)) < tol))


def realization_name(r: Realization, prefix: str = "") -> str:
    formula = Composition(r.atoms.get_chemical_formula()).reduced_formula
    return f"{prefix}{r.strategy}{r.index}-{r.supercell}_{formula}"


def write_realizations(realizations: list[Realization], out_dir: Path, mixtures: dict | None = None,
                       subset: str = "production", prefix: str = "",
                       manifest_name: str = "manifest.csv", list_name: str = "mlip_structures.txt") -> pd.DataFrame:
    """Write CIFs, and append to out_dir's manifest and structure list
    (rows for names already present are replaced). Raises before writing
    anything if out_dir already holds another subset, or if a production
    directory would end up with two realizations of one formula."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    if "_" in prefix or "." in prefix:
        raise ValueError(f"prefix {prefix!r} can't contain '_' or '.' (run names are <id>_<formula>)")
    out_dir = Path(out_dir)
    manifest_path = out_dir / manifest_name
    old = pd.read_csv(manifest_path) if manifest_path.exists() else None
    if old is not None and len(old):
        held = set(old["subset"].dropna()) if "subset" in old else {"production"}
        if held != {subset}:
            raise ValueError(f"{out_dir} holds the {', '.join(sorted(held))} subset; write {subset} elsewhere")

    names = [realization_name(r, prefix) for r in realizations]
    # Same name, different structure (e.g. another --seed): refuse. A name is
    # also an MLIP run directory, and run_mlip_phonons.py skips one that has a
    # meta.json, so a silently replaced CIF would leave results that describe
    # the old decoration.
    for r, name in zip(realizations, names):
        existing = out_dir / "structures" / f"{name}.cif"
        if existing.exists() and not same_structure(ase.io.read(existing), r.atoms):
            raise ValueError(f"{existing} already holds a different structure; remove it (and any MLIP run "
                             f"named {name}) first, or write to another directory")
    if subset == "production":
        formulas = [n.split("_", 1)[1] for n in names]
        if old is not None and len(old):
            formulas += list(old.loc[~old["name"].isin(names), "formula"])
        repeated = sorted({f for f in formulas if formulas.count(f) > 1})
        if repeated:
            raise ValueError(f"production subset would hold several realizations of {', '.join(repeated)}; "
                             "use one per formula, or the noise_check subset")

    (out_dir / "structures").mkdir(parents=True, exist_ok=True)
    rows = []
    for r, name in zip(realizations, names):
        if "." in name:
            raise ValueError(f"realization name {name!r} contains '.', which run_mlip_phonons.py would truncate")
        path = out_dir / "structures" / f"{name}.cif"
        ase.io.write(path, r.atoms, format="cif")
        formula = name.split("_", 1)[1]
        rows.append({
            "name": name, "cif": str(path.relative_to(out_dir)), "formula": formula,
            "ensemble": f"{formula}/{r.supercell}/{r.strategy}", "strategy": r.strategy, "index": r.index,
            "supercell": r.supercell, "n_atoms": len(r.atoms), "weight": r.weight,
            "multiplicity": r.multiplicity, "correlation_error": r.correlation_error, "seed": r.seed,
            "subset": subset,
            "mixtures": json.dumps(mixtures) if mixtures is not None else "",
            "cutoffs": json.dumps(r.info.get("cutoffs")),
        })
    new = pd.DataFrame(rows)

    if old is not None and len(old):
        new = pd.concat([old[~old["name"].isin(new["name"])], new], ignore_index=True)
    new.to_csv(manifest_path, index=False)
    # largest cells first: run_mlip_phonons.py strides array tasks through the list
    ordered = new.sort_values("n_atoms", ascending=False, kind="stable")["cif"]
    (out_dir / list_name).write_text("\n".join(ordered) + "\n")
    return new


def write_todo_list(out_dir: Path, sweep_root: Path, list_name: str = "mlip_structures.txt",
                    todo_name: str = "mlip_structures_new.txt") -> tuple[int, int]:
    """Write <todo_name>: the structure list minus structures with a finished
    run under sweep_root (one model's output directory). Returns (n to do, n
    listed); NotADirectoryError if sweep_root is missing."""
    lines = (out_dir / list_name).read_text().split()
    pending = set(unfinished([Path(line).stem for line in lines], sweep_root))
    todo = [line for line in lines if Path(line).stem in pending]
    (out_dir / todo_name).write_text("\n".join(todo) + ("\n" if todo else ""))
    return len(todo), len(lines)


def reference_status(formula: str, polymorphs: pd.DataFrame, queued: set[str]) -> str:
    """Whether a pure end member has an MLIP-sweep reference for mixing energies."""
    hit = polymorphs[polymorphs["formula"] == formula]
    if len(hit):
        return "stable" if hit["dynamically_stable"].any() else "unstable"
    return "queued" if formula in queued else "none"
