"""Generate Quantum ESPRESSO SCF + DFPT inputs for the cubic subset of the
Materials Project perovskite dataset (data/mp_perovskites/).

Reads data/mp_perovskites/metadata_stable.csv (structurally verified, near-hull
entries; see scripts/dataset/filter_stable_mp_perovskites.py), restricted to
crystal_system == "Cubic",
the population directly comparable to data/cubic_structures (built as ideal
cubic ABX3 by construction via scripts/qe/generate_inputs.py). Lower-symmetry MP
entries are left out of QE-input generation for now.

Every structure is symmetrized to its primitive standard cell at MP's own
symmetry tolerance (symmetry.MP_SYMPREC) before writing. MP's CIFs are only symmetric to
~1e-3, well outside pw.x's much tighter tolerance, so as-downloaded cells make
pw.x find near-but-not-exact operations and abort ("d_matrix: D_S (l=2) for
this symmetry operation is not orthogonal"). A few CIFs are also supercells of
their primitive cell. Structures whose symmetrized spacegroup doesn't match
metadata's spacegroup_symbol are skipped and reported.

Unlike data/cubic_structures, where every ion's oxidation state comes from a
NOMAD lookup, these are real, arbitrary MP compositions -- so whether a
structure needs nspin=2/smearing (and therefore no Born charges/LO-TO, see
pbesol_config_for) is decided per-structure via pymatgen's BVAnalyzer
bond-valence guess. BVAnalyzer can't resolve valences for a chunk of these
(mixed/exotic compositions); those structures are still generated, just
without that magnetism check, and are called out in the printed summary
rather than silently passed off as checked.

Structures with an element missing from the local PBEsol pseudopotential
library (data/pseudopotentials/) are skipped and reported, same as
scripts/qe/generate_inputs.py.

Usage:
    uv run python3 scripts/qe/generate_mp_inputs.py
    uv run python3 scripts/qe/generate_mp_inputs.py --max-structures 20  # smoke test
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.paths import MP_METADATA_STABLE, MP_QE_INPUTS_DIR, MP_STRUCTURES_DIR
from src.analysis.composition_features import bond_valence_ions
from src.data.mp_metadata import mp_run_name
from src.qe.generate_qe_input import generate_inputs_for_atoms, pbesol_config_for
from src.utils.symmetry import primitive_standard_structure


def cubic_rows() -> list[dict]:
    with open(MP_METADATA_STABLE, newline="") as f:
        return [row for row in csv.DictReader(f) if row["crystal_system"] == "Cubic"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-structures", type=int, default=None, help="Cap the number generated, for a quick test run"
    )
    return parser.parse_args()


def generate_all(max_structures: int | None = None) -> None:
    rows = cubic_rows()
    if max_structures is not None:
        rows = rows[:max_structures]
    print(f"{len(rows)} cubic-crystal-system MP entries to process (from {MP_METADATA_STABLE})")

    MP_QE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    built = []
    skipped = []
    valence_unresolved = []
    for row in rows:
        name = mp_run_name(row["material_id"], row["formula_pretty"])
        cif_path = MP_STRUCTURES_DIR / Path(row["cif_path"]).name

        structure, spacegroup = primitive_standard_structure(Structure.from_file(cif_path))
        if spacegroup != row["spacegroup_symbol"]:
            skipped.append((name, f"symmetrized to {spacegroup}, metadata says {row['spacegroup_symbol']}"))
            continue

        ions = bond_valence_ions(structure)
        if ions is None:
            valence_unresolved.append(name)

        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            config = pbesol_config_for(atoms, ions=ions)
            generate_inputs_for_atoms(atoms, name, config, MP_QE_INPUTS_DIR / name)
        except ValueError as error:
            skipped.append((name, str(error)))
            continue
        built.append(name)

    print(f"Built {len(built)} QE input pairs in {MP_QE_INPUTS_DIR}.")
    if valence_unresolved:
        print(
            f"{len(valence_unresolved)} generated without bond-valence-based magnetism "
            "detection (BVAnalyzer couldn't resolve oxidation states) -- occupations "
            "left at the default 'fixed':"
        )
        for name in valence_unresolved:
            print(f"  {name}")
    if skipped:
        print(f"Skipped {len(skipped)} (spacegroup mismatch, or missing pseudopotential/cutoff data):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    args = parse_args()
    generate_all(max_structures=args.max_structures)
