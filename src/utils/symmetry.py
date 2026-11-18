"""Space-group identification and symmetrization shared across the dataset,
QE-input and polymorph-analysis code.
"""

from __future__ import annotations

import ase
import spglib
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# pymatgen's default, and the tolerance MP assigns spacegroup/crystal_system at.
MP_SYMPREC = 0.1


def crystal_system(number: int) -> str:
    """Lowercase crystal system of a space-group number (MP's metadata capitalises it)."""
    for upper, name in [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                        (167, "trigonal"), (194, "hexagonal"), (230, "cubic")]:
        if number <= upper:
            return name
    raise ValueError(number)


def symmetry(atoms: ase.Atoms, symprec: float) -> dict:
    """spglib's space group, point group, crystal system and primitive-cell
    atom count of `atoms` at `symprec` (Angstrom)."""
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:  # spglib gives up on badly distorted cells; P1 is the honest answer
        return {"spacegroup": "P1", "spacegroup_number": 1, "point_group": "1", "crystal_system": "triclinic",
                "n_atoms_primitive": len(atoms)}
    return {
        "spacegroup": ds.international,
        "spacegroup_number": int(ds.number),
        "point_group": ds.pointgroup,
        "crystal_system": crystal_system(int(ds.number)),
        "n_atoms_primitive": len(set(ds.mapping_to_primitive)),
    }


def primitive_standard_structure(structure: Structure, symprec: float = MP_SYMPREC) -> tuple[Structure, str]:
    """Primitive standard cell with exact symmetry, plus its spacegroup symbol."""
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    return analyzer.get_primitive_standard_structure(), analyzer.get_space_group_symbol()
