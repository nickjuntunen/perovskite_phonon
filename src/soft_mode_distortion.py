"""Freeze an unstable phonon mode from a Quantum ESPRESSO .dyn file into a real
supercell structure.

DFPT on the small (5-atom) cubic perovskite cell detects zone-boundary
instabilities like octahedral tilting (imaginary frequencies at R, M, X) but
can't relax into them: the cell has no atomic degrees of freedom that can
express a wavevector that isn't commensurate with its own periodicity. This
module reads the eigenvector QE already diagonalized at such a q-point
directly out of the .dyn file, builds a supercell large enough to fit that
wavevector, and displaces the atoms along it -- a standard frozen-phonon seed
for a subsequent (typically much cheaper, e.g. MACE) relaxation into the true,
lower-symmetry minimum.

.dyn file format notes (QE ph.x output):
- celldm(1) and the "Basis vectors" block give the cell in units of alat (bohr).
- atomic positions are Cartesian, in units of alat.
- masses are printed in QE's internal mass units, not amu -- but since every
  atom's mass is quoted in the same unit, ratios (all this module needs) are
  unaffected, so no unit conversion is done.
- the "Diagonalizing the dynamical matrix" block gives the eigenvectors QE
  already computed, one (Re, Im) pair per Cartesian component per atom, so
  this module reads them directly rather than re-diagonalizing the raw
  dynamical matrix itself.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

BOHR_TO_ANGSTROM = 0.529177210903


@dataclass
class DynMatrixData:
    q_frac: np.ndarray  # (3,), in reciprocal-lattice-vector units (cubic: same as cartesian here)
    symbols: list[str]  # length nat, one label per atom
    masses: np.ndarray  # (nat,), QE internal units (ratios only are meaningful)
    positions_angstrom: np.ndarray  # (nat, 3), cartesian
    cell_angstrom: np.ndarray  # (3, 3)
    frequencies_cm1: np.ndarray  # (3*nat,)
    eigenvectors: np.ndarray  # (3*nat modes, nat atoms, 3), complex, mass-weighted


_FLOAT = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_dyn_file(path: Path) -> DynMatrixData:
    lines = Path(path).read_text().splitlines()

    ntyp, nat, ibrav, celldm1 = (
        int(lines[2].split()[0]),
        int(lines[2].split()[1]),
        int(lines[2].split()[2]),
        float(lines[2].split()[3]),
    )
    if ibrav != 0:
        raise NotImplementedError(f"Only ibrav=0 .dyn files are supported, got ibrav={ibrav}")
    alat_angstrom = celldm1 * BOHR_TO_ANGSTROM

    basis_start = next(i for i, line in enumerate(lines) if line.strip() == "Basis vectors") + 1
    cell_alat = np.array([[float(x) for x in lines[basis_start + i].split()] for i in range(3)])
    cell_angstrom = cell_alat * alat_angstrom

    species_re = re.compile(r"^\s*\d+\s+'([^']*)'\s+(" + _FLOAT + r")")
    species: list[tuple[str, float]] = []
    cursor = basis_start + 3
    while len(species) < ntyp:
        match = species_re.match(lines[cursor])
        if match:
            species.append((match.group(1).strip(), float(match.group(2))))
        cursor += 1

    atom_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")")
    symbols = []
    masses = []
    positions_alat = []
    while len(symbols) < nat:
        match = atom_re.match(lines[cursor])
        if match:
            type_idx = int(match.group(2))
            label, mass = species[type_idx - 1]
            symbols.append(label)
            masses.append(mass)
            positions_alat.append([float(match.group(i)) for i in (3, 4, 5)])
        cursor += 1
    positions_angstrom = np.array(positions_alat) * alat_angstrom

    q_line = next(line for line in lines if line.strip().startswith("q = ("))
    q_frac = np.array([float(x) for x in re.findall(_FLOAT, q_line)])

    diag_start = next(
        i for i, line in enumerate(lines) if line.strip() == "Diagonalizing the dynamical matrix"
    )
    freq_re = re.compile(r"freq\s*\(\s*\d+\s*\)\s*=\s*" + _FLOAT + r"\s*\[THz\]\s*=\s*(" + _FLOAT + r")\s*\[cm-1\]")
    row_re = re.compile(r"\(\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*\)")

    frequencies = []
    eigenvectors = []
    i = diag_start
    while i < len(lines) and len(frequencies) < 3 * nat:
        match = freq_re.search(lines[i])
        if match:
            frequencies.append(float(match.group(1)))
            mode_vec = []
            for row in range(nat):
                row_match = row_re.search(lines[i + 1 + row])
                values = [float(v) for v in row_match.groups()]
                mode_vec.append([complex(values[0], values[1]), complex(values[2], values[3]), complex(values[4], values[5])])
            eigenvectors.append(mode_vec)
            i += 1 + nat
        else:
            i += 1

    return DynMatrixData(
        q_frac=q_frac,
        symbols=symbols,
        masses=np.array(masses),
        positions_angstrom=positions_angstrom,
        cell_angstrom=cell_angstrom,
        frequencies_cm1=np.array(frequencies),
        eigenvectors=np.array(eigenvectors),
    )


def most_unstable_mode(dyn: DynMatrixData) -> int:
    """Index of the most negative (most unstable) frequency."""
    return int(np.argmin(dyn.frequencies_cm1))


def build_distorted_supercell(
    dyn: DynMatrixData,
    mode_index: int,
    supercell: tuple[int, int, int],
    amplitude_angstrom: float,
) -> Atoms:
    """Build a supercell with `mode_index`'s eigenvector frozen in as a real displacement.

    The supercell must be large enough that dyn.q_frac is commensurate with it
    (e.g. (2, 2, 2) for q = R = (1/2, 1/2, 1/2)), or the frozen pattern will not
    tile periodically and ASE will silently build a structure with a
    discontinuous (physically meaningless) displacement at the cell boundary.
    """
    nat = len(dyn.symbols)
    sx, sy, sz = supercell

    primitive = Atoms(
        symbols=dyn.symbols,
        positions=dyn.positions_angstrom,
        cell=dyn.cell_angstrom,
        pbc=True,
    )
    supercell_atoms = primitive.repeat(supercell)

    mode = dyn.eigenvectors[mode_index]  # (nat, 3) complex, mass-weighted
    unweighted = mode / np.sqrt(dyn.masses)[:, None]

    displacements = np.zeros((len(supercell_atoms), 3))
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                phase = np.exp(2j * np.pi * (dyn.q_frac[0] * i + dyn.q_frac[1] * j + dyn.q_frac[2] * k))
                block = np.real(unweighted * phase)
                start = ((i * sy + j) * sz + k) * nat
                displacements[start : start + nat] = block

    max_disp = np.max(np.linalg.norm(displacements, axis=1))
    if max_disp < 1e-12:
        raise ValueError("Computed displacement pattern is (numerically) zero -- check mode_index/q.")
    displacements *= amplitude_angstrom / max_disp

    supercell_atoms.positions += displacements
    return supercell_atoms
