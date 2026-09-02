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

import math
import re
from dataclasses import dataclass
from fractions import Fraction
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


def is_commensurate(q_frac: np.ndarray, supercell: tuple[int, int, int], tol: float = 1e-6) -> bool:
    """Whether `supercell` is large enough to tile a frozen-phonon pattern at q_frac.

    True iff q_i * n_i is (numerically) an integer along every axis -- i.e. the
    phase exp(2*pi*i * q.R) returns to the same value after n_i primitive-cell
    translations along that axis, so the displacement pattern actually repeats
    instead of having a discontinuous seam at the supercell boundary.

    Note this is checked per axis independently: doubling every axis (e.g.
    (2, 2, 2)) is only sufficient when every nonzero q-component is a
    half-integer (as at R, M, X on a coarse grid). A ph.x run on a 4x4x4
    q-grid (see templates/qe/ph_dfpt.in.j2 / run_ph.sbatch) also produces
    q-points with quarter-integer components (1/4, 3/4), which need a
    multiple of 4 along that axis, not 2 -- "even" is not a general rule.
    """
    return all(abs(q * n - round(q * n)) < tol for q, n in zip(q_frac, supercell))


def minimal_commensurate_supercell(q_frac: np.ndarray, max_denominator: int = 48, tol: float = 1e-6) -> tuple[int, int, int]:
    """Smallest (nx, ny, nz) commensurate with q_frac, found from each component's
    rational denominator (any integer multiple of the returned size, per axis,
    is also commensurate).

    Rounds each q-component to the nearest fraction with denominator up to
    `max_denominator` (ph.x prints q in the same units it was requested in, so
    values like 1/4, 1/3, etc. are exact up to print precision) and takes that
    denominator as the multiplicity needed along that axis. Raises if a
    component doesn't look rational within `tol` at that search depth --
    widen `max_denominator` for finer q-grids.
    """
    sizes = []
    for q in q_frac:
        frac = Fraction(q).limit_denominator(max_denominator)
        if abs(float(frac) - q) > tol:
            raise ValueError(
                f"q-component {q} doesn't look rational within tol={tol} "
                f"(searched denominators up to {max_denominator}); "
                "pass a larger max_denominator if this q comes from a finer grid."
            )
        sizes.append(frac.denominator)
    return tuple(sizes)


def _phase_tiled_field(dyn: DynMatrixData, mode_index: int, supercell: tuple[int, int, int]) -> np.ndarray:
    """(len(supercell)*nat, 3) real displacement field: `mode_index`'s eigenvector at
    dyn.q_frac, Bloch-phase-tiled across the `supercell` copies of the primitive cell.
    Unnormalized -- callers scale to whatever amplitude they want.
    """
    nat = len(dyn.symbols)
    sx, sy, sz = supercell

    mode = dyn.eigenvectors[mode_index]  # (nat, 3) complex, mass-weighted
    unweighted = mode / np.sqrt(dyn.masses)[:, None]

    field = np.zeros((sx * sy * sz * nat, 3))
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                phase = np.exp(2j * np.pi * (dyn.q_frac[0] * i + dyn.q_frac[1] * j + dyn.q_frac[2] * k))
                block = np.real(unweighted * phase)
                start = ((i * sy + j) * sz + k) * nat
                field[start : start + nat] = block
    return field


def build_distorted_supercell(
    dyn: DynMatrixData,
    mode_index: int,
    supercell: tuple[int, int, int],
    amplitude_angstrom: float,
    tol: float = 1e-6,
) -> Atoms:
    """Build a supercell with `mode_index`'s eigenvector frozen in as a real displacement.

    The supercell must be large enough that dyn.q_frac is commensurate with it
    (e.g. (2, 2, 2) for q = R = (1/2, 1/2, 1/2)), or the frozen pattern will not
    tile periodically and ASE will silently build a structure with a
    discontinuous (physically meaningless) displacement at the cell boundary.
    This is checked explicitly (see is_commensurate) and raises before ASE
    ever gets a chance to build that broken structure silently.

    For combining more than one unstable mode (e.g. simultaneous instabilities
    at different q-points, as in most real Glazer octahedral-tilt systems),
    see build_multi_mode_distorted_supercell.
    """
    if not is_commensurate(dyn.q_frac, supercell, tol=tol):
        suggested = minimal_commensurate_supercell(dyn.q_frac)
        raise ValueError(
            f"Supercell {supercell} is not commensurate with q={tuple(dyn.q_frac)}: "
            "q_i * n_i must be (numerically) an integer along every axis. "
            f"Minimal commensurate supercell for this q is {suggested} "
            "(any integer multiple of each axis is also commensurate, but a "
            "multiple that isn't a multiple of the minimal size per-axis is not)."
        )

    primitive = Atoms(
        symbols=dyn.symbols,
        positions=dyn.positions_angstrom,
        cell=dyn.cell_angstrom,
        pbc=True,
    )
    supercell_atoms = primitive.repeat(supercell)

    displacements = _phase_tiled_field(dyn, mode_index, supercell)
    max_disp = np.max(np.linalg.norm(displacements, axis=1))
    if max_disp < 1e-12:
        raise ValueError("Computed displacement pattern is (numerically) zero -- check mode_index/q.")
    displacements *= amplitude_angstrom / max_disp

    supercell_atoms.positions += displacements
    return supercell_atoms


@dataclass
class ModeSeed:
    """One phonon mode to freeze in, at its own q-point and target amplitude.

    amplitude_angstrom is this mode's own max-single-atom displacement,
    applied before it's summed with any other seeds -- i.e. it's a relative
    weight between simultaneously-frozen modes, not a displacement of the
    combined pattern (see build_multi_mode_distorted_supercell).
    """

    dyn: DynMatrixData
    mode_index: int
    amplitude_angstrom: float


def _reference_geometry(dyn: DynMatrixData) -> tuple:
    """Fingerprint of the undistorted primitive cell a .dyn file was computed on
    (symbols, cell, positions), used to check that multiple .dyn files -- read
    at different q-points -- actually came from the same DFPT run before their
    modes are combined into one supercell.
    """
    return (
        tuple(dyn.symbols),
        dyn.cell_angstrom.round(6).tobytes(),
        dyn.positions_angstrom.round(6).tobytes(),
    )


def combined_commensurate_supercell(
    seeds: list["ModeSeed"], max_denominator: int = 48, tol: float = 1e-6
) -> tuple[int, int, int]:
    """Smallest supercell commensurate with every seed's q-point at once.

    Each seed alone needs at least its own minimal_commensurate_supercell; a
    supercell that satisfies several q's simultaneously needs the per-axis
    LCM of those (e.g. an R-point mode's (2, 2, 2) combined with a
    quarter-integer-q mode's (4, 1, 1) needs (4, 2, 2), not (2, 2, 2)).
    """
    per_seed_sizes = [
        minimal_commensurate_supercell(seed.dyn.q_frac, max_denominator=max_denominator, tol=tol)
        for seed in seeds
    ]
    return tuple(math.lcm(*(sizes[axis] for sizes in per_seed_sizes)) for axis in range(3))


def build_multi_mode_distorted_supercell(
    seeds: list[ModeSeed],
    supercell: tuple[int, int, int] | None = None,
    tol: float = 1e-6,
) -> Atoms:
    """Freeze several phonon modes -- potentially at different q-points -- into one
    supercell simultaneously, e.g. combining an out-of-phase R-point tilt with an
    in-phase M-point tilt into a single combined (Glazer-type) tilt pattern.

    All seeds must share the same undistorted reference structure (checked via
    _reference_geometry) -- they should, since they're just different q-points
    from the same DFPT run on the same cell, but this catches an accidental mix
    of .dyn files from different structures before it produces a nonsense atoms
    object.

    `supercell`, if given, is checked for commensurability with every seed's q
    (all must pass, not just one); if omitted, the smallest supercell
    commensurate with all of them is used (combined_commensurate_supercell).

    Each seed's displacement field is independently scaled to its own
    amplitude_angstrom, then the fields are summed -- so amplitudes set the
    relative weight between modes, not the size of the final combined
    displacement. Note the *sign* of each seed's amplitude also matters: for
    two given tilt modes, flipping one's relative sign changes which combined
    tilt pattern results (e.g. a-a-c+ vs a-a-a- in Glazer notation), not just
    its magnitude -- check the relaxed structure's symmetry rather than assume
    a given sign combination is the physically realized one.
    """
    if not seeds:
        raise ValueError("Need at least one ModeSeed.")

    reference = _reference_geometry(seeds[0].dyn)
    for seed in seeds[1:]:
        if _reference_geometry(seed.dyn) != reference:
            raise ValueError(
                "All seeds must share the same undistorted reference structure "
                "(symbols/cell/positions) -- do these .dyn files come from the "
                "same DFPT run?"
            )

    if supercell is None:
        supercell = combined_commensurate_supercell(seeds, tol=tol)
    else:
        incommensurate = [
            tuple(seed.dyn.q_frac) for seed in seeds if not is_commensurate(seed.dyn.q_frac, supercell, tol=tol)
        ]
        if incommensurate:
            raise ValueError(
                f"Supercell {supercell} is not commensurate with q={incommensurate}; "
                f"the smallest supercell commensurate with every seed is "
                f"{combined_commensurate_supercell(seeds, tol=tol)}."
            )

    dyn0 = seeds[0].dyn
    primitive = Atoms(
        symbols=dyn0.symbols,
        positions=dyn0.positions_angstrom,
        cell=dyn0.cell_angstrom,
        pbc=True,
    )
    supercell_atoms = primitive.repeat(supercell)

    total_displacements = np.zeros((len(supercell_atoms), 3))
    for seed in seeds:
        field = _phase_tiled_field(seed.dyn, seed.mode_index, supercell)
        max_disp = np.max(np.linalg.norm(field, axis=1))
        if max_disp < 1e-12:
            raise ValueError(
                f"Mode {seed.mode_index} at q={tuple(seed.dyn.q_frac)} has a "
                "(numerically) zero displacement pattern -- check mode_index/q."
            )
        field *= seed.amplitude_angstrom / max_disp
        total_displacements += field

    supercell_atoms.positions += total_displacements
    return supercell_atoms