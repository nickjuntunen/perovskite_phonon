"""Read the outputs of this project's QE phonon chain (ph.x -> q2r.x ->
matdyn.x, see scripts/jobs/sherlock/run_ph.sbatch) back into Python in
the form the MLIP benchmark needs.

Three things are read here:

- The header of q2r.x's real-space force-constant file (<prefix>.fc). It
  holds the *exact* geometry the DFPT run used (alat, lattice vectors, masses,
  positions), and, when the DFPT run had epsil/zeu on, the high-frequency
  dielectric tensor and Born effective charges. Building the MLIP reference
  structure from here (rather than a CIF that may since have been regenerated)
  guarantees QE and the MLIP phonons are computed on the same cell, and the
  Born charges let the MLIP side add the same LO-TO non-analytic correction
  QE's matdyn.x applies (MLIPs don't predict Born charges themselves).
- matdyn.x's band-structure frequency file (flfrq, <prefix>.freq): q-points in
  Cartesian 2pi/alat units, then 3*nat ascending frequencies (cm-1) per q.
- The q-path vertices from a q_in_band_form matdyn.in, for axis labels.

The vDOS file (fldos) is read by phonon_dos.parse_matdyn_dos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import ase
import ase.units
import numpy as np

# q2r.x writes masses in Rydberg atomic units (amu * AMU_RY).
AMU_RY = ase.units._amu / ase.units._me / 2.0


@dataclass
class QEForceConstantHeader:
    """Everything in a <prefix>.fc file above the force-constant blocks."""

    alat_bohr: float
    at_alat: np.ndarray  # (3, 3) lattice vectors as rows, in units of alat
    symbols: list[str]  # one per atom
    masses_amu: np.ndarray  # one per atom
    positions_alat: np.ndarray  # (nat, 3) Cartesian, in units of alat
    epsilon: np.ndarray | None  # (3, 3), None if the DFPT run had epsil=False
    born_charges: np.ndarray | None  # (nat, 3, 3), Z*[atom, E-field, displacement]

    @property
    def alat_ang(self) -> float:
        return self.alat_bohr * ase.units.Bohr

    @property
    def cell_ang(self) -> np.ndarray:
        return self.at_alat * self.alat_ang

    def to_atoms(self) -> ase.Atoms:
        atoms = ase.Atoms(
            symbols=self.symbols,
            positions=self.positions_alat * self.alat_ang,
            cell=self.cell_ang,
            pbc=True,
        )
        atoms.set_masses(self.masses_amu)
        return atoms

    def q_cart_to_frac(self, q_cart_2pi_alat: np.ndarray) -> np.ndarray:
        """Convert q-points from QE's Cartesian 2pi/alat units to fractional
        coordinates of this cell's reciprocal lattice (q_frac_i = a_i . q)."""
        return np.asarray(q_cart_2pi_alat) @ self.at_alat.T


def parse_fc_header(fc_path: Path) -> QEForceConstantHeader:
    """Parse the header of a q2r.x force-constant file.

    Layout (QE 6.x/7.x, formatted output): `ntyp nat ibrav celldm(1..6)`,
    three lattice-vector lines if ibrav == 0, one `i 'Sym' mass` line per
    species, one `i ityp tau(1:3)` line per atom, then `T`/`F` for whether
    dielectric data follows (3 lines of epsilon, then per atom an index line
    and 3 lines of Z*), then the q-grid and IFC blocks, which aren't read.
    """
    lines = Path(fc_path).read_text().splitlines()
    it = iter(lines)
    head = next(it).split()
    ntyp, nat, ibrav = int(head[0]), int(head[1]), int(head[2])
    alat = float(head[3])
    if ibrav != 0:
        raise NotImplementedError(
            f"{fc_path}: ibrav={ibrav}; only ibrav=0 (explicit CELL_PARAMETERS, as every "
            "input from generate_qe_input.py uses) is supported"
        )
    at = np.array([[float(x) for x in next(it).split()] for _ in range(3)])

    species_symbols, species_masses = [], []
    for _ in range(ntyp):
        match = re.match(r"\s*\d+\s+'([^']*)'\s+(\S+)", next(it))
        assert match is not None, f"{fc_path}: malformed species line"
        species_symbols.append(match.group(1).strip())
        species_masses.append(float(match.group(2)) / AMU_RY)

    symbols, masses, positions = [], [], []
    for _ in range(nat):
        fields = next(it).split()
        ityp = int(fields[1]) - 1
        symbols.append(species_symbols[ityp])
        masses.append(species_masses[ityp])
        positions.append([float(x) for x in fields[2:5]])

    epsilon = born = None
    if next(it).strip().upper().startswith("T"):
        epsilon = np.array([[float(x) for x in next(it).split()] for _ in range(3)])
        born = np.zeros((nat, 3, 3))
        for a in range(nat):
            next(it)  # atom index line
            born[a] = [[float(x) for x in next(it).split()] for _ in range(3)]

    return QEForceConstantHeader(
        alat_bohr=alat,
        at_alat=at,
        symbols=symbols,
        masses_amu=np.array(masses),
        positions_alat=np.array(positions),
        epsilon=epsilon,
        born_charges=born,
    )


def parse_matdyn_freq(freq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a matdyn.x flfrq file: `&plot nbnd=N, nks=M /`, then for each of
    the M q-points a line of 3 Cartesian (2pi/alat) coordinates followed by N
    frequencies (cm-1, ascending, imaginary as negative) wrapped over lines.

    Returns (q_cart (M, 3), freqs_cm1 (M, N)).
    """
    text = Path(freq_path).read_text()
    match = re.search(r"nbnd\s*=\s*(\d+)\s*,\s*nks\s*=\s*(\d+)", text)
    assert match is not None, f"{freq_path}: no '&plot nbnd=, nks=' header"
    nbnd, nks = int(match.group(1)), int(match.group(2))
    values = np.array(text[text.index("/", match.end()) + 1:].split(), dtype=float)
    values = values.reshape(nks, 3 + nbnd)
    return values[:, :3], values[:, 3:]


def parse_matdyn_band_path(matdyn_in_path: Path) -> list[tuple[np.ndarray, int]]:
    """Read the q_in_band_form vertex list from a matdyn.in: after the
    namelist's closing '/', a vertex count then one `q1 q2 q3 npts` line per
    vertex. Returns [(q_vertex, npts_to_next_vertex), ...] in whatever
    coordinates the input used (crystal, in run_ph.sbatch).
    """
    text = Path(matdyn_in_path).read_text()
    tail = text[text.index("/", text.index("&input")) + 1:].split("\n")
    rows = [line.split() for line in tail if line.strip()]
    n_vertices = int(rows[0][0])
    return [
        (np.array([float(x) for x in row[:3]]), int(row[3]))
        for row in rows[1:1 + n_vertices]
    ]


def path_distances(q_cart: np.ndarray) -> np.ndarray:
    """Cumulative |dq| along a q-path (same units as q_cart), for a band
    plot's x-axis. Zero-length steps (a repeated vertex) stay flat."""
    steps = np.linalg.norm(np.diff(q_cart, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def parse_ph_q_grid(ph_in: Path) -> tuple[int, int, int] | None:
    """(nq1, nq2, nq3) from a ph.x ldisp input, or None if the file is missing
    or doesn't set all three."""
    ph_in = Path(ph_in)
    if not ph_in.exists():
        return None
    text = ph_in.read_text()
    values = [re.search(rf"nq{i}\s*=\s*(\d+)", text) for i in (1, 2, 3)]
    if not all(values):
        return None
    return tuple(int(m.group(1)) for m in values)


def run_names(root: Path, exclude_suffix: str | None = None) -> list[str]:
    """Sorted names of the per-structure run directories under a synced
    scf_first/ or ph_first/ root, leaving out derived runs whose name ends in
    `exclude_suffix` (e.g. "_tilted", "_rattled")."""
    return sorted(
        d.name for d in Path(root).iterdir()
        if d.is_dir() and not (exclude_suffix and d.name.endswith(exclude_suffix))
    )
