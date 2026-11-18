"""Finite-displacement (frozen-phonon) harmonic phonons from an MLIP, via
phonopy, producing the same two observables the QE DFPT chain produces: a
band dispersion along a q-path and a full-BZ vibrational DOS.

Why phonopy rather than ase.phonons.Phonons: ASE's Fourier interpolation sums
supercell force constants over a fixed block of lattice vectors with no
Wigner-Seitz / minimum-image folding, so a pair interaction sitting exactly
at half a supercell away gets assigned to one side only. That breaks
symmetry of the interpolated dynamical matrix and shows up as spurious
wiggles between the commensurate q-points -- an engine artifact that would
then be blamed on the MLIP. phonopy folds with multiplicities (the same idea
as q2r.x/matdyn.x's WS weights), and also uses the crystal symmetry to cut
the number of displaced supercells to the irreducible set.

Supercell <-> q-grid correspondence: an N1 x N2 x N3 supercell samples force
constants exactly at the same q-points an nq1 x nq2 x nq3 = N1 x N2 x N3 DFPT
grid does, so matching them (the default in scripts/mlip/run_mlip_phonons.py)
compares the two methods at the same real-space interaction range. Anything
that remains is then the MLIP's (plus the harmonic finite-difference step's)
error, not a difference in interpolation.

LO-TO splitting: QE's matdyn.x adds the non-analytic (dipole-dipole) term from
the DFPT Born charges and dielectric tensor. MLIPs predict neither, so an
MLIP dispersion has no LO-TO splitting unless set_nac() is given Born charges
from somewhere -- typically the same QE run (see
qe_phonon_io.QEForceConstantHeader), which isolates the MLIP's short-range
force-constant quality from the long-range electrostatics it can't model.

Units: phonopy works in THz; everything returned here is cm-1, matching
matdyn.x.
"""

from __future__ import annotations

import itertools
import math
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import ase
import numpy as np
import phonopy
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

from src.qe.soft_mode_distortion import (
    DynMatrixData,
    ModeSeed,
    build_multi_mode_distorted_supercell,
    combined_commensurate_supercell,
    minimal_commensurate_supercell,
)

THZ_TO_CM1 = 1e12 / 2.99792458e10


def relax(
    atoms: ase.Atoms,
    calc: Calculator,
    mode: str = "none",
    fmax: float = 1e-3,
    steps: int = 1000,
    symprec: float = 1e-3,
    keep_symmetry: bool = True,
) -> ase.Atoms:
    """Relax `atoms` on the MLIP's own potential-energy surface.

    mode: 'none' (phonons on the given geometry -- e.g. the exact DFT cell),
    'positions' (fixed cell), or 'full' (positions + cell). By default
    relaxation keeps the starting space group (FixSymmetry): a cubic
    aristotype that's dynamically unstable would otherwise be free to roll off
    its saddle point into a tilted structure, and then the MLIP and DFT phonons
    would no longer describe the same phase. Distorting on purpose is
    soft_mode_distortion's job. keep_symmetry=False drops the constraint --
    used after rattle() to let a structure with imaginary modes descend off
    its saddle point.

    The tight default fmax (1e-3 eV/A, vs relax_mace.py's 1e-2) is on purpose:
    residual forces on the reference cell feed straight into the soft,
    low-frequency branches this benchmark cares most about.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    if mode == "none":
        return atoms
    converged, _ = optimize(atoms, mode=mode, fmax=fmax, steps=steps, symprec=symprec, keep_symmetry=keep_symmetry)
    if not converged:
        raise RuntimeError(f"MLIP {mode} relaxation did not reach fmax={fmax} in {steps} steps")
    return atoms


def optimize(
    atoms: ase.Atoms,
    mode: str = "full",
    fmax: float = 1e-3,
    steps: int = 1000,
    symprec: float = 1e-3,
    keep_symmetry: bool = False,
    logfile: str | None = None,
) -> tuple[bool, int]:
    """LBFGS-relax `atoms` in place with the calculator already attached to
    it -- the loop under relax(), for callers that want the outcome rather
    than an exception when the step budget runs out (a structure that stopped
    at the limit is still worth writing and logging as such). Returns
    (converged, steps taken). Modes and keep_symmetry as in relax()."""
    if mode not in ("positions", "full"):
        raise ValueError(f"relax mode must be 'none', 'positions' or 'full', got {mode!r}")
    if keep_symmetry:
        atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
    target = FrechetCellFilter(atoms) if mode == "full" else atoms
    optimizer = LBFGS(target, logfile=logfile)
    optimizer.run(fmax=fmax, steps=steps)
    converged = bool(optimizer.converged())
    if keep_symmetry:
        atoms.set_constraint()
    return converged, optimizer.nsteps


def max_force(atoms: ase.Atoms) -> float:
    """Largest per-atom force norm, eV/A, from the attached calculator."""
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def supercell_for_length(atoms: ase.Atoms, min_length: float, even: bool = True) -> tuple[int, int, int]:
    """Smallest diagonal supercell whose every perpendicular width (distance
    between opposite faces, 1/|b_i|) is >= min_length.

    Widths rather than cell lengths: in a skewed cell (rhombohedral R3c,
    hexagonal, the 60-degree primitive of an Fm-3m double perovskite) the
    lengths overstate the real-space range -- for 630 of the 2759 clustered MP
    structures, lengths and widths pick different supercells.

    even: round each N > 1 up to even, so the zone-boundary q (components 0 or
    1/2: R, M, X of the aristotype) are on the force-constant grid. The
    soft-mode step seeds only from modes on that grid, so with an odd N --
    e.g. 3x3x3 for cubic CsPbBr3, a = 5.87 A -- its R/M tilts are only ever
    interpolated, never found, and the step has nothing to freeze in. N = 1
    stays 1: that cell is already >= min_length, and doubling it is 8x the
    cost for the largest cells in the set.
    """
    widths = 1.0 / np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    sizes = [max(1, int(np.ceil(min_length / width))) for width in widths]
    if even:
        sizes = [n + n % 2 if n > 1 else n for n in sizes]
    return tuple(sizes)


def rattle(atoms: ase.Atoms, stdev: float = 0.05, seed: int = 0) -> ase.Atoms:
    """Copy of `atoms` with every position displaced by Gaussian noise
    (ase.Atoms.rattle, stdev in Angstrom), keeping the calculator. Breaks the
    symmetry that holds a dynamically unstable structure on its saddle point,
    so a following relax(..., keep_symmetry=False) can descend off it."""
    rattled = atoms.copy()
    rattled.calc = atoms.calc
    rattled.rattle(stdev=stdev, seed=seed)
    return rattled


def min_mesh_frequency_cm1(phonon: Phonopy) -> float:
    """Lowest frequency (cm-1, imaginary as negative) on the mesh from the
    last run_mesh -- e.g. the one total_dos_cm1 just ran."""
    return float(np.min(phonon.get_mesh_dict()["frequencies"]) * THZ_TO_CM1)


def _to_phonopy_atoms(atoms: ase.Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.array(atoms.cell),
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )


def _to_ase_atoms(cell: PhonopyAtoms) -> ase.Atoms:
    return ase.Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


@dataclass
class ForceConstantRun:
    phonon: Phonopy
    n_displacements: int
    wall_time_s: float | None  # None: loaded from disk (load_force_constant_run), not timed


def compute_force_constants(
    atoms: ase.Atoms,
    calc: Calculator,
    supercell: tuple[int, int, int],
    displacement: float = 0.01,
    symprec: float = 1e-5,
) -> ForceConstantRun:
    """Harmonic force constants by central finite differences in a diagonal
    supercell of `atoms`.

    The unit cell is used as phonopy's primitive cell as-is (no automatic
    primitive reduction), so fractional q-points mean the same thing here as
    in QE's ibrav=0 input built from the same cell.

    Central (+/-) differences are forced rather than phonopy's 'auto': with
    relax mode 'none' the reference cell is generally not at the MLIP's
    minimum, and the residual force then cancels exactly in a +/- pair
    rather than leaking into the force constants.
    """
    phonon = Phonopy(
        _to_phonopy_atoms(atoms),
        supercell_matrix=np.diag(supercell),
        primitive_matrix="P",  # identity; phonopy >= 4 reads None as "auto" (reduce to primitive)
        symprec=symprec,
    )
    phonon.generate_displacements(distance=displacement, is_plusminus=True)

    start = time.perf_counter()
    forces = []
    for displaced in phonon.supercells_with_displacements:
        sc = _to_ase_atoms(displaced)
        sc.calc = calc
        forces.append(sc.get_forces().copy())
    wall = time.perf_counter() - start

    phonon.forces = np.array(forces)
    phonon.produce_force_constants(show_drift=False)
    phonon.symmetrize_force_constants(show_drift=False)
    return ForceConstantRun(phonon, len(forces), wall)


def set_nac(phonon: Phonopy, epsilon: np.ndarray, born_charges: np.ndarray) -> None:
    """Attach Born charges + dielectric tensor for the LO-TO correction (see
    module docstring). born_charges is (nat, 3, 3) in the unit cell's atom
    order. `factor` is Hartree*Bohr, the conversion phonopy's default
    (VASP-like eV/Angstrom -- i.e. ASE) unit system uses."""
    phonon.nac_params = {
        "born": np.asarray(born_charges),
        "dielectric": np.asarray(epsilon),
        "factor": get_physical_units().Hartree * get_physical_units().Bohr,
    }


def save(phonon: Phonopy, path: Path) -> None:
    """phonopy_params.yaml with the force constants (and NAC params, if set)
    embedded, so compare_phonons.py can re-evaluate at any q without the
    MLIP."""
    phonon.save(str(path), settings={"force_constants": True})


def load(path: Path) -> Phonopy:
    return phonopy.load(str(path), produce_fc=False, symmetrize_fc=False)


def load_force_constant_run(path: Path) -> tuple[ase.Atoms, ForceConstantRun]:
    """A save()d phonopy_params.yaml back as (unit cell, ForceConstantRun):
    the geometry the force constants were computed on and the force constants
    themselves, without evaluating any forces. The returned atoms have no
    calculator."""
    phonon = load(path)
    return _to_ase_atoms(phonon.unitcell), ForceConstantRun(phonon, len(phonon.displacements), None)


def _skip(lines, n: int) -> None:
    next(itertools.islice(lines, n, n), None)


def _parse_fc_rows(lines: list[str], n_cols: int) -> np.ndarray:
    text = "".join(line for line in lines if "#" not in line)
    text = text.replace("- [", " ").replace("]", " ").replace(",", " ")
    return np.array(text.split(), dtype=float).reshape(n_cols, 3, 3)


def load_compact(path: Path) -> Phonopy:
    """load(), for when only the compact force constants (one row per unit-cell
    atom) are needed -- everything that evaluates frequencies does.

    save() writes the full (n_super, n_super) force constants, which make up
    nearly all of a phonopy_params.yaml (median ~100 MB, up to ~2.6 GB in the
    MP sweep) and which PyYAML builds into Python floats one by one -- minutes
    and tens of GB per large file. Here the header goes through phonopy as usual
    (minus the displacement dataset, not needed with force constants present)
    and the force-constant block is streamed: the rows of non-unit-cell atoms
    are skipped unparsed (each 3x3 element is a fixed 4-line block), and the
    kept rows are parsed by numpy.
    """
    header = []
    with open(path) as f:
        lines = iter(f)
        in_displacements = False
        for line in lines:
            if line.startswith("force_constants:"):
                break
            if line[:1] not in (" ", "-", "\n", ""):  # a top-level key
                in_displacements = line.startswith("displacements:")
            if not in_displacements:
                header.append(line)
        else:
            raise ValueError(f"{path}: no force_constants block")

        fc_meta = {}
        for line in lines:
            key, _, value = line.strip().partition(":")
            if key == "elements":
                break
            fc_meta[key] = value.strip().strip('"')
        n_rows, n_cols = (int(x) for x in fc_meta["shape"].strip("[] ").split(","))

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as tmp:
            tmp.writelines(header)
            tmp.flush()
            phonon = phonopy.load(tmp.name, produce_fc=False, symmetrize_fc=False)

        p2s = np.asarray(phonon.primitive.p2s_map)
        wanted = range(n_rows) if fc_meta.get("format") == "compact" else sorted(p2s)
        rows, current = {}, 0
        for row in wanted:
            _skip(lines, 4 * n_cols * (row - current))
            rows[row] = _parse_fc_rows(list(itertools.islice(lines, 4 * n_cols)), n_cols)
            current = row + 1

    keys = range(n_rows) if fc_meta.get("format") == "compact" else p2s
    phonon.force_constants = np.array([rows[k] for k in keys])
    return phonon


def frequencies_cm1(phonon: Phonopy, q_frac: np.ndarray) -> np.ndarray:
    """(nq, 3*nat) ascending frequencies (cm-1, imaginary as negative) at
    fractional q-points.

    With NAC on, the non-analytic term at exactly q = 0 depends on the
    direction of approach. Each Gamma point in the list takes its direction
    from its neighbour on the path -- what matdyn.x does in q_in_band_form --
    so an MLIP+NAC band plot splits LO/TO at Gamma the same way QE's does.
    """
    q_frac = np.asarray(q_frac, dtype=float)
    phonon.run_qpoints(q_frac)
    freqs = np.array(phonon.get_qpoints_dict()["frequencies"])

    if phonon.nac_params is not None:
        for i in np.where(np.linalg.norm(q_frac, axis=1) < 1e-8)[0]:
            neighbours = [j for j in (i + 1, i - 1) if 0 <= j < len(q_frac)]
            directions = [q_frac[j] - q_frac[i] for j in neighbours]
            directions = [d for d in directions if np.linalg.norm(d) > 1e-8]
            if not directions:
                continue
            phonon.run_qpoints([q_frac[i]], nac_q_direction=directions[0])
            freqs[i] = phonon.get_qpoints_dict()["frequencies"][0]
    return np.sort(freqs * THZ_TO_CM1, axis=1)


def total_dos_cm1(
    phonon: Phonopy, mesh: tuple[int, int, int] = (20, 20, 20), delta_e_cm1: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedron-method total vDOS on a Gamma-centred mesh -- the same
    method and default mesh/bin width as phonon_dos.render_matdyn_dos_input
    -- as (freq_cm1, states per cm-1 per unit cell, integrating to 3*nat)."""
    phonon.run_mesh(list(mesh), is_gamma_center=True)
    phonon.run_total_dos(freq_pitch=delta_e_cm1 / THZ_TO_CM1, use_tetrahedron_method=True)
    dos = phonon.get_total_dos_dict()
    return np.asarray(dos["frequency_points"]) * THZ_TO_CM1, np.asarray(dos["total_dos"]) / THZ_TO_CM1


def write_dos(path: Path, freq_cm1: np.ndarray, dos: np.ndarray, label: str) -> None:
    """Two-column file in matdyn.x's fldos layout, so
    phonon_dos.parse_matdyn_dos reads MLIP and QE vDOS alike."""
    header = f"# {label}: frequency (cm-1), DOS (states/cm-1/cell)"
    np.savetxt(path, np.column_stack([freq_cm1, dos]), fmt="%14.6f %16.8e", header=header, comments="")


def default_band_path(atoms: ase.Atoms, npoints: int = 200):
    """ASE's standard special-point path for this cell's Bravais lattice, as
    (q_frac, x, special_x, special_labels) -- used for standalone MLIP runs.
    Benchmarks against QE instead reuse QE's own q-points (see
    scripts/mlip/compare_phonons.py)."""
    path = atoms.cell.bandpath(npoints=npoints)
    x, special_x, labels = path.get_linear_kpoint_axis()
    labels = [r"$\Gamma$" if label == "G" else label for label in labels]
    return path.kpts, x, special_x, labels


# --- soft-mode freezing: MLIP counterpart of scripts/soft_modes/batch_relax_soft_mode.py ---


def _mode_as_dyn(phonon: Phonopy, q_frac: np.ndarray, freqs_cm1: np.ndarray, eigvecs: np.ndarray) -> DynMatrixData:
    """One q-point's phonopy eigenvectors repackaged as the DynMatrixData a QE
    .dyn file parses into, so soft_mode_distortion freezes MLIP and DFT modes
    with the same code (amplitude convention, commensurability checks).

    Phase convention: phonopy's Bloch phase includes each atom's position in
    the cell, u(j, l) ~ e_j exp(2 pi i q.(R_l + tau_j)), where QE's (and
    soft_mode_distortion's tiling) is exp(2 pi i q.R_l) only -- so e_QE_j =
    e_phonopy_j exp(2 pi i q.tau_j). Skipping this flips the sign of e.g.
    half the X atoms at R/M, i.e. freezes in a different pattern entirely.
    Each mode's arbitrary global phase is then rotated to make it as real as
    possible, since the tiling keeps only the real part.
    """
    cell = phonon.primitive
    nat = len(cell.symbols)
    tau_phase = np.exp(2j * np.pi * cell.scaled_positions @ q_frac)  # (nat,)
    modes = eigvecs.T.reshape(3 * nat, nat, 3) * tau_phase[None, :, None]
    for m in modes:
        m *= np.exp(-0.5j * np.angle(np.sum(m * m)))
    return DynMatrixData(
        q_frac=np.asarray(q_frac, dtype=float),
        symbols=list(cell.symbols),
        masses=np.asarray(cell.masses),
        positions_angstrom=cell.positions,
        cell_angstrom=np.asarray(cell.cell),
        frequencies_cm1=np.asarray(freqs_cm1),
        eigenvectors=modes,
    )


@dataclass
class MLIPUnstableMode:
    dyn: DynMatrixData
    mode_index: int
    frequency_cm1: float
    # Off the force-constant grid: frequency/eigenvector from Fourier
    # interpolation, not exact (see unstable_modes_at_qpoints).
    interpolated: bool = False
    label: str | None = None


def unstable_modes_on_mesh(phonon: Phonopy, mesh: tuple[int, int, int], tol_cm1: float = 1.0) -> list[MLIPUnstableMode]:
    """Deepest unstable branch at each irreducible q of a Gamma-centred mesh,
    most unstable first -- the MLIP analogue of
    top_modes_per_q(rank_unstable_modes(dyn_dir), n_per_q=1) on a DFPT run.
    Use the force-constant supercell's own grid: those q are where the
    finite-displacement force constants are exact, as DFPT's are on its grid.
    """
    phonon.run_mesh(list(mesh), is_gamma_center=True, with_eigenvectors=True, is_mesh_symmetry=True)
    data = phonon.get_mesh_dict()
    found = []
    for q, freqs, eigvecs in zip(data["qpoints"], data["frequencies"], data["eigenvectors"]):
        freqs_cm1 = np.asarray(freqs) * THZ_TO_CM1
        band = int(np.argmin(freqs_cm1))
        if freqs_cm1[band] < -tol_cm1:
            found.append(MLIPUnstableMode(_mode_as_dyn(phonon, np.asarray(q), freqs_cm1, eigvecs), band, float(freqs_cm1[band])))
    return sorted(found, key=lambda m: m.frequency_cm1)


def commensurate_special_points(atoms: ase.Atoms, max_denominator: int = 3, tol: float = 1e-4) -> dict[str, np.ndarray]:
    """The non-Gamma special points of `atoms`' own Bravais lattice (ASE's
    tables, in this cell's fractional reciprocal coordinates) that a modest
    diagonal supercell can freeze in: every component rational with
    denominator <= max_denominator.

    The default 3 is what admits hexagonal K (1/3, 1/3, 0) and H (1/3, 1/3,
    1/2) -- trimerizing modes like YMnO3's K3 -- while dropping FCC's W/K/U
    (quarters, eighths: 4x-8x supercells of an already 10-atom primitive) and
    RHL's B/P/Q, whose coordinates depend on the rhombohedral angle and are
    generally irrational, i.e. commensurate with no supercell at all.
    """
    try:
        points = atoms.cell.bandpath(npoints=0).special_points
    except Exception:  # ASE can fail to classify a badly distorted cell; seed from the grid alone
        return {}
    found = {}
    for name, q in points.items():
        q = np.asarray(q, dtype=float)
        fracs = [Fraction(x).limit_denominator(max_denominator) for x in q]
        if np.allclose(q, 0.0) or any(abs(float(f) - x) > tol for f, x in zip(fracs, q)):
            continue
        found[name] = np.array([float(f) for f in fracs])
    return found


def half_integer_points() -> dict[str, np.ndarray]:
    """The 7 nonzero q with every component 0 or 1/2 (fractional), keyed by
    their coordinates. These are where any cubic-aristotype zone-boundary
    point (X, M, R -- tilts, Jahn-Teller M2+/R3+, antipolar modes) lands in
    a cell that's an integer supercell of the cubic one, so they're always
    worth checking -- including along an axis the force-constant supercell
    doesn't repeat, where they're off the grid."""
    return {
        "(" + ", ".join(f"{x:g}" for x in q) + ")": np.array(q, dtype=float)
        for q in itertools.product((0.0, 0.5), repeat=3)
        if any(q)
    }


def symmetry_distinct_qpoints(phonon: Phonopy, qpoints: dict[str, np.ndarray], tol: float = 1e-6) -> dict[str, np.ndarray]:
    """`qpoints` with symmetry-equivalent duplicates dropped (first one kept),
    under the crystal's point group plus time reversal (q ~ -q). Without
    this, (1/2, 0, 0) and (0, 1/2, 0) in a tetragonal cell -- one
    instability -- would be taken as two seeds; unstable_modes_on_mesh
    avoids the same thing by working on irreducible q only."""
    rotations = np.asarray(phonon.primitive_symmetry.reciprocal_operations)

    def equivalent(q1: np.ndarray, q2: np.ndarray) -> bool:
        for image in np.concatenate([rotations @ q1, -(rotations @ q1)]):
            diff = image - q2
            if np.all(np.abs(diff - np.round(diff)) < tol):
                return True
        return False

    kept: dict[str, np.ndarray] = {}
    for name, q in qpoints.items():
        if not any(equivalent(q, other) for other in kept.values()):
            kept[name] = q
    return kept


def unstable_modes_at_qpoints(
    phonon: Phonopy, qpoints: dict[str, np.ndarray], tol_cm1: float = 1.0
) -> list[MLIPUnstableMode]:
    """Deepest unstable branch at each named q (fractional), most unstable
    first -- the off-grid counterpart of unstable_modes_on_mesh, for q-points
    the force-constant supercell doesn't contain (e.g. hexagonal K on an even
    grid). These come from Fourier interpolation of the force constants, so
    they're estimates, not the exact values on-grid modes are: an
    interpolated instability that isn't real shows up downstream as a frozen
    pattern that relaxes back out (energy gain ~ 0) rather than as a wrong
    final structure, since the distorted cell's phonons are recomputed from
    scratch."""
    if not qpoints:
        return []
    names = list(qpoints)
    phonon.run_qpoints(np.array([qpoints[n] for n in names]), with_eigenvectors=True)
    data = phonon.get_qpoints_dict()
    found = []
    for name, freqs, eigvecs in zip(names, data["frequencies"], data["eigenvectors"]):
        freqs_cm1 = np.asarray(freqs) * THZ_TO_CM1
        band = int(np.argmin(freqs_cm1))
        if freqs_cm1[band] < -tol_cm1:
            dyn = _mode_as_dyn(phonon, qpoints[name], freqs_cm1, eigvecs)
            found.append(MLIPUnstableMode(dyn, band, float(freqs_cm1[band]), interpolated=True, label=name))
    return sorted(found, key=lambda m: m.frequency_cm1)


def select_seeds(modes: list[MLIPUnstableMode], n_atoms: int, top: int, max_atoms: int) -> list[MLIPUnstableMode]:
    """Up to `top` of `modes` (already ranked, most unstable first) whose
    combined commensurate supercell stays within max_atoms. The deepest mode
    is always taken; each later one is skipped, not just the rest truncated,
    if adding it would overshoot -- e.g. hexagonal K (3x3x1) + M (2x1x1)
    needs 6x3x1, so for a 40-atom cell (720 atoms, over the default 640) the
    M seed gives way to the next mode that fits, such as A (3x3x2 with K)."""
    chosen, sizes = [], []
    for mode in modes:
        if len(chosen) == top:
            break
        trial = sizes + [minimal_commensurate_supercell(mode.dyn.q_frac)]
        combined = [math.lcm(*(s[axis] for s in trial)) for axis in range(3)]
        if chosen and n_atoms * math.prod(combined) > max_atoms:
            continue
        chosen.append(mode)
        sizes = trial
    return chosen


def freeze_unstable_modes(
    modes: list[MLIPUnstableMode], amplitude_angstrom: float = 0.15
) -> tuple[ase.Atoms, tuple[int, int, int]]:
    """Freeze `modes` into the smallest supercell commensurate with all of
    them (soft_mode_distortion.build_multi_mode_distorted_supercell); returns
    the distorted supercell and its size in primitive cells."""
    seeds = [ModeSeed(dyn=m.dyn, mode_index=m.mode_index, amplitude_angstrom=amplitude_angstrom) for m in modes]
    supercell = combined_commensurate_supercell(seeds)
    return build_multi_mode_distorted_supercell(seeds, supercell=supercell), supercell
