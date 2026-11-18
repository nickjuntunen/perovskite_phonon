"""The MLIP phonon pipeline behind scripts/mlip/run_mlip_phonons.py, one
structure and one model at a time (run_one):

    1. relax (symmetry-constrained) -> force constants -> vDOS + dispersion
    2. if imaginary: rattle, relax without the symmetry constraint, recompute
       (the first result is kept in pre_rattle/)
    3. if still imaginary: freeze the strongest soft modes into a supercell,
       relax into them, recompute (written to soft_mode/)

The building blocks are in mlip_phonons.py; this module is the order they
run in, what gets written where, and what goes into meta.json. PhononRunConfig
holds every knob, with the defaults the CLI exposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ase
import ase.io
import ase.units
import numpy as np

from src.mlip import mlip_phonons
from src.mlip.mlip_calculators import dispersion_xc, model_label
from src.qe.qe_phonon_io import parse_fc_header, parse_ph_q_grid
from src.qe.soft_mode_distortion import is_commensurate
from src.qe.unstable_modes import HIGH_SYMMETRY_LABELS, label_q


@dataclass
class PhononRunConfig:
    """Settings for run_one. Lengths in Angstrom, frequencies in cm-1."""

    # None: 'none' for a QE reference geometry (same cell as DFT), 'full' for a structure file
    relax: str | None = None
    fmax: float = 1e-3  # eV/A
    # Diagonal force-constant supercell. None: the QE run's q-grid, else the
    # smallest with every perpendicular width >= min_length
    supercell: tuple[int, int, int] | None = None
    min_length: float = 12.0
    even_supercell: bool = True  # see mlip_phonons.supercell_for_length
    displacement: float = 0.01
    nac: str = "auto"  # 'auto': LO-TO from the QE run's Born charges when there are any; 'off'
    mesh: tuple[int, int, int] = (20, 20, 20)  # vDOS q-mesh, as prepare_dos_inputs.py; with mesh_length, its cap
    # vDOS q-mesh = ceil(mesh_length / width) per axis, capped at mesh; 0 = always mesh (see _dos_mesh)
    mesh_length: float = 120.0
    deltaE: float = 0.5  # vDOS bin width
    imag_tol: float = 1.0  # below -imag_tol (band path or vDOS mesh) counts as imaginary
    # Rebuild the first attempt from the phonopy_params.yaml an unfinished run
    # left behind, if its supercell matches these settings (see _reused_attempt)
    reuse_fc: bool = False

    rattle_retry: bool = True
    # 0.01 is too small to get CsPbBr3 off its Gamma saddle before the relax already meets fmax
    rattle_stdev: float = 0.05
    rattle_seed: int = 0

    soft_mode_retry: bool = True
    soft_mode_top: int = 2  # instabilities frozen in, one (deepest) per irreducible q
    soft_mode_q: str = "high-symmetry"  # or 'any' q of the force-constant grid
    soft_mode_off_grid: bool = True  # also seed, interpolated, from special q the grid misses
    soft_mode_max_atoms: int = 640  # hexagonal K + M needs 6x3x1, i.e. 18 cells
    soft_mode_amplitude: float = 0.15  # max single-atom displacement per frozen mode
    # at 1e-3, CsPbBr3's relaxed R+M cells -- Pnma at 1e-2 -- read as P-1 and kept spurious soft modes
    soft_mode_symprec: float = 1e-2


# Where Glazer tilts (R, M) and polar/antipolar modes (Gamma, X) live. Seeding
# from a general grid q instead can land in a frustrated low-symmetry cell: for
# CsPbBr3 / mace-mp-0b3, R + (1/2, 1/2, 1/4) relaxed to P-1, still unstable at
# -17 cm-1, where R + M reached a stable P2_1/c at the same energy. These are
# label_q's cubic names, so in any cell they mean "every component 0 or 1/2";
# non-cubic special points (hexagonal K/H) come from the cell's own Bravais
# lattice instead, see PhononRunConfig.soft_mode_off_grid.
HIGH_SYMMETRY_Q = HIGH_SYMMETRY_LABELS


def read_structure_list(path: Path) -> list[str]:
    """resolve_structure arguments from a list file, one per line. A line naming a
    file relative to the list's own directory resolves there, so a list
    shipped next to its structures/ directory works wherever that pair is
    synced to; anything else (an absolute path, a qe_ph_root name) is kept
    as written."""
    specs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            candidate = path.parent / line
            specs.append(str(candidate) if candidate.exists() else line)
    return specs


def resolve_structure(spec: str, qe_ph_root: Path | None, symmetrize_symprec: float = 0.0) -> dict:
    """Reference atoms, name, and (if it came from a QE run) the QE q-grid
    and Born charges for one STRUCTURE argument: a structure file (anything
    ase.io.read understands), or a name under qe_ph_root, read from that DFPT
    run's <name>.fc. symmetrize_symprec > 0 symmetrizes a structure file
    first; QE geometries are never touched."""
    path = Path(spec)
    if path.exists():
        atoms = ase.io.read(path)
        if symmetrize_symprec > 0:
            # Make the file's symmetry exact before anything uses it. FixSymmetry
            # (relax) preserves symmetry but doesn't impose it, and phonopy looks
            # at symprec 1e-5: for 492 of the 2759 clustered MP CIFs that sees a
            # lower group than MP's (often P1), i.e. up to 6*nat displacements, no
            # space-group FC symmetrization, and symmetry-equivalent copies of one
            # soft mode taking both seed slots. 5e-2 recovers MP's group for 2685
            # of the 2759 (1e-2: 2596), moving atoms 0.026 A at the 99th percentile,
            # 0.064 A at most; 1e-1 gets 2740 but moves them up to 0.13 A. A
            # lower-symmetry minimum on the MLIP PES is still found by the rattle retry.
            from ase.spacegroup.symmetrize import refine_symmetry

            refine_symmetry(atoms, symprec=symmetrize_symprec)
        name = path.name.split(".")[0]
        return {"name": name, "atoms": atoms, "source": str(path), "q_grid": None, "nac": None}

    if qe_ph_root is None:
        raise FileNotFoundError(f"{spec}: not a file, and no QE phonon root given to look it up in")
    fc_path = qe_ph_root / spec / f"{spec}.fc"
    if not fc_path.exists():
        raise FileNotFoundError(f"{fc_path} not found -- has q2r.x run for {spec}?")
    header = parse_fc_header(fc_path)
    nac = None
    if header.born_charges is not None:
        nac = (header.epsilon, header.born_charges)
    return {
        "name": spec,
        "atoms": header.to_atoms(),
        "source": str(fc_path),
        "q_grid": parse_ph_q_grid(qe_ph_root / spec / f"{spec}.ph.in"),
        "nac": nac,
    }


def _phonon_attempt(
    start_atoms: ase.Atoms, calc, entry: dict, cfg: PhononRunConfig, relax_mode: str,
    keep_symmetry: bool = True, supercell: tuple[int, int, int] | None = None, band_path=None,
    save_dir: Path | None = None,
) -> dict:
    """One relax -> force constants -> vDOS + dispersion pass. `supercell` and
    `band_path` default to what this geometry implies; the rattle retry passes
    the first attempt's so both sample the same q-points. With `save_dir`,
    phonopy_params.yaml goes there right after the force constants, so a
    failure in the DOS/band step still leaves the GPU work on disk."""
    atoms = mlip_phonons.relax(start_atoms, calc, mode=relax_mode, fmax=cfg.fmax, keep_symmetry=keep_symmetry)
    supercell = supercell or _default_supercell(atoms, entry, cfg)

    run = mlip_phonons.compute_force_constants(atoms, calc, supercell, displacement=cfg.displacement)
    use_nac = entry["nac"] is not None and cfg.nac == "auto"
    if use_nac:
        mlip_phonons.set_nac(run.phonon, *entry["nac"])
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        mlip_phonons.save(run.phonon, save_dir / "phonopy_params.yaml")
    return _spectra_attempt(atoms, run, cfg, relax_mode, keep_symmetry, supercell, use_nac, band_path)


def _default_supercell(atoms: ase.Atoms, entry: dict, cfg: PhononRunConfig) -> tuple[int, int, int]:
    if cfg.supercell:
        return tuple(cfg.supercell)
    if entry["q_grid"] is not None:
        return entry["q_grid"]
    return mlip_phonons.supercell_for_length(atoms, cfg.min_length, even=cfg.even_supercell)


def _dos_mesh(atoms: ase.Atoms, cfg: PhononRunConfig) -> tuple[int, int, int]:
    """vDOS q-mesh: ceil(mesh_length / width) along each axis, capped at
    mesh -- a fixed q-point density rather than a fixed count. Widths as in
    mlip_phonons.supercell_for_length. With the defaults (120 A, 20 20 20) any
    cell at most 6 A wide -- every ABX3 aristotype, the QE benchmark set --
    keeps the 20^3 of prepare_dos_inputs.py.

    Why not 20^3 for everything: phonopy 4's run_mesh builds the dynamical
    matrix at every irreducible mesh q before diagonalizing any, (3*nat)^2 * 16 B
    each. For the largest MP cells (160-200 atoms, low symmetry) that was 15-45
    GB at 20^3, and job 46906416 hit its 32G limit on every task's first
    structure. At 120 A the same cells need ~1/10 of that, and their DOS
    is sampled no more coarsely than a 5-atom cell's.

    The tetrahedron DOS adds an (n_q, n_bins, n_bands) float64 array on top:
    for organic cations (N-H/C-H stretches to ~4000 cm-1, ~8000 bins at 0.5
    cm-1) that is ~34 GiB for an 84-atom cell (job 52574151), so those runs
    need 64G even at 120 A."""
    if not cfg.mesh_length:
        return tuple(cfg.mesh)
    widths = 1.0 / np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return tuple(min(cap, max(1, int(np.ceil(cfg.mesh_length / w)))) for cap, w in zip(cfg.mesh, widths))


def _reused_attempt(entry: dict, calc, cfg: PhononRunConfig, relax_mode: str, out_dir: Path) -> dict | None:
    """reuse_fc: the first attempt rebuilt from the phonopy_params.yaml a
    killed run left behind (it is saved before the DOS/band step), so only
    the DOS, band and later steps run again -- no relax, no displaced
    supercells. None (compute from scratch) if there's no such file, or if
    pre_rattle/ exists: then the top-level file may be the rattle retry's,
    not the first attempt's. Also None if the file's supercell isn't what the
    current settings would pick for its cell -- it's from a run with other settings."""
    path = out_dir / "phonopy_params.yaml"
    if not path.exists() or (out_dir / "pre_rattle").exists():
        return None
    atoms, run = mlip_phonons.load_force_constant_run(path)
    supercell = tuple(int(n) for n in np.diag(run.phonon.supercell_matrix))
    expected = _default_supercell(atoms, entry, cfg)
    if supercell != tuple(expected):
        print(f"[reuse-fc] {path}: supercell {supercell}, current settings give {expected} -- recomputing")
        return None
    # A single-point evaluation, for the residual force / pressure / energy in meta.json.
    atoms.calc = calc
    use_nac = entry["nac"] is not None and cfg.nac == "auto"
    if use_nac:
        mlip_phonons.set_nac(run.phonon, *entry["nac"])
    print(f"[reuse-fc] {entry['name']}: force constants from {path}, {run.n_displacements} displacements not recomputed")
    return _spectra_attempt(atoms, run, cfg, relax_mode, True, supercell, use_nac)


def _spectra_attempt(
    atoms: ase.Atoms, run, cfg: PhononRunConfig, relax_mode: str, keep_symmetry: bool,
    supercell: tuple[int, int, int], use_nac: bool, band_path=None,
) -> dict:
    """The vDOS + dispersion half of an attempt, from its force constants."""
    dos_mesh = _dos_mesh(atoms, cfg)
    freq_dos, dos = mlip_phonons.total_dos_cm1(run.phonon, mesh=dos_mesh, delta_e_cm1=cfg.deltaE)
    min_mesh = mlip_phonons.min_mesh_frequency_cm1(run.phonon)
    # A rattled + freely relaxed cell is no longer exactly cubic/tetragonal/...,
    # and ASE would pick a different (lower-symmetry) path for it.
    band_path = band_path or mlip_phonons.default_band_path(atoms)
    band = mlip_phonons.frequencies_cm1(run.phonon, band_path[0])

    return {
        "atoms": atoms,
        "relax": relax_mode,
        "keep_symmetry": keep_symmetry,
        "supercell": supercell,
        "run": run,
        "use_nac": use_nac,
        "dos_mesh": dos_mesh,
        "freq_dos": freq_dos,
        "dos": dos,
        "band_path": band_path,
        "band": band,
        "min_freq_cm1": min(float(band.min()), min_mesh),
    }


def _attempt_meta(attempt: dict, cfg: PhononRunConfig) -> dict:
    atoms, run = attempt["atoms"], attempt["run"]
    return {
        "relax": attempt["relax"],
        "keep_symmetry": attempt["keep_symmetry"],
        "fmax": cfg.fmax if attempt["relax"] != "none" else None,
        "max_residual_force_eV_per_A": mlip_phonons.max_force(atoms),
        # Positive = the MLIP wants this cell to expand (e.g. a DFT geometry it underbinds).
        "pressure_GPa": -float(atoms.get_stress(voigt=True)[:3].mean()) / ase.units.GPa,
        "energy_eV_per_atom": float(atoms.get_potential_energy() / len(atoms)),
        "cell_lengths_A": atoms.cell.lengths().tolist(),
        "supercell": list(attempt["supercell"]),
        "n_displaced_supercells": run.n_displacements,
        "force_eval_wall_time_s": run.wall_time_s,
        "force_constants_reused": run.wall_time_s is None,
        "nac": attempt["use_nac"],
        "dos_mesh": list(attempt["dos_mesh"]),
        "min_freq_on_path_cm1": float(attempt["band"].min()),
        "min_freq_cm1": attempt["min_freq_cm1"],
    }


def _write_attempt(attempt: dict, out_dir: Path, name: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mlip_phonons.save(attempt["run"].phonon, out_dir / "phonopy_params.yaml")
    ase.io.write(out_dir / "structure.cif", attempt["atoms"])
    mlip_phonons.write_dos(out_dir / f"{name}.dos", attempt["freq_dos"], attempt["dos"], f"{name} / {label}")
    q_frac, x, special_x, labels = attempt["band_path"]
    np.savez(out_dir / "band.npz", q_frac=q_frac, x=x, special_x=special_x, labels=labels, freqs_cm1=attempt["band"])


def _soft_mode_attempt(
    first: dict, calc, entry: dict, cfg: PhononRunConfig, out_dir: Path, name: str, label: str
) -> dict:
    """Step 3, the MLIP mirror of batch_relax_soft_mode.py: freeze the
    soft_mode_top strongest instabilities of the symmetric first attempt
    (deepest branch per irreducible q of its force-constant grid, by default
    only the high-symmetry ones -- see HIGH_SYMMETRY_Q -- plus, interpolated,
    the lattice's own special points the grid misses, e.g. hexagonal K) into
    the smallest supercell commensurate with all of them (within
    soft_mode_max_atoms), relax that freely
    (positions + cell), re-symmetrize to whatever space group it landed in,
    tighten under that symmetry, and recompute phonons.

    The phonon supercell keeps the first attempt's real-space range (its
    supercell divided by the distortion's), and the band path is the first
    attempt's, folded into the distorted cell's BZ -- so every branch the
    distorted structure has shows up along the same q-path, and any still
    negative is an instability the distortion didn't remove. No NAC (see
    below), so no LO-TO splitting in this step.
    """
    from ase.spacegroup.symmetrize import check_symmetry, refine_symmetry

    phonon = first["run"].phonon
    special = mlip_phonons.commensurate_special_points(first["atoms"]) if cfg.soft_mode_off_grid else {}
    modes = mlip_phonons.unstable_modes_on_mesh(phonon, first["supercell"], tol_cm1=cfg.imag_tol)
    for m in modes:
        m.label = next((n for n, q in special.items() if np.allclose(q, m.dyn.q_frac)), None)
    if cfg.soft_mode_q == "high-symmetry":
        # A special point that is on the grid (K on a 3x3 or 6x6 in-plane grid) stays in.
        modes = [m for m in modes if m.label is not None or label_q(m.dyn.q_frac) in HIGH_SYMMETRY_Q]
    for m in modes:
        m.label = m.label or label_q(m.dyn.q_frac)
    # What the grid misses: the lattice's own special points (hexagonal K/H on
    # an even grid), and any 0/1/2 point along an axis the supercell doesn't
    # repeat (N = 1) -- the latter always, not only when ASE's table for this
    # lattice happens to list it. Special points go first so they keep their names.
    candidates = dict(special)
    if cfg.soft_mode_off_grid:
        for q_name, q in mlip_phonons.half_integer_points().items():
            if not any(np.allclose(q, other) for other in candidates.values()):
                candidates[q_name] = q
    off_grid = mlip_phonons.symmetry_distinct_qpoints(
        phonon, {n: q for n, q in candidates.items() if not is_commensurate(q, first["supercell"])}
    )
    modes = sorted(
        modes + mlip_phonons.unstable_modes_at_qpoints(phonon, off_grid, tol_cm1=cfg.imag_tol),
        key=lambda m: m.frequency_cm1,
    )
    if not modes:
        raise RuntimeError(
            f"no unstable {cfg.soft_mode_q} q-point on the {first['supercell']} grid"
            f"{' or at ' + '/'.join(off_grid) if off_grid else ''} to freeze in"
        )
    seeds = mlip_phonons.select_seeds(modes, len(first["atoms"]), cfg.soft_mode_top, cfg.soft_mode_max_atoms)
    distorted, seed_sc = mlip_phonons.freeze_unstable_modes(seeds, amplitude_angstrom=cfg.soft_mode_amplitude)
    # A frozen-in pattern keeps its own symmetry under relaxation (the forces
    # respect it), so a "free" relax never leaves that subgroup -- and which
    # subgroup you get depends on the arbitrary vector picked inside each
    # degenerate R/M subspace. For CsPbBr3 one such pick sat in P2_1/m, a saddle
    # 0.2 meV/atom below cubic, where another reached Pnma at -22. Rattling
    # first lets the relax find the minimum whichever vector it started from.
    distorted.calc = calc
    distorted = mlip_phonons.rattle(distorted, stdev=cfg.rattle_stdev, seed=cfg.rattle_seed)

    relaxed = mlip_phonons.relax(distorted, calc, mode="full", fmax=cfg.fmax, steps=3000, keep_symmetry=False)
    sym_before = check_symmetry(relaxed, symprec=cfg.soft_mode_symprec)
    refine_symmetry(relaxed, symprec=cfg.soft_mode_symprec)
    atoms = mlip_phonons.relax(relaxed, calc, mode="full", fmax=cfg.fmax, steps=3000, keep_symmetry=True)
    spacegroup = check_symmetry(atoms, symprec=1e-5)

    seed_sc = np.array(seed_sc)
    fc_sc = tuple(int(n) for n in np.maximum(1, np.ceil(np.array(first["supercell"]) / seed_sc)))
    run = mlip_phonons.compute_force_constants(atoms, calc, fc_sc, displacement=cfg.displacement)
    # No NAC here: the only Born charges available are the reference (cubic)
    # cell's, and tiled onto the distorted cell -- unrotated with the tilted
    # octahedra, not that phase's own Z* -- their dipole term disagrees with
    # the short-range force constants near Gamma. For CsPbBr3 that alone put
    # 20/93 irreducible mesh q of an otherwise fully stable Pnma cell below
    # -1 cm-1 (down to -15.6). So no LO-TO splitting in this step.
    use_nac = False

    mesh = tuple(int(n) for n in np.maximum(1, np.ceil(np.array(first["dos_mesh"]) / seed_sc)))
    freq_dos, dos = mlip_phonons.total_dos_cm1(run.phonon, mesh=mesh, delta_e_cm1=cfg.deltaE)
    min_mesh = mlip_phonons.min_mesh_frequency_cm1(run.phonon)
    q_frac, x, special_x, labels = first["band_path"]
    band = mlip_phonons.frequencies_cm1(run.phonon, q_frac * seed_sc)

    sm_dir = out_dir / "soft_mode"
    sm_dir.mkdir(parents=True, exist_ok=True)
    mlip_phonons.save(run.phonon, sm_dir / "phonopy_params.yaml")
    ase.io.write(sm_dir / "structure.cif", atoms)
    mlip_phonons.write_dos(sm_dir / f"{name}.dos", freq_dos, dos, f"{name} / {label} / soft-mode distorted")
    np.savez(sm_dir / "band.npz", q_frac=q_frac, q_frac_supercell=q_frac * seed_sc, x=x, special_x=special_x,
             labels=labels, freqs_cm1=band)

    return {
        "seed_q": cfg.soft_mode_q,
        "off_grid_qpoints": list(off_grid),
        "seeds": [
            {"q": m.label, "q_frac": m.dyn.q_frac.tolist(), "mode_index": m.mode_index,
             "frequency_cm1": m.frequency_cm1, "interpolated": m.interpolated}
            for m in seeds
        ],
        "amplitude_A": cfg.soft_mode_amplitude,
        "rattle_stdev_A": cfg.rattle_stdev,
        "distortion_supercell": seed_sc.tolist(),
        "n_atoms": len(atoms),
        "spacegroup_after_free_relax": f"{sym_before.international} ({sym_before.number})",
        "spacegroup": f"{spacegroup.international} ({spacegroup.number})",
        "symprec": cfg.soft_mode_symprec,
        # vs the reference cell at the first attempt's geometry, same number of atoms
        "energy_gain_meV_per_atom": 1000 * float(
            atoms.get_potential_energy() / len(atoms) - first["atoms"].get_potential_energy() / len(first["atoms"])
        ),
        "cell_lengths_A": atoms.cell.lengths().tolist(),
        "max_residual_force_eV_per_A": mlip_phonons.max_force(atoms),
        "supercell": list(fc_sc),
        "n_displaced_supercells": run.n_displacements,
        "force_eval_wall_time_s": run.wall_time_s,
        "nac": use_nac,
        "dos_mesh": list(mesh),
        "min_freq_on_path_cm1": float(band.min()),
        "min_freq_cm1": min(float(band.min()), min_mesh),
    }


def run_one(
    entry: dict, calc, model_spec: str, cfg: PhononRunConfig, out_root: Path,
    overwrite: bool = False, device: str | None = None,
) -> Path:
    """Run the pipeline (module docstring) for one resolve_structure entry
    with `calc`, the calculator load_calculator(model_spec) returned, writing
    <out_root>/<model label>/<name>/ -- layout in run_mlip_phonons.py's
    docstring. A structure that already has a meta.json there is skipped
    unless `overwrite`. `device` is only recorded in meta.json."""
    name = entry["name"]
    label = model_label(model_spec)
    out_dir = Path(out_root) / label / name
    # meta.json, not phonopy_params.yaml: the top-level phonopy_params.yaml is
    # written before the soft-mode step, so a run preempted (owners) or timed
    # out during that step would otherwise be skipped on requeue, half done.
    if (out_dir / "meta.json").exists() and not overwrite:
        print(f"[{label}] {name}: already done, skipping (--overwrite to redo)")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    relax_mode = cfg.relax or ("none" if entry["q_grid"] is not None or entry["nac"] is not None else "full")
    # Both attempts save to the top level: the first is copied to pre_rattle/
    # before the retry overwrites it, and _write_attempt(final) below rewrites
    # it with whichever attempt is kept.
    first = _reused_attempt(entry, calc, cfg, relax_mode, out_dir) if cfg.reuse_fc else None
    first = first or _phonon_attempt(entry["atoms"], calc, entry, cfg, relax_mode, save_dir=out_dir)
    final = first

    # Imaginary modes: rattle off the (symmetric) saddle point, relax without
    # the symmetry constraint, and recompute once. 'none' becomes 'positions'
    # for the retry -- a rattle with nothing relaxing it out is just noise.
    rattle = None
    if first["min_freq_cm1"] < -cfg.imag_tol and cfg.rattle_retry:
        _write_attempt(first, out_dir / "pre_rattle", name, label)
        retry_relax = relax_mode if relax_mode != "none" else "positions"
        rattle = {"stdev_A": cfg.rattle_stdev, "seed": cfg.rattle_seed, "first_attempt": _attempt_meta(first, cfg)}
        print(
            f"[{label}] {name}: imaginary modes (min {first['min_freq_cm1']:.1f} cm-1), "
            f"rattling (stdev {cfg.rattle_stdev} A) and retrying with relax={retry_relax}, no symmetry constraint"
        )
        try:
            final = _phonon_attempt(
                mlip_phonons.rattle(first["atoms"], stdev=cfg.rattle_stdev, seed=cfg.rattle_seed),
                calc, entry, cfg, retry_relax, keep_symmetry=False,
                supercell=first["supercell"], band_path=first["band_path"], save_dir=out_dir,
            )
        except Exception as exc:  # e.g. the unconstrained relax didn't converge: keep the first attempt
            rattle["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{label}] {name}: rattle retry FAILED, keeping first attempt -- {rattle['error']}")

    _write_attempt(final, out_dir, name, label)

    # Still imaginary (typically R/M tilts, which no unit-cell rattle can
    # express): freeze the soft modes into a supercell and relax into them.
    # Written to soft_mode/, not the top level: the top level stays on the
    # reference cell, which is what compare_phonons.py lines up against DFT.
    soft_mode = None
    if final["min_freq_cm1"] < -cfg.imag_tol and cfg.soft_mode_retry:
        print(f"[{label}] {name}: still imaginary (min {final['min_freq_cm1']:.1f} cm-1), freezing in soft modes")
        try:
            soft_mode = _soft_mode_attempt(first, calc, entry, cfg, out_dir, name, label)
            print(
                f"[{label}] {name}: soft-mode relax -> {soft_mode['spacegroup']}, {soft_mode['n_atoms']} atoms, "
                f"{soft_mode['energy_gain_meV_per_atom']:+.2f} meV/atom, {soft_mode['n_displaced_supercells']} displacements, "
                f"min freq {soft_mode['min_freq_cm1']:.1f} cm-1 -> {out_dir / 'soft_mode'}"
            )
        except Exception as exc:  # keep steps 1-2 even if the distorted relax/phonons fail
            soft_mode = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[{label}] {name}: soft-mode step FAILED -- {soft_mode['error']}")

    meta = {
        "name": name,
        "model_spec": model_spec,
        "model_label": label,
        "dispersion": (lambda xc: f"D3BJ-{xc}" if xc else None)(dispersion_xc(model_spec)),
        "source": entry["source"],
        **_attempt_meta(final, cfg),
        "reference_cell_lengths_A": entry["atoms"].cell.lengths().tolist(),
        "displacement_A": cfg.displacement,
        "dos_mesh_length_A": cfg.mesh_length,
        "dos_deltaE_cm1": cfg.deltaE,
        "rattle_retry": rattle,
        "soft_mode": soft_mode,
        "device": device,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    run = final["run"]
    timing = "reused" if run.wall_time_s is None else f"in {run.wall_time_s:.1f}s"
    print(
        f"[{label}] {name}: relax={final['relax']}{'' if rattle is None else ' (after rattle)'}, "
        f"supercell={final['supercell']}, {run.n_displacements} displacements {timing}, "
        f"dos mesh={final['dos_mesh']}, min freq {final['min_freq_cm1']:.1f} cm-1, "
        f"NAC={'on' if final['use_nac'] else 'off'} -> {out_dir}"
    )
    return out_dir
