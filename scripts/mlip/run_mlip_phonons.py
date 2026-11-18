"""Compute MLIP phonon dispersions + vDOS (finite displacements, phonopy) for
benchmarking against this project's QE DFPT results.

Each STRUCTURE argument is either a structure file (anything ase.io.read
understands, including a QE *.scf.in), or a name to look up under
--qe-ph-root. For a name, the reference geometry is read from that DFPT run's
q2r.x force-constant file <root>/<name>/<name>.fc -- the exact cell DFPT used
-- along with its Born charges/dielectric tensor (for LO-TO, see
src/mlip/mlip_phonons.py) and its q-grid from <name>.ph.in (default supercell).
The pipeline itself is src/mlip/phonon_pipeline.py.

Every model runs with a DFT-D3(BJ) correction for its training functional
(src/mlip/mlip_calculators.py): the model label gains -D3BJ-<xc>, and a
'+d3-<xc>' / '+nod3' spec suffix overrides the functional / turns it off.

Output, per model and structure, in <out-root>/<model-label>/<name>/
(e.g. data/mlip_phonons/MACE-matpes-r2scan-omat-ft-D3BJ-r2scan/CsPbBr3/):
    phonopy_params.yaml   force constants, displacements and forces (+ NAC) -- what
                          compare_phonons.py reads. Written as soon as the force
                          constants exist, so a failure in the DOS/band step
                          after them still leaves the GPU work on disk
                          (--reuse-fc picks it up)
    structure.cif         geometry the phonons were computed on
    <name>.dos            vDOS, matdyn.x fldos layout (phonon_dos.parse_matdyn_dos)
    band.npz              dispersion on ASE's standard path for this lattice
    meta.json             model, relax mode, supercell, timings, residual force, ...
    pre_rattle/           (only if the first result had imaginary modes) the same
                          files for that first attempt -- the top-level ones are
                          then from the rattle + unconstrained-relax retry, see
                          meta.json's rattle_retry
    soft_mode/            (only if still imaginary after that) the same files for
                          the soft-mode-distorted supercell: strongest instabilities
                          frozen in and relaxed into, phonons recomputed, band.npz
                          on the reference path folded into its BZ -- see
                          meta.json's soft_mode. The top-level files stay on the
                          reference cell, which compare_phonons.py needs

Usage:
    # benchmark two MACE models on every synced DFPT run, same geometry as DFT
    uv run python3 scripts/mlip/run_mlip_phonons.py \\
        --qe-ph-root sherlock_outputs/ph_first \\
        --models models/MACE-matpes-r2scan-omat-ft.model \\
                 models/2023-12-10-mace-128-L0_energy_epoch-249.model

    # ... or relaxed on each MLIP's own PES (cell + positions), GPU
    uv run python3 scripts/mlip/run_mlip_phonons.py CsPbBr3 KNbO3 \\
        --qe-ph-root sherlock_outputs/ph_first --relax full --device cuda \\
        --models mace:medium-mpa-0

    # a standalone structure file, no QE counterpart
    uv run python3 scripts/mlip/run_mlip_phonons.py data/cubic_structures/CsPbI3.cif \\
        --models models/MACE-matpes-r2scan-omat-ft.model --relax full

    # one array task's share of the MP perovskite set
    # (list from scripts/dataset/write_mp_structure_list.py; see
    # scripts/jobs/sherlock/run_mlip_mp_array.sbatch)
    uv run python3 scripts/mlip/run_mlip_phonons.py \\
        --structure-list data/mp_perovskites/mlip_structures.txt --task-index 3 --num-tasks 99 \\
        --models models/MACE-matpes-r2scan-omat-ft.model --out-root data/mlip_phonons_mp
"""

import argparse
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlip.mlip_calculators import load_calculator, model_label
from src.paths import MLIP_PHONONS_DIR
from src.mlip.phonon_pipeline import PhononRunConfig, read_structure_list, resolve_structure, run_one

DEFAULTS = PhononRunConfig()


def _is_environment_error(exc: Exception) -> bool:
    """Errors that come from the software stack or the GPU rather than the
    structure: a package API mismatch (phonopy calling a renamed phonors
    function), a missing module, or a CUDA build with no kernels for the GPU.
    CUDA OOM is deliberately not one -- it depends on the supercell size."""
    return isinstance(exc, (AttributeError, ImportError)) or type(exc).__name__ == "AcceleratorError"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "structures", nargs="*",
        help="Structure files, or names under --qe-ph-root (default: every run there with a .fc file)",
    )
    parser.add_argument(
        "--structure-list", type=Path, default=None,
        help="Text file of STRUCTURE arguments, one per line ('#' comments ok); relative paths resolve "
             "against the list file's directory. Added to any positional STRUCTUREs",
    )
    parser.add_argument(
        "--task-index", type=int, default=0,
        help="Run only every NUM_TASKS-th structure starting from this one -- one Slurm array task's share "
             "(default: 0)",
    )
    parser.add_argument("--num-tasks", type=int, default=1, help="See --task-index (default: 1, i.e. everything)")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="MLIP specs, see src/mlip/mlip_calculators.py. Each runs with DFT-D3(BJ) for its training "
             "functional (in the output label as -D3BJ-<xc>); '+d3-<xc>' picks the functional, '+nod3' turns it off",
    )
    parser.add_argument("--qe-ph-root", type=Path, default=None, help="Synced ph_first/ root (one directory per structure)")
    parser.add_argument("--out-root", type=Path, default=MLIP_PHONONS_DIR)
    parser.add_argument(
        "--relax", choices=["none", "positions", "full"], default=None,
        help="Relax on the MLIP PES first (symmetry-constrained). Default: 'none' for QE "
             "structures (same geometry as DFT), 'full' for structure files",
    )
    parser.add_argument(
        "--symmetrize", type=float, default=5e-2,
        help="Symmetrize structure-file inputs at this symprec (Angstrom) before anything else; 0 = off. "
             "QE .fc geometries are never touched (default: 5e-2)",
    )
    parser.add_argument("--fmax", type=float, default=DEFAULTS.fmax, help="Relaxation force tolerance, eV/A (default: 1e-3)")
    parser.add_argument(
        "--supercell", type=int, nargs=3, default=None,
        help="Diagonal supercell. Default: the QE run's nq1 nq2 nq3, else smallest with every side >= --min-length",
    )
    parser.add_argument(
        "--min-length", type=float, default=DEFAULTS.min_length,
        help="Angstrom, see --supercell; compared against perpendicular widths, not cell lengths (default: 12)",
    )
    parser.add_argument(
        "--even-supercell", action=argparse.BooleanOptionalAction, default=DEFAULTS.even_supercell,
        help="Round each --min-length supercell dimension > 1 up to even, so zone-boundary q (R, M, X) are on "
             "the force-constant grid the soft-mode step seeds from (default: on)",
    )
    parser.add_argument("--displacement", type=float, default=DEFAULTS.displacement, help="Finite displacement, Angstrom (default: 0.01)")
    parser.add_argument(
        "--nac", choices=["auto", "off"], default=DEFAULTS.nac,
        help="'auto': add LO-TO using the QE run's Born charges/epsilon when available (default)",
    )
    parser.add_argument(
        "--mesh", type=int, nargs=3, default=DEFAULTS.mesh,
        help="vDOS q-mesh, or with --mesh-length its per-axis cap (default: 20 20 20, as prepare_dos_inputs.py)",
    )
    parser.add_argument(
        "--mesh-length", type=float, default=DEFAULTS.mesh_length,
        help="vDOS q-mesh = ceil(MESH_LENGTH / width) per axis, capped at --mesh; 0 = always --mesh. Keeps "
             "the largest cells' mesh within host memory (default: 120 A, i.e. 20 for cells up to 6 A wide)",
    )
    parser.add_argument(
        "--reuse-fc", action="store_true",
        help="Rebuild the first attempt from the phonopy_params.yaml an unfinished run left in the output "
             "directory (no relax, no displaced supercells), if its supercell matches the current settings",
    )
    parser.add_argument("--deltaE", type=float, default=DEFAULTS.deltaE, help="vDOS bin width, cm-1 (default: 0.5)")
    parser.add_argument(
        "--rattle-retry", action=argparse.BooleanOptionalAction, default=DEFAULTS.rattle_retry,
        help="If the first result has imaginary modes, rattle the atoms, relax without the symmetry "
             "constraint, and recompute once; the first result is kept in pre_rattle/ (default: on)",
    )
    parser.add_argument(
        "--rattle-stdev", type=float, default=DEFAULTS.rattle_stdev,
        help="Rattle amplitude, Angstrom (default: 0.05; 0.01 is too small to get CsPbBr3 off its Gamma saddle "
             "before the relax already meets --fmax)",
    )
    parser.add_argument("--rattle-seed", type=int, default=DEFAULTS.rattle_seed)
    parser.add_argument(
        "--soft-mode-retry", action=argparse.BooleanOptionalAction, default=DEFAULTS.soft_mode_retry,
        help="If still imaginary after the rattle retry, freeze the strongest soft modes into a supercell, "
             "relax into them, and recompute; written to soft_mode/ (default: on)",
    )
    parser.add_argument(
        "--soft-mode-top", type=int, default=DEFAULTS.soft_mode_top,
        help="Freeze this many instabilities, one (deepest) per irreducible q (default: 2, as batch_relax_soft_mode.py)",
    )
    parser.add_argument(
        "--soft-mode-q", choices=["high-symmetry", "any"], default=DEFAULTS.soft_mode_q,
        help="Seed only from Gamma/X/M/R (default), or from any q of the grid, as batch_relax_soft_mode.py does",
    )
    parser.add_argument(
        "--soft-mode-off-grid", action=argparse.BooleanOptionalAction, default=DEFAULTS.soft_mode_off_grid,
        help="Also seed, by interpolation, from q the force-constant grid misses: this lattice's special points "
             "with denominators <= 3 (hexagonal K/H) and every 0/1/2 point along an axis the supercell doesn't "
             "repeat; symmetry-equivalent copies dropped (default: on)",
    )
    parser.add_argument(
        "--soft-mode-max-atoms", type=int, default=DEFAULTS.soft_mode_max_atoms,
        help="Skip a further seed if the combined distorted cell would exceed this many atoms; the deepest seed "
             "is always kept (default: 640; hexagonal K + M needs 6x3x1, i.e. 18 cells)",
    )
    parser.add_argument(
        "--soft-mode-amplitude", type=float, default=DEFAULTS.soft_mode_amplitude,
        help="Max single-atom displacement per frozen mode, Angstrom (default: 0.15, as batch_relax_soft_mode.py)",
    )
    parser.add_argument(
        "--soft-mode-symprec", type=float, default=DEFAULTS.soft_mode_symprec,
        help="Tolerance for re-symmetrizing the freely relaxed distorted cell, Angstrom (default: 1e-2; at "
             "1e-3, CsPbBr3's relaxed R+M cells -- Pnma at 1e-2 -- read as P-1 and kept spurious soft modes)",
    )
    parser.add_argument(
        "--imag-tol", type=float, default=DEFAULTS.imag_tol,
        help="Frequencies below -IMAG_TOL cm-1 (path or vDOS mesh) count as imaginary (default: 1.0)",
    )
    parser.add_argument(
        "--max-env-failures", type=int, default=3,
        help="Stop a model's sweep after this many consecutive structures fail with the same environment-type "
             "error (AttributeError, ImportError, CUDA AcceleratorError); 0 = never (default: 3)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    specs = list(args.structures)
    if args.structure_list is not None:
        specs += read_structure_list(args.structure_list)
    if not specs:
        if args.qe_ph_root is None:
            parser.error("give STRUCTURE arguments, --structure-list, or --qe-ph-root to run every DFPT result found there")
        specs = sorted(d.name for d in args.qe_ph_root.iterdir() if (d / f"{d.name}.fc").exists())
    if not 0 <= args.task_index < args.num_tasks:
        parser.error(f"--task-index must be in [0, --num-tasks={args.num_tasks}), got {args.task_index}")
    # Strided, not contiguous: write_mp_structure_list.py sorts the list by
    # cell size, so every slice gets the same mix of cheap and expensive cells.
    specs = specs[args.task_index :: args.num_tasks]
    print(f"task {args.task_index}/{args.num_tasks}: {len(specs)} structure(s)")

    failures = []
    entries = []
    for spec in specs:
        try:
            entries.append(resolve_structure(spec, args.qe_ph_root, args.symmetrize))
        except Exception as exc:  # e.g. an unreadable CIF: skip it, not the whole slice
            failures.append(("<all models>", spec, exc))
            print(f"{spec}: could not read structure -- {type(exc).__name__}: {exc}")

    cfg = PhononRunConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PhononRunConfig)})
    for spec in args.models:
        try:
            label = model_label(spec)
            calc = load_calculator(spec, device=args.device)
        except Exception as exc:  # e.g. a missing checkpoint or MACE head: skip the model, not the sweep
            failures.append((spec, "<load>", exc))
            print(f"[{spec}] could not load model: {type(exc).__name__}: {exc}")
            continue
        env_streak: tuple[str, int] = ("", 0)  # (error type, consecutive structures failing with it)
        for i, entry in enumerate(entries):
            try:
                run_one(entry, calc, spec, cfg, args.out_root, overwrite=args.overwrite, device=args.device)
                env_streak = ("", 0)
            except Exception as exc:  # keep going: one bad structure shouldn't sink a sweep
                failures.append((label, entry["name"], exc))
                print(f"[{label}] {entry['name']}: FAILED -- {type(exc).__name__}: {exc}")
                kind = type(exc).__name__
                if not _is_environment_error(exc):
                    env_streak = ("", 0)
                    continue
                env_streak = (kind, env_streak[1] + 1 if env_streak[0] == kind else 1)
                if args.max_env_failures and env_streak[1] >= args.max_env_failures:
                    # ...unless it's the environment, in which case every structure
                    # after this one fails the same way (job 46355821: 2759/2759).
                    skipped = [e["name"] for e in entries[i + 1 :]]
                    print(
                        f"[{label}] {env_streak[1]} consecutive {kind}s -- looks like the environment, not the "
                        f"structures; aborting this model, {len(skipped)} structure(s) not attempted"
                    )
                    failures += [(label, name, RuntimeError(f"not attempted (aborted after {kind}s)")) for name in skipped]
                    break

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for label, name, exc in failures:
            print(f"  {label} / {name}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
