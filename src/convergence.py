"""Generate and check convergence tests for Quantum ESPRESSO SCF+DFPT phonon
calculations.

Sweeps one numerical parameter -- plane-wave cutoff (ecutwfc), SCF k-point
grid density (kpts), the phonon self-consistency threshold (tr2_ph), or the
DFPT q-point grid density itself (qpts) -- while holding everything else
fixed, and reports how much the computed phonon frequencies move between
consecutive values.

ecutwfc/kpts/tr2_ph sweeps are run Gamma-only (qpts overridden to (1, 1, 1),
i.e. ldisp with a 1x1x1 grid samples only q=(0,0,0)): these parameters affect
the electronic structure and phonon self-consistency uniformly across q, so
if frequencies are converged at Gamma they're converged elsewhere too, and a
single q-point is far cheaper than a full dispersion. A qpts sweep is the one
exception -- q-grid density is exactly what's being tested, so each point
runs the real dispersion; convergence is then checked at the high-symmetry
points (Gamma/X/M/R) common to every even grid, via unstable_modes.label_q.

Generated inputs are written into data/qe_inputs/ under sweep-specific
STRUCTURE names (e.g. "CsPbI3_conv_ecutwfc_60Ry"), not a separate directory
tree, so the existing run_scf.sbatch/run_ph.sbatch scripts work against them
completely unmodified -- see generate_convergence_test()'s docstring for the
submission command it prints. This module only builds/reads QE input and
output files; it never runs pw.x/ph.x itself.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import ase.io
import numpy as np

from generate_qe_input import QE_INPUTS_DIR, QEInputConfig, generate_inputs_for_structure
from paths import CONVERGENCE_MANIFEST_DIR
from soft_mode_distortion import parse_dyn_file
from unstable_modes import HIGH_SYMMETRY_LABELS, is_numbered_dyn_file, label_q

# ecutwfc/kpts/tr2_ph sweeps run at a single q-point -- see module docstring.
GAMMA_ONLY_QPTS = (1, 1, 1)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")


def _slugify(text: str) -> str:
    """Make a formatted sweep value safe to use in a directory/STRUCTURE name."""
    return _UNSAFE_CHARS.sub("_", text)


@dataclass
class ConvergenceParameter:
    """One numerical knob to sweep, and how to apply a swept value to a QEInputConfig."""

    name: str
    values: list
    apply: Callable[[QEInputConfig, object], None]  # mutates config in place
    format_value: Callable[[object], str] = str
    # False only for a qpts sweep, where the q-grid density is itself the
    # thing under test and can't be collapsed to a single q-point.
    gamma_only: bool = True


def _set_ecutwfc(config: QEInputConfig, value: float, ecutrho_ratio: float = 8.0) -> None:
    config.ecutwfc = value
    config.ecutrho = value * ecutrho_ratio


def _set_kpts(config: QEInputConfig, value: int) -> None:
    config.kpts = (value, value, value)


def _set_tr2_ph(config: QEInputConfig, value: float) -> None:
    config.tr2_ph = value


def _set_qpts(config: QEInputConfig, value: int) -> None:
    config.qpts = (value, value, value)


ECUTWFC_SWEEP = ConvergenceParameter(
    "ecutwfc", values=[40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
    apply=_set_ecutwfc, format_value=lambda v: f"{v:g}Ry",
)
KPTS_SWEEP = ConvergenceParameter(
    "kpts", values=[2, 4, 6, 8],
    apply=_set_kpts, format_value=lambda v: f"{v}x{v}x{v}",
)
TR2_PH_SWEEP = ConvergenceParameter(
    "tr2_ph", values=[1e-12, 1e-14, 1e-16],
    apply=_set_tr2_ph, format_value=lambda v: f"{v:.0e}",
)
# Even values only: an odd grid (e.g. 3x3x3) doesn't land on X/M/R, breaking
# the cross-density comparison read_convergence_sweep relies on.
QPTS_SWEEP = ConvergenceParameter(
    "qpts", values=[2, 4, 6, 8],
    apply=_set_qpts, format_value=lambda v: f"{v}x{v}x{v}", gamma_only=False,
)

STANDARD_SWEEPS: dict[str, ConvergenceParameter] = {
    p.name: p for p in (ECUTWFC_SWEEP, KPTS_SWEEP, TR2_PH_SWEEP, QPTS_SWEEP)
}


def generate_convergence_test(
    structure_path: Path,
    parameter: ConvergenceParameter,
    base_config: QEInputConfig | Callable[[object], QEInputConfig],
) -> list[str]:
    """Write one SCF+DFPT input pair per value of `parameter`, all otherwise
    identical, into data/qe_inputs/<structure>_conv_<parameter>_<value>/.

    `base_config` is a QEInputConfig (deep-copied fresh for each value, so
    sweeps don't accumulate stray mutations) or a callable that builds one
    from the structure's ase.Atoms (e.g. generate_qe_input.pbesol_config_for).

    Returns the sweep names (== STRUCTURE identifiers), in sweep order, and
    writes a manifest recording that order + each value for
    read_convergence_sweep to use later.
    """
    structure_path = Path(structure_path)
    atoms = ase.io.read(structure_path)
    structure_name = structure_path.stem

    sweep_names = []
    manifest_points = []
    for value in parameter.values:
        config = base_config(atoms) if callable(base_config) else copy.deepcopy(base_config)
        parameter.apply(config, value)
        if parameter.gamma_only:
            config.qpts = GAMMA_ONLY_QPTS

        formatted = _slugify(parameter.format_value(value))
        sweep_name = f"{structure_name}_conv_{parameter.name}_{formatted}"
        out_dir = QE_INPUTS_DIR / sweep_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # generate_inputs_for_structure names the rendered files after the
        # structure path's stem -- copy the structure under the sweep name so
        # that stem equals STRUCTURE=<sweep_name>, matching what
        # run_scf.sbatch/run_ph.sbatch expect to find in this directory.
        renamed_structure = out_dir / f"{sweep_name}{structure_path.suffix}"
        renamed_structure.write_bytes(structure_path.read_bytes())
        generate_inputs_for_structure(renamed_structure, config, out_dir)

        sweep_names.append(sweep_name)
        manifest_points.append({"value": value, "formatted": formatted, "sweep_name": sweep_name})

    manifest_dir = CONVERGENCE_MANIFEST_DIR / structure_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"structure": structure_name, "parameter": parameter.name, "points": manifest_points}
    (manifest_dir / f"{parameter.name}.json").write_text(json.dumps(manifest, indent=2, default=str))

    return sweep_names


@dataclass
class ConvergencePoint:
    """One sweep value's resulting frequencies, keyed by q-point label.

    Gamma-only sweeps (ecutwfc/kpts/tr2_ph) always have exactly one key,
    "Gamma". A qpts sweep has one key per high-symmetry point present in that
    grid's irreducible set (see unstable_modes.HIGH_SYMMETRY_LABELS).
    """

    value: object
    formatted_value: str
    frequencies_by_q: dict[str, np.ndarray]


def read_convergence_sweep(structure_name: str, parameter_name: str, results_root: Path) -> list[ConvergencePoint]:
    """Read back frequencies for a completed sweep, in the order it was generated.

    `results_root` is wherever each sweep point's ph.x output directory (named
    after its STRUCTURE / sweep name) was synced to locally -- e.g. after
    `rsync`-ing down $SCRATCH/perovskite_phonon/ph_first/ from the cluster.
    This module never assumes $SCRATCH is set locally.
    """
    parameter = STANDARD_SWEEPS[parameter_name]
    manifest_path = CONVERGENCE_MANIFEST_DIR / structure_name / f"{parameter_name}.json"
    manifest = json.loads(manifest_path.read_text())

    points = []
    for entry in manifest["points"]:
        sweep_name = entry["sweep_name"]
        point_dir = Path(results_root) / sweep_name

        if parameter.gamma_only:
            dyn_path = point_dir / f"{sweep_name}.dyn1"
            if not dyn_path.exists():
                raise FileNotFoundError(
                    f"{dyn_path} not found -- has ph.x finished for {sweep_name}? "
                    f"(sbatch --export=ALL,STRUCTURE={sweep_name},PROJECT_ROOT=<repo root> "
                    "run_scf.sbatch, then the same for run_ph.sbatch)"
                )
            frequencies_by_q = {"Gamma": parse_dyn_file(dyn_path).frequencies_cm1}
        else:
            dyn_paths = sorted(p for p in point_dir.glob(f"{sweep_name}.dyn*") if is_numbered_dyn_file(p))
            if not dyn_paths:
                raise FileNotFoundError(f"No numbered .dyn* files found in {point_dir} -- has ph.x finished?")
            frequencies_by_q = {}
            for dyn_path in dyn_paths:
                dyn = parse_dyn_file(dyn_path)
                label = label_q(dyn.q_frac)
                if label in HIGH_SYMMETRY_LABELS:
                    frequencies_by_q[label] = dyn.frequencies_cm1

        points.append(ConvergencePoint(entry["value"], entry["formatted"], frequencies_by_q))
    return points


def check_convergence(points: list[ConvergencePoint], tol_cm1: float = 1.0) -> list[dict]:
    """Compare each sweep point's frequencies to the previous point's.

    Returns one row per point (the first has no comparison), each with the
    max absolute per-branch change across whichever q-labels are shared
    between the two points, plus a per-label breakdown. The earliest point
    after which every remaining max-diff stays under `tol_cm1` is flagged
    "converged_from_here" -- worth a look by eye either way, since a single
    coincidentally-small step doesn't guarantee the trend is monotonic.
    """
    if len(points) < 2:
        raise ValueError("Need at least two sweep points to compare.")

    rows = [{"formatted_value": points[0].formatted_value, "max_diff_cm1": None, "per_q": None, "converged_from_here": None}]
    for prev, curr in zip(points, points[1:]):
        shared_labels = sorted(set(prev.frequencies_by_q) & set(curr.frequencies_by_q))
        if not shared_labels:
            raise ValueError(
                f"No shared q-point labels between {prev.formatted_value} and "
                f"{curr.formatted_value} to compare -- for a qpts sweep this "
                "shouldn't happen with even grid values."
            )
        per_q = {}
        for label in shared_labels:
            f_prev, f_curr = prev.frequencies_by_q[label], curr.frequencies_by_q[label]
            if len(f_prev) != len(f_curr):
                raise ValueError(
                    f"Branch count mismatch at {label} between {prev.formatted_value} "
                    f"and {curr.formatted_value} -- same structure/supercell size?"
                )
            per_q[label] = float(np.max(np.abs(np.asarray(f_curr) - np.asarray(f_prev))))
        rows.append({
            "formatted_value": curr.formatted_value,
            "max_diff_cm1": max(per_q.values()),
            "per_q": per_q,
            "converged_from_here": None,
        })

    diffs = [row["max_diff_cm1"] for row in rows[1:]]
    for i in range(len(diffs)):
        if all(d is not None and d < tol_cm1 for d in diffs[i:]):
            rows[i + 1]["converged_from_here"] = True
            break

    return rows