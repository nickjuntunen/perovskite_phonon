"""Identify and rank unstable (imaginary-frequency) phonon branches across a
set of Quantum ESPRESSO ph.x .dyn files.

Each .dyn file (see soft_mode_distortion.parse_dyn_file) is the full
dynamical-matrix diagonalization -- all 3*nat branches -- at one q-point.
"The strongest instability" means the (q, branch) pair with the most negative
frequency, not the q-point with the most negative branches: a q-point can
have several shallow negative branches, or one very deep one, and those call
for different treatment when picking modes to seed into a supercell (see
scripts/relax_soft_mode.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from soft_mode_distortion import DynMatrixData, parse_dyn_file

_DYN_INDEX_RE = re.compile(r"\.dyn(\d+)$")


def is_numbered_dyn_file(path) -> bool:
    """Whether `path` is a per-q-point <prefix>.dynN file (N >= 1) that
    parse_dyn_file can actually read.

    Excludes <prefix>.dyn0, the extra file ph.x writes once per ldisp run:
    it holds the q-grid dimensions, the count of irreducible q-points, and
    their coordinates -- no dynamical matrix, no atomic positions, nothing
    parse_dyn_file's regexes expect -- so globbing "*.dyn*" would otherwise
    hand it to parse_dyn_file and crash rather than skip it.
    """
    match = _DYN_INDEX_RE.search(str(path))
    return bool(match) and int(match.group(1)) >= 1

# Canonical cubic (Pm-3m) high-symmetry q-points, given as the sorted
# absolute value of each fractional component. That pattern is invariant
# under the full O_h point group (axis permutations + independent per-axis
# sign flips), so it identifies the symmetry orbit regardless of which
# representative of the star a given .dyn file happens to report -- e.g.
# (0, 0, 1/2), (0, 1/2, 0), and (0, 0, -1/2) are all "X".
_HIGH_SYMMETRY_PATTERNS: dict[tuple[float, float, float], str] = {
    (0.0, 0.0, 0.0): "Gamma",
    (0.0, 0.0, 0.5): "X",
    (0.0, 0.5, 0.5): "M",
    (0.5, 0.5, 0.5): "R",
}

# The label set every even-density cubic q-grid (2x2x2, 4x4x4, 6x6x6, ...)
# is guaranteed to include, used elsewhere (src/convergence.py) to compare
# frequencies across sweeps of different q-grid density at points common to
# all of them.
HIGH_SYMMETRY_LABELS = frozenset(_HIGH_SYMMETRY_PATTERNS.values())


def label_q(q_frac, tol: float = 1e-4) -> str:
    """Human-readable label for a q-point: a high-symmetry name (Gamma/X/M/R)
    if it matches one of cubic Pm-3m's special points, else its fractional
    coordinates, e.g. "(0.25, 0, 0)".
    """
    pattern = tuple(sorted(abs(round(float(x), 6)) for x in q_frac))
    for ref_pattern, label in _HIGH_SYMMETRY_PATTERNS.items():
        if all(abs(a - b) < tol for a, b in zip(pattern, ref_pattern)):
            return label
    return "(" + ", ".join(f"{float(x):.4g}" for x in q_frac) + ")"


@dataclass
class UnstableMode:
    """One negative-frequency phonon branch found in a .dyn file."""

    dyn_path: Path
    dyn: DynMatrixData
    mode_index: int
    frequency_cm1: float
    q_label: str

    def __repr__(self) -> str:
        return (
            f"UnstableMode({self.dyn_path.name}, mode={self.mode_index}, "
            f"q={self.q_label}, freq={self.frequency_cm1:.2f} cm-1)"
        )


def find_unstable_modes(dyn_paths, freq_tol_cm1: float = -1e-3) -> list[UnstableMode]:
    """Parse each .dyn file in `dyn_paths` and collect every branch below
    `freq_tol_cm1`.

    The small default tolerance (rather than a bare `< 0`) exists so an
    acoustic branch at Gamma that's numerically -1e-5 cm-1 from imperfect
    acoustic-sum-rule enforcement isn't reported as a genuine instability.
    Any path that isn't a numbered per-q .dynN file (see is_numbered_dyn_file
    -- this silently skips a stray .dyn0 grid-manifest file) is skipped.
    """
    found = []
    for path in dyn_paths:
        path = Path(path)
        if not is_numbered_dyn_file(path):
            continue
        dyn = parse_dyn_file(path)
        q_label = label_q(dyn.q_frac)
        for mode_index, freq in enumerate(dyn.frequencies_cm1):
            if freq < freq_tol_cm1:
                found.append(UnstableMode(path, dyn, mode_index, float(freq), q_label))
    return found


def rank_unstable_modes(dyn_dir, pattern: str = "*.dyn*", freq_tol_cm1: float = -1e-3) -> list[UnstableMode]:
    """Every unstable branch among the .dyn files in `dyn_dir` matching
    `pattern`, sorted most-unstable (most negative frequency) first.

    Any <prefix>.dyn0 grid-manifest file the glob picks up is excluded (see
    is_numbered_dyn_file) rather than passed to parse_dyn_file, which would
    otherwise crash on it.
    """
    dyn_paths = sorted(p for p in Path(dyn_dir).glob(pattern) if is_numbered_dyn_file(p))
    if not dyn_paths:
        raise FileNotFoundError(f"No numbered .dyn files (excluding .dyn0) matching {pattern!r} in {dyn_dir}")
    modes = find_unstable_modes(dyn_paths, freq_tol_cm1=freq_tol_cm1)
    return sorted(modes, key=lambda m: m.frequency_cm1)


def summarize_by_q(modes: list[UnstableMode]) -> dict[str, list[UnstableMode]]:
    """Group unstable modes by q-point label, each group sorted most-unstable first."""
    groups: dict[str, list[UnstableMode]] = {}
    for mode in modes:
        groups.setdefault(mode.q_label, []).append(mode)
    for group in groups.values():
        group.sort(key=lambda m: m.frequency_cm1)
    return groups


def top_modes_per_q(modes: list[UnstableMode], n_per_q: int = 1) -> list[UnstableMode]:
    """The `n_per_q` most unstable modes at each distinct q, flattened and
    re-sorted most-unstable-overall first.

    Useful for picking a set of ModeSeeds that spans several q-points without
    also dragging in every shallow branch at each one -- e.g. n_per_q=1 with
    --top 2 in scripts/find_unstable_modes.py picks the single deepest branch
    at each of the two most unstable q-points, a reasonable starting point
    for scripts/relax_soft_mode.py's --auto-seed.
    """
    grouped = summarize_by_q(modes)
    picked = [mode for group in grouped.values() for mode in group[:n_per_q]]
    return sorted(picked, key=lambda m: m.frequency_cm1)


def as_seed_arg(mode: UnstableMode, amplitude_angstrom: float = 0.15) -> str:
    """Format an UnstableMode as a scripts/relax_soft_mode.py --seed argument."""
    return f"{mode.dyn_path}:{mode.mode_index}:{amplitude_angstrom}"