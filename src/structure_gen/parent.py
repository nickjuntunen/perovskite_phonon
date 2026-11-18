"""Parent lattices and mixing specifications for substitutionally disordered
perovskites -- mixed halides first (CsPb(Br,I)3), but any sublattice can mix
(Cs/Rb on A, Sn/Pb on B).

A ParentLattice is an ordered structure plus a rule for which of its species
become mixtures: {"Br": {"Br": 2/3, "I": 1/3}} turns every Br site into a
Br/I site at that composition. Every atom of a mixed species belongs to one
sublattice, so on a low-symmetry parent (a relaxed Pnma end member, with
apical and equatorial X on different Wyckoff orbits) the target composition is
over all X sites together -- the random-alloy assumption, not a site-
selective one.

Supercells. Disorder lives in a supercell of the parent, and the choice
constrains what the later MLIP relaxation can do: octahedral tilts are
zone-boundary instabilities of the cubic cell (M = (1/2, 1/2, 0), R =
(1/2, 1/2, 1/2)), so a cubic-parent supercell can only tilt if those q are
commensurate with it. SUPERCELL_PRESETS lists the useful ones;
commensurate_tilt_points reports which tilt q-points a given cell admits.
The supercell also fixes which compositions are reachable: with n sites on a
mixed sublattice, fractions come in steps of 1/n (2x2x2: 24 X sites, steps of
1/24), and check_commensurate refuses anything else rather than rounding.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction

import ase
import numpy as np
from ase.build import make_supercell

from src.chemistry import octahedral_factor, tolerance_factor
from src.data.perovskite_builder import lookup_ion

SITE_LABELS = ("A", "B", "X")

# Supercell matrices of the 5-atom cubic aristotype.
SUPERCELL_PRESETS: dict[str, np.ndarray] = {
    # 40 atoms, 24 X sites. Every cubic M and R point is commensurate, so any
    # Glazer tilt system can develop on relaxation.
    "2x2x2": np.diag([2, 2, 2]),
    # 20 atoms, 12 X sites: the Pnma / I4/mcm cell (sqrt2 a, sqrt2 a, 2a). R and
    # the one M point along z are commensurate -- a-a-c+ about z, nothing else.
    "sqrt2xsqrt2x2": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
    # 10 atoms, 6 X sites: rock-salt (R-point) ordering only; for quick tests.
    "fcc2": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
}

# Zone-boundary q-points of the cubic cell that carry octahedral tilts.
TILT_QPOINTS = {
    "R": (0.5, 0.5, 0.5),
    "M_z": (0.5, 0.5, 0.0), "M_y": (0.5, 0.0, 0.5), "M_x": (0.0, 0.5, 0.5),
}


def _as_fraction_map(mixture: dict[str, float]) -> dict[str, float]:
    total = sum(mixture.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"site fractions {mixture} sum to {total}, not 1")
    if any(v <= 0 for v in mixture.values()):
        raise ValueError(f"site fractions must be positive: {mixture}")
    return dict(sorted(mixture.items()))


@dataclass
class ParentLattice:
    """An ordered parent structure and which of its species become mixtures."""

    primitive: ase.Atoms
    mixtures: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        present = set(self.primitive.get_chemical_symbols())
        for parent_symbol in self.mixtures:
            if parent_symbol not in present:
                raise ValueError(f"mixing rule for {parent_symbol!r}, which isn't in the parent ({sorted(present)})")
        self.mixtures = {s: _as_fraction_map(m) for s, m in self.mixtures.items()}
        species_sets = [tuple(m) for m in self.mixtures.values()]
        if len(set(species_sets)) != len(species_sets):
            # icet defines a sublattice by its set of allowed species, so two
            # mixtures over the same species would be merged into one
            raise ValueError(f"two mixed sublattices allow the same species: {species_sets}")

    @property
    def chemical_symbols(self) -> list[list[str]]:
        """Allowed species per parent atom, in icet's ClusterSpace format."""
        return [list(self.mixtures[s]) if s in self.mixtures else [s] for s in self.primitive.get_chemical_symbols()]

    @property
    def mixed_species(self) -> list[str]:
        return list(self.mixtures)

    @property
    def pseudocubic_a(self) -> float:
        """Cube root of the volume per 5-atom formula unit (A)."""
        return (self.primitive.get_volume() * 5 / len(self.primitive)) ** (1 / 3)

    def supercell(self, matrix) -> ase.Atoms:
        """The parent tiled by `matrix` (a SUPERCELL_PRESETS key or a 3x3
        integer matrix), still with its ordered parent species."""
        P = SUPERCELL_PRESETS[matrix] if isinstance(matrix, str) else np.asarray(matrix)
        return make_supercell(self.primitive, P)

    def site_counts(self, matrix) -> dict[str, int]:
        """Number of sites on each mixed sublattice in that supercell."""
        symbols = self.supercell(matrix).get_chemical_symbols()
        return {s: symbols.count(s) for s in self.mixtures}

    def species_counts(self, matrix) -> dict[str, dict[str, int]]:
        """Exact species counts per mixed sublattice in that supercell; raises
        if a target fraction isn't a whole number of sites."""
        counts = {}
        for parent_symbol, n_sites in self.site_counts(matrix).items():
            mixture = self.mixtures[parent_symbol]
            raw = {s: c * n_sites for s, c in mixture.items()}
            rounded = {s: round(v) for s, v in raw.items()}
            if any(abs(raw[s] - rounded[s]) > 1e-6 for s in raw) or sum(rounded.values()) != n_sites:
                step = Fraction(1, n_sites)
                raise ValueError(
                    f"{'/'.join(mixture)} sublattice: fractions {mixture} need "
                    f"{ {s: round(v, 3) for s, v in raw.items()} } of {n_sites} sites; "
                    f"this supercell only reaches multiples of {step}")
            counts[parent_symbol] = rounded
        return counts

    def target_concentrations(self) -> dict[str, dict[str, float]]:
        """{parent species: {species: fraction}} -- converted to icet's
        sublattice letters in disorder.icet_target_concentrations."""
        return self.mixtures


def commensurate_tilt_points(matrix) -> dict[str, bool]:
    """Which cubic tilt q-points a supercell of the cubic cell admits: q is
    commensurate iff q . P_i is an integer for every supercell vector P_i."""
    P = SUPERCELL_PRESETS[matrix] if isinstance(matrix, str) else np.asarray(matrix)
    out = {}
    for name, q in TILT_QPOINTS.items():
        phase = np.mod(P @ np.asarray(q), 1.0)
        out[name] = bool(np.all(np.minimum(phase, 1 - phase) < 1e-8))
    return out


def vegard_lattice_parameter(a: dict[str, float], b: dict[str, float], x: dict[str, float]) -> float:
    """Cubic lattice parameter 2 (r_B + r_X) with composition-averaged radii
    (pyrovskite's B-X = r_B + r_X convention, as perovskite_builder uses). A
    starting guess only: the MLIP relaxes the cell."""
    del a  # the A site doesn't enter the corner-sharing B-X-B length
    r_b = sum(f * lookup_ion("B", s).ionic_radius_angstrom for s, f in b.items())
    r_x = sum(f * lookup_ion("X", s).ionic_radius_angstrom for s, f in x.items())
    return 2 * (r_b + r_x)


def formability(a: dict[str, float] | str, b: dict[str, float] | str, x: dict[str, float] | str) -> dict[str, float]:
    """Goldschmidt tolerance factor t = (r_A + r_X) / (sqrt2 (r_B + r_X)) and
    octahedral factor mu = r_B / r_X from composition-averaged radii -- the
    same radii as vegard_lattice_parameter. Averages hide size mismatch (an
    F/I mixture can average into the window when neither halide alone fits
    with the other), so screen a mixture on its end members, not on this."""
    sites = [{s: 1.0} if isinstance(s, str) else s for s in (a, b, x)]
    r = [sum(f * lookup_ion(label, el).ionic_radius_angstrom for el, f in m.items())
         for label, m in zip(SITE_LABELS, sites)]
    return {"t": tolerance_factor(*r), "mu": octahedral_factor(r[1], r[2])}


def cubic_parent(a: dict[str, float] | str, b: dict[str, float] | str, x: dict[str, float] | str,
                 lattice_parameter: float | None = None) -> ParentLattice:
    """The 5-atom cubic Pm-3m aristotype with any site mixed, e.g.
    cubic_parent("Cs", "Pb", {"Br": 2/3, "I": 1/3}).

    Each site is written in the parent as its first species (alphabetically),
    and the mixing rule maps that placeholder to the full mixture. The lattice
    parameter defaults to vegard_lattice_parameter.
    """
    sites = {label: ({v: 1.0} if isinstance(v, str) else _as_fraction_map(v)) for label, v in zip(SITE_LABELS, (a, b, x))}
    if lattice_parameter is None:
        lattice_parameter = vegard_lattice_parameter(*sites.values())
    placeholder = {label: next(iter(m)) for label, m in sites.items()}
    primitive = ase.Atoms(
        [placeholder["A"], placeholder["B"]] + [placeholder["X"]] * 3,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        cell=[lattice_parameter] * 3, pbc=True,
    )
    mixtures = {placeholder[label]: m for label, m in sites.items() if len(m) > 1}
    return ParentLattice(primitive, mixtures)


def conventional_cell(atoms: ase.Atoms, symprec: float = 1e-2) -> ase.Atoms:
    """Conventional standard cell (pymatgen/spglib) -- orthogonal axes for the
    common perovskite space groups, so a diagonal supercell of it stays
    reasonably isotropic. MP CIFs are symmetric only to ~1e-3 (see
    scripts/qe/generate_mp_inputs.py), hence the loose default symprec. The cell
    as given if spglib finds nothing at that tolerance."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    structure = AseAtomsAdaptor.get_structure(atoms)
    try:
        conventional = SpacegroupAnalyzer(structure, symprec=symprec).get_conventional_standard_structure()
    except Exception:
        return atoms.copy()
    return AseAtomsAdaptor.get_atoms(conventional)


def scaled_to_volume(atoms: ase.Atoms, volume_per_atom: float) -> ase.Atoms:
    """Isotropic rescale of the cell (fractional positions fixed) to a target
    volume per atom."""
    out = atoms.copy()
    factor = (volume_per_atom * len(out) / out.get_volume()) ** (1 / 3)
    out.set_cell(out.cell[:] * factor, scale_atoms=True)
    return out


def supercell_for_sites(parent: ParentLattice, min_sites: int = 24, multiple_of: int = 6,
                        max_repeat: int = 4) -> tuple[int, int, int]:
    """Smallest diagonal supercell of parent.primitive whose every mixed
    sublattice has at least min_sites sites, a multiple of multiple_of (24 and 6
    keep the 1/6 composition grid reachable, as in the cubic 2x2x2 cell). Ties
    in atom count go to the most isotropic cell (perpendicular widths closest to
    equal)."""
    counts = {s: parent.primitive.get_chemical_symbols().count(s) for s in parent.mixtures}
    widths = 1.0 / np.linalg.norm(parent.primitive.cell.reciprocal(), axis=1)
    best = None
    for reps in itertools.product(range(1, max_repeat + 1), repeat=3):
        n = math.prod(reps)
        if any(c * n < min_sites or (c * n) % multiple_of for c in counts.values()):
            continue
        w = widths * np.array(reps)
        key = (n, w.max() / w.min())
        if best is None or key < best[0]:
            best = (key, reps)
    if best is None:
        raise ValueError(f"no diagonal supercell up to {max_repeat}^3 gives >= {min_sites} sites "
                         f"(multiple of {multiple_of}) on every mixed sublattice: {counts}")
    return best[1]


def binary_fractions(n_sites: int, denominators: tuple[int, ...] = (2, 3, 4, 6)) -> list[float]:
    """Interior fractions with the given denominators that n_sites can
    represent exactly -- e.g. 24 X sites: 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6."""
    fractions = {Fraction(k, d) for d in denominators for k in range(1, d)}
    return [float(f) for f in sorted(fractions) if (f * n_sites).denominator == 1]


def halide_pairs(halides=("Cl", "Br", "I")) -> list[tuple[str, str]]:
    return list(itertools.combinations(halides, 2))


def expected_formula(parent: ParentLattice, supercell) -> str:
    """Reduced formula of a decoration of parent in supercell (a preset name
    or a diagonal (nx, ny, nz)), before generating it."""
    from collections import Counter

    from pymatgen.core import Composition

    matrix = supercell if isinstance(supercell, str) else np.diag(supercell)
    counts = Counter(parent.supercell(matrix).get_chemical_symbols())
    for parent_symbol, species in parent.species_counts(matrix).items():
        del counts[parent_symbol]
        counts.update(species)
    return Composition(dict(counts)).reduced_formula
