"""Finite-cell representations of a substitutionally disordered perovskite
(parent.ParentLattice), each a weighted ensemble of ordered supercells whose
weighted average stands in for the disordered crystal.

A phonon property P of a random alloy is the configurational average <P> over
all decorations of the infinite lattice. A finite supercell can only
approximate that average, and the strategies here trade cost for rigor
differently:

    sqs          special quasirandom structures (icet, simulated annealing):
                 single decorations whose short-range pair and triplet
                 correlations match the infinite random alloy's as closely as
                 the cell allows. Several independent anneals (different
                 seeds) give distinct SQSs; their spread is a lower bound on
                 the disorder uncertainty of <P>, and a check that one SQS
                 isn't an outlier.
    random       plain random decorations at the exact composition: an
                 unbiased Monte Carlo estimate of <P> over this cell, but
                 individual cells deviate from random correlations far more
                 than an SQS does, so it needs many more samples.
    enumerated   every symmetry-distinct decoration of the cell, weighted by
                 its multiplicity (how many of the cell's decorations it
                 stands for). The weighted average is the *exact* ideal-mixing
                 (infinite-temperature) average for that cell -- no sampling
                 error, only finite-cell error -- and the class with the
                 smallest correlation error is the cell's exact best SQS.
                 Feasible only while the number of decorations stays small
                 (MAX_DECORATIONS). "Exact for the cell" still isn't the
                 infinite alloy: fixing exactly k of n sites anticorrelates
                 them, so the average pair correlation is ((n-2k)^2 - n) /
                 (n(n-1)) rather than (1-2x)^2 -- 0.030 vs 0.111 for 4 of 12
                 sites, 0.072 vs 0.111 for 8 of 24 -- an O(1/n) bias that SQS,
                 which targets the infinite alloy, doesn't have. Comparing the
                 two is a direct measure of the finite-cell error.

Every realization reports correlation_error: the RMS deviation of its
pair/triplet cluster correlations from the infinite random alloy's, in icet's
cluster basis. Points (order-1 correlations) are fixed by the composition and
left out; that leaves short-range order -- what a finite cell gets wrong.

Equivalence. Two decorations of one supercell are the same structure if a
symmetry operation of the undecorated supercell (point operations and
translations) maps one to the other. canonical_codes reduces each decoration
to the minimum over that group of its integer encoding -- an exact test, used
both to enumerate classes and to drop duplicate SQS/random cells.

Cutoffs. In a small supercell, a pair whose length exceeds about half the cell
width sees its own periodic image, and its correlation stops describing a
short-range pair. default_cutoffs keeps pairs inside that limit (and triplets
to the nearest-neighbor shell), and icet's is_supercell_self_interacting is
checked on every cell.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import ase
import ase.data
import icet
import numpy as np
import spglib
from icet.tools.structure_generation import generate_sqs_from_supercells, occupy_structure_randomly

from src.structure_gen.parent import ParentLattice

# Upper limit on decorations enumerate_orderings will canonicalize: 2x2x2 at
# x = 1/3 (C(24, 8) = 735,471) takes about a minute against its 384 symmetry operations.
MAX_DECORATIONS = 2_000_000
_CHUNK = 50_000


@dataclass
class Realization:
    atoms: ase.Atoms
    strategy: str  # "sqs" | "random" | "enumerated"
    index: int
    weight: float  # within its ensemble; the weights of an ensemble sum to 1
    correlation_error: float
    supercell: str
    seed: int | None = None
    multiplicity: int = 1  # enumerated: decorations of this cell this class stands for
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- cluster space


def _label(supercell) -> str:
    if isinstance(supercell, str):
        return supercell
    P = np.asarray(supercell)
    if np.count_nonzero(P - np.diag(np.diag(P))) == 0:
        return "x".join(str(int(v)) for v in np.diag(P))
    return "P" + "".join(str(int(v)).replace("-", "m") for v in P.ravel())


def default_cutoffs(parent: ParentLattice, supercell) -> list[float]:
    """[pair, triplet] cutoffs (A): pairs out to the second X-X shell (a_pc,
    covering a/sqrt2 and a) or half the smallest supercell width, whichever is
    shorter -- but never short of the first shell (0.75 a_pc > a/sqrt2): in a
    cell too small for even that, nearest-neighbour pairs see their own images
    (flagged as self-interacting), which is still better than characterizing
    the cell by no pairs at all. Triplets within the nearest-neighbour shell."""
    cell = parent.supercell(supercell).cell
    # perpendicular widths: volume / area of the opposite face
    widths = [cell.volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    a = parent.pseudocubic_a
    pair = max(0.75 * a, min(1.05 * a, 0.5 * min(widths) - 1e-3))
    return [pair, 0.75 * a]


def cluster_space(parent: ParentLattice, supercell, cutoffs: list[float] | None = None) -> icet.ClusterSpace:
    if not parent.mixtures:
        raise ValueError("parent has no mixed sublattice")
    cutoffs = cutoffs if cutoffs is not None else default_cutoffs(parent, supercell)
    return icet.ClusterSpace(parent.primitive, cutoffs, parent.chemical_symbols)


def icet_target_concentrations(cs: icet.ClusterSpace, parent: ParentLattice) -> dict[str, dict[str, float]]:
    """parent.mixtures re-keyed by icet's sublattice letters. icet letters the
    *mixed* sublattices first ("A" is often the halide site), so the mapping
    goes through the allowed-species sets, never the letters."""
    by_species = {tuple(m): m for m in parent.mixtures.values()}
    out = {}
    for sublattice in cs.get_sublattices(cs.primitive_structure):
        species = tuple(sorted(sublattice.chemical_symbols))
        if len(species) > 1:
            out[sublattice.symbol] = by_species[species]
    return out


def random_alloy_correlations(cs: icet.ClusterSpace, parent: ParentLattice) -> np.ndarray:
    """Cluster vector of the infinite random alloy at the target composition."""
    # icet's own SQS target; private in icet 4.0, but it's the definition its
    # SQS generators optimize against, so using it keeps the two consistent
    from icet.tools.structure_generation import _get_sqs_cluster_vector
    return _get_sqs_cluster_vector(cs, icet_target_concentrations(cs, parent))


def correlation_error(cs: icet.ClusterSpace, atoms: ase.Atoms, target_cv: np.ndarray) -> float:
    orders = np.array([c["order"] for c in cs.as_list])
    mask = orders >= 2
    diff = cs.get_cluster_vector(atoms)[mask] - target_cv[mask]
    return float(np.sqrt(np.mean(diff**2)))


# ---------------------------------------------------------------- symmetry


def _mixed_sites(parent: ParentLattice, supercell_atoms: ase.Atoms):
    """Indices of mixed sites in the undecorated supercell (grouped by
    sublattice, in parent.mixed_species order), the allowed species of each
    sublattice, and of each mixed site.

    A decoration is stored as one label per mixed site: the index of its
    species in that site's own allowed list. Symmetry operations never move a
    site to another sublattice, so these local labels are enough to tell
    decorations apart, and the code base is the largest sublattice's species
    count rather than the total -- Cs/Rb + Br/I is 1 bit per site, not 2."""
    symbols = np.array(supercell_atoms.get_chemical_symbols())
    order = parent.mixed_species
    per_sublattice = [np.where(symbols == s)[0] for s in order]
    allowed = [list(parent.mixtures[s]) for s in order]
    site_allowed = [a for a, sites in zip(allowed, per_sublattice) for _ in sites]
    return np.concatenate(per_sublattice), order, allowed, site_allowed


def site_permutations(supercell_atoms: ase.Atoms, symprec: float = 1e-4) -> np.ndarray:
    """(n_ops, n_atoms) atom permutations for every symmetry operation of the
    undecorated supercell (spglib), translations included."""
    numbers = ase.data.atomic_numbers
    cell = (supercell_atoms.cell[:], supercell_atoms.get_scaled_positions(),
            [numbers[s] for s in supercell_atoms.get_chemical_symbols()])
    sym = spglib.get_symmetry(cell, symprec=symprec)
    pos = supercell_atoms.get_scaled_positions()
    perms = []
    for R, t in zip(sym["rotations"], sym["translations"]):
        new = pos @ R.T + t
        diff = new[:, None, :] - pos[None, :, :]
        diff -= np.round(diff)
        perm = np.argmin(np.linalg.norm(diff, axis=2), axis=1)
        if len(set(perm)) != len(perm):
            raise RuntimeError("symmetry operation did not map the supercell onto itself")
        perms.append(perm)
    return np.array(perms)


def _mixed_permutations(perms: np.ndarray, mixed_idx: np.ndarray) -> np.ndarray:
    """Permutations restricted to the mixed sites, as positions within mixed_idx."""
    where = -np.ones(perms.shape[1], dtype=int)
    where[mixed_idx] = np.arange(len(mixed_idx))
    out = where[perms[:, mixed_idx]]
    if (out < 0).any():
        raise RuntimeError("a symmetry operation maps a mixed site onto an ordered one")
    return out


def canonical_codes(labels: np.ndarray, mixed_perms: np.ndarray, n_species: int) -> np.ndarray:
    """Minimum over the group of each decoration's base-n_species integer code
    (labels: (n_conf, n_mixed) local species labels, see _mixed_sites). Equal
    codes <=> equivalent."""
    n = labels.shape[1]
    if n * math.log2(max(n_species, 2)) >= 63:
        raise ValueError(f"{n} mixed sites x {n_species} species overflow a 64-bit code; use a smaller cell")
    powers = n_species ** np.arange(n, dtype=np.int64)[::-1]
    out = np.empty(len(labels), dtype=np.int64)
    for start in range(0, len(labels), _CHUNK):
        block = labels[start:start + _CHUNK].astype(np.int64)
        best = block @ powers
        for p in mixed_perms:
            np.minimum(best, block[:, p] @ powers, out=best)
        out[start:start + _CHUNK] = best
    return out


def _decode(code: int, n: int, n_species: int) -> np.ndarray:
    digits = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        code, digits[i] = divmod(code, n_species)
    return digits


# ---------------------------------------------------------------- decorations


def _decorate(supercell_atoms: ase.Atoms, mixed_idx: np.ndarray, labels: np.ndarray, site_allowed) -> ase.Atoms:
    atoms = supercell_atoms.copy()
    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    symbols[mixed_idx] = [allowed[k] for allowed, k in zip(site_allowed, labels)]
    atoms.set_chemical_symbols(list(symbols))
    return atoms


def _labels_of(atoms: ase.Atoms, mixed_idx: np.ndarray, site_allowed) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    return np.array([allowed.index(symbols[i]) for i, allowed in zip(mixed_idx, site_allowed)])


def _sublattice_labelings(n_sites: int, counts: list[int], species_ids: list[int]) -> np.ndarray:
    """All arrangements of `counts` of each species over n_sites (one row each)."""
    if len(counts) == 1:
        return np.full((1, n_sites), species_ids[0], dtype=np.int8)
    rows = []
    for chosen in itertools.combinations(range(n_sites), counts[0]):
        rest = [i for i in range(n_sites) if i not in chosen]
        for sub in _sublattice_labelings(len(rest), counts[1:], species_ids[1:]):
            row = np.empty(n_sites, dtype=np.int8)
            row[list(chosen)] = species_ids[0]
            row[rest] = sub
            rows.append(row)
    return np.array(rows)


def _binary_labelings(n_sites: int, k: int, id0: int, id1: int) -> np.ndarray:
    """Fast path for a binary sublattice: every choice of k sites for id1."""
    combos = np.array(list(itertools.combinations(range(n_sites), k)), dtype=np.int64).reshape(-1, k)
    out = np.full((len(combos), n_sites), id0, dtype=np.int8)
    np.put_along_axis(out, combos, id1, axis=1)
    return out


def n_decorations(parent: ParentLattice, supercell) -> int:
    total = 1
    for counts in parent.species_counts(supercell).values():
        n = sum(counts.values())
        total *= math.factorial(n) // math.prod(math.factorial(c) for c in counts.values())
    return total


# ---------------------------------------------------------------- strategies


def sqs(parent: ParentLattice, supercell="2x2x2", n: int = 3, cutoffs=None, n_steps: int | None = None,
        seed: int = 0, dedupe: bool = True) -> list[Realization]:
    """n independent SQS anneals (seeds seed, seed+1, ...) in a fixed supercell,
    equally weighted; equivalent results are merged and their weight pooled."""
    parent.species_counts(supercell)  # fail early on an unreachable composition
    cs = cluster_space(parent, supercell, cutoffs)
    target = icet_target_concentrations(cs, parent)
    target_cv = random_alloy_correlations(cs, parent)
    base = parent.supercell(supercell)
    if cs.is_supercell_self_interacting(base):
        warnings.warn(f"cutoffs {cs.cutoffs} are self-interacting in the {_label(supercell)} cell")

    cells = [generate_sqs_from_supercells(cs, [base], target, n_steps=n_steps, random_seed=seed + i) for i in range(n)]
    return _ensemble(cells, "sqs", parent, supercell, cs, target_cv, [seed + i for i in range(n)], dedupe)


def random_configurations(parent: ParentLattice, supercell="2x2x2", n: int = 10, cutoffs=None, seed: int = 0,
                          dedupe: bool = True) -> list[Realization]:
    """n random decorations at the exact composition, equally weighted."""
    parent.species_counts(supercell)
    cs = cluster_space(parent, supercell, cutoffs)
    target = icet_target_concentrations(cs, parent)
    target_cv = random_alloy_correlations(cs, parent)
    base = parent.supercell(supercell)
    cells = []
    for i in range(n):
        atoms = base.copy()
        occupy_structure_randomly(atoms, cs, target, random_seed=seed + i)
        cells.append(atoms)
    return _ensemble(cells, "random", parent, supercell, cs, target_cv, [seed + i for i in range(n)], dedupe)


def _ensemble(cells, strategy, parent, supercell, cs, target_cv, seeds, dedupe) -> list[Realization]:
    base = parent.supercell(supercell)
    mixed_idx, _, allowed, site_allowed = _mixed_sites(parent, base)
    weights = np.full(len(cells), 1 / len(cells))
    keep = list(range(len(cells)))
    if dedupe and len(cells) > 1:
        perms = _mixed_permutations(site_permutations(base), mixed_idx)
        labels = np.array([_labels_of(a, mixed_idx, site_allowed) for a in cells])
        codes = canonical_codes(labels, perms, max(len(a) for a in allowed))
        uniq, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
        keep = sorted(first)
        pooled = np.bincount(inverse, minlength=len(uniq)) / len(cells)
        weights = np.zeros(len(cells))
        weights[first] = pooled
        if len(keep) < len(cells):
            print(f"{strategy}: {len(cells) - len(keep)} of {len(cells)} cells were symmetry-equivalent duplicates (weight pooled)")
    return [
        Realization(atoms=cells[i], strategy=strategy, index=n, weight=float(weights[i]),
                    correlation_error=correlation_error(cs, cells[i], target_cv), supercell=_label(supercell),
                    seed=seeds[i], info={"cutoffs": list(cs.cutoffs)})
        for n, i in enumerate(keep)
    ]


def enumerate_orderings(parent: ParentLattice, supercell="sqrt2xsqrt2x2", cutoffs=None,
                        max_decorations: int = MAX_DECORATIONS) -> list[Realization]:
    """Every symmetry-distinct decoration of one supercell, weighted by
    multiplicity / total, sorted by correlation_error (best SQS first)."""
    counts = parent.species_counts(supercell)
    total = n_decorations(parent, supercell)
    if total > max_decorations:
        raise ValueError(f"{total:,} decorations in the {_label(supercell)} cell (> {max_decorations:,}); "
                         "use a smaller cell, or sqs()/random_configurations()")
    base = parent.supercell(supercell)
    mixed_idx, order, allowed, site_allowed = _mixed_sites(parent, base)
    base_n = max(len(a) for a in allowed)

    # Cartesian product of each sublattice's arrangements
    blocks = []
    for parent_symbol, allowed_species in zip(order, allowed):
        c = counts[parent_symbol]
        n_sites = sum(c.values())
        ids = list(range(len(allowed_species)))
        if len(allowed_species) == 2:
            blocks.append(_binary_labelings(n_sites, c[allowed_species[1]], ids[0], ids[1]))
        else:
            blocks.append(_sublattice_labelings(n_sites, [c[s] for s in allowed_species], ids))
    labels = blocks[0]
    for block in blocks[1:]:
        labels = np.hstack([np.repeat(labels, len(block), axis=0), np.tile(block, (len(labels), 1))])

    perms = _mixed_permutations(site_permutations(base), mixed_idx)
    codes, multiplicity = np.unique(canonical_codes(labels, perms, base_n), return_counts=True)

    cs = cluster_space(parent, supercell, cutoffs)
    target_cv = random_alloy_correlations(cs, parent)
    if cs.is_supercell_self_interacting(base):
        warnings.warn(f"cutoffs {cs.cutoffs} are self-interacting in the {_label(supercell)} cell")
    out = []
    for code, m in zip(codes, multiplicity):
        atoms = _decorate(base, mixed_idx, _decode(int(code), len(mixed_idx), base_n), site_allowed)
        out.append(Realization(atoms=atoms, strategy="enumerated", index=-1, weight=float(m / total),
                               correlation_error=correlation_error(cs, atoms, target_cv), supercell=_label(supercell),
                               multiplicity=int(m), info={"cutoffs": list(cs.cutoffs)}))
    out.sort(key=lambda r: (r.correlation_error, -r.multiplicity))
    for i, r in enumerate(out):
        r.index = i
    return out


# ---------------------------------------------------------------- averaging


def ensemble_average(values, weights) -> dict[str, float]:
    """Weighted mean of a property over an ensemble, its weighted standard
    deviation (the disorder spread between realizations), and the standard
    error of the mean from Kish's effective sample size. For an enumerated
    ensemble the mean is exact for the cell and the 'sem' is not a sampling
    error; it's reported for comparison only."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = float(np.sum(w * v))
    std = float(np.sqrt(np.sum(w * (v - mean) ** 2)))
    n_eff = float(1 / np.sum(w**2))
    return {"mean": mean, "std": std, "n_eff": n_eff, "sem": std / math.sqrt(n_eff)}
