"""Verify that a structure actually has the perovskite (corner-sharing BX6
octahedral framework) topology, rather than merely matching an ABX3 / A2BB'X6
stoichiometry.

Why this exists: mp_perovskites.py's anonymized-formula search (formula="ABC3"
etc.) matches any compound with that reduced composition, with no regard for
structure. Plenty of non-perovskite compounds -- intermetallics, silicides,
pyroxenes, garnets -- happen to reduce to a 1:1:3 or 1:1:2:6 ratio (e.g.
YNiSi3, CaMg(SiO3)2) and get pulled in purely on stoichiometry. robocrys
keyword hits are much more reliable since they come from real bonding
analysis, but aren't collected for every candidate and can still miss/mislabel
edge cases, so this gives every candidate its own independent structural
check instead of relying on text-mined descriptions.

The perovskite framework is defined by two structural facts, checked
directly rather than inferred from composition or text:
  1. The B-site cation(s) sit in ~6-fold, near-ideal octahedral coordination
     to the bridging anion (X).
  2. Those BX6 octahedra share corners (a single bridging X per B-B pair),
     not edges or faces, and do so in all 3 dimensions -- an isolated
     octahedron, a chain, or a 2D sheet is not a perovskite.
The A-site's large (~8-12) coordination number is checked as a supporting
signal, since some edge/face-sharing sublattices (ilmenite, corundum) also
have 6-coordinate B cations but no such A site.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure

# B-site coordination number must fall in this range to count as "octahedral".
B_SITE_CN_RANGE = (5.5, 6.5)
# A-site coordination number must be at least this to count as a perovskite
# A-site (cuboctahedral in the ideal cubic case, lower under tilting).
A_SITE_MIN_CN = 8.0
# Fraction of B-B bridges (via a shared X) that must be single-atom
# (corner-sharing) rather than edge/face-sharing.
MIN_CORNER_SHARING_FRACTION = 0.8
# Average number of distinct corner-sharing B neighbors per B atom; ~6 in the
# ideal cubic case, lower under tilting/vacancy ordering, but a 1D chain or
# 2D sheet (rather than a 3D framework) will fall well short of this.
MIN_B_NETWORK_DEGREE = 4.0
# Supercell size used for the connectivity graph, so periodic self-images of
# a B atom aren't collapsed onto the same index (which would silently hide
# their shared corners). Must be >=3 per axis: with only 2 repeats, a site's
# +axis and -axis neighbor both wrap onto the same image, undercounting
# network degree by up to half (e.g. simple ABX3 perovskites with a
# 1-formula-unit primitive cell).
SUPERCELL = (3, 3, 3)


@dataclass
class PerovskiteCheckResult:
    is_perovskite: bool
    x_site: str | None
    b_sites: list[str] = field(default_factory=list)
    a_sites: list[str] = field(default_factory=list)
    b_site_cn: dict[str, float] = field(default_factory=dict)
    a_site_cn: dict[str, float] = field(default_factory=dict)
    corner_sharing_fraction: float | None = None
    avg_b_network_degree: float | None = None
    reasons: list[str] = field(default_factory=list)


def _cation_cn_to_anion(structure: Structure, cnn: CrystalNN, cation_symbol: str, x_symbol: str) -> list[int]:
    cns = []
    for i, site in enumerate(structure):
        if site.specie.symbol != cation_symbol:
            continue
        nn_info = cnn.get_nn_info(structure, i)
        cns.append(sum(1 for n in nn_info if n["site"].specie.symbol == x_symbol))
    return cns


def _b_network_stats(structure: Structure, b_symbols: list[str], x_symbol: str) -> tuple[float, float]:
    """Corner-sharing fraction and average B-network degree, computed on a
    supercell so periodic images of the same primitive B site are distinct
    graph nodes (see SUPERCELL docstring above)."""
    super_structure = structure * SUPERCELL
    cnn = CrystalNN()

    b_indices = [i for i, site in enumerate(super_structure) if site.specie.symbol in b_symbols]
    x_to_bs: dict[tuple, set[int]] = defaultdict(set)
    for bi in b_indices:
        nn_info = cnn.get_nn_info(super_structure, bi)
        for n in nn_info:
            if n["site"].specie.symbol != x_symbol:
                continue
            # Wrap into [0, 1) before rounding: get_nn_info returns each
            # neighbor's *unwrapped* frac_coords (offset by its periodic
            # image), so the same physical anion reached from two different
            # B atoms can otherwise land on two different keys (e.g. -0.25
            # vs 0.75) and never get recognized as a shared bridge.
            x_key = tuple(np.round(np.mod(n["site"].frac_coords, 1.0), 4))
            x_to_bs[x_key].add(bi)

    b_neighbor_sets: dict[int, set[int]] = defaultdict(set)
    corner_pairs = 0
    non_corner_pairs = 0
    for bs in x_to_bs.values():
        if len(bs) < 2:
            continue
        bs = sorted(bs)
        if len(bs) == 2:
            corner_pairs += 1
            b_neighbor_sets[bs[0]].add(bs[1])
            b_neighbor_sets[bs[1]].add(bs[0])
        else:
            # An X shared by 3+ B atoms means at least one edge/face-sharing
            # bridge; still register the network edges so degree reflects it.
            non_corner_pairs += len(bs) - 1
            for a in bs:
                for b in bs:
                    if a != b:
                        b_neighbor_sets[a].add(b)

    total_bridges = corner_pairs + non_corner_pairs
    corner_fraction = corner_pairs / total_bridges if total_bridges else 0.0
    degree = np.mean([len(v) for v in b_neighbor_sets.values()]) if b_neighbor_sets else 0.0
    return corner_fraction, degree


def check_perovskite_structure(structure: Structure) -> PerovskiteCheckResult:
    """Independently verify perovskite (corner-sharing BX6 octahedral
    framework) topology from the structure itself.

    Handles both simple ABX3 and A2BB'X6 double perovskites: any number of
    cation species can be classified as B-site, as long as each is
    ~6-coordinate/octahedral to X and, together, they form a corner-sharing
    3D network.
    """
    cnn = CrystalNN()
    counts = structure.composition.element_composition.get_el_amt_dict()
    if len(counts) < 3:
        return PerovskiteCheckResult(
            is_perovskite=False, x_site=None, reasons=["fewer than 3 distinct elements"]
        )

    # X = bridging anion = the highest-count element (3x in ABX3, 6x in
    # A2BB'X6). Ties are broken arbitrarily; a real perovskite anion sublattice
    # is unambiguously the largest.
    x_symbol = max(counts, key=counts.get)
    cation_symbols = [el for el in counts if el != x_symbol]

    reasons: list[str] = []
    b_sites, a_sites = [], []
    b_site_cn, a_site_cn = {}, {}
    for symbol in cation_symbols:
        cns = _cation_cn_to_anion(structure, cnn, symbol, x_symbol)
        if not cns:
            reasons.append(f"{symbol}: no coordination to {x_symbol} found")
            continue
        mean_cn = float(np.mean(cns))
        if B_SITE_CN_RANGE[0] <= mean_cn <= B_SITE_CN_RANGE[1]:
            b_sites.append(symbol)
            b_site_cn[symbol] = mean_cn
        elif mean_cn >= A_SITE_MIN_CN:
            a_sites.append(symbol)
            a_site_cn[symbol] = mean_cn
        else:
            reasons.append(
                f"{symbol}: coordination to {x_symbol} is {mean_cn:.1f} "
                f"(neither B-site octahedral [{B_SITE_CN_RANGE[0]}-{B_SITE_CN_RANGE[1]}] "
                f"nor A-site [{A_SITE_MIN_CN}+])"
            )

    if not b_sites:
        reasons.append("no cation found in octahedral (~6-fold) coordination to the anion sublattice")
        return PerovskiteCheckResult(
            is_perovskite=False,
            x_site=x_symbol,
            a_sites=a_sites,
            a_site_cn=a_site_cn,
            reasons=reasons,
        )
    if not a_sites:
        # Informational only, not a gate: CrystalNN's short-range algorithm
        # reliably undercounts A-site CN under octahedral tilting (e.g.
        # GdFeO3-type EuFeO3 measures Eu-O CN=5, not the true ~8) since the
        # coordination shell isn't a clean single-distance cutoff there. The
        # corner-sharing BX6 network is the decisive, chemistry-agnostic
        # signature of the perovskite framework; what fills the resulting
        # A-cavity is secondary.
        reasons.append("no cation found in large (8+-fold) A-site coordination to the anion sublattice")

    corner_fraction, degree = _b_network_stats(structure, b_sites, x_symbol)
    if corner_fraction < MIN_CORNER_SHARING_FRACTION:
        reasons.append(
            f"only {corner_fraction:.0%} of B-B anion bridges are corner-sharing "
            f"(need >={MIN_CORNER_SHARING_FRACTION:.0%}); likely edge/face-sharing octahedra"
        )
    if degree < MIN_B_NETWORK_DEGREE:
        reasons.append(
            f"B-site network degree is {degree:.1f} (need >={MIN_B_NETWORK_DEGREE}); "
            "framework isn't a 3D corner-sharing network (isolated/chain/layered)"
        )

    is_perovskite = bool(corner_fraction >= MIN_CORNER_SHARING_FRACTION and degree >= MIN_B_NETWORK_DEGREE)

    return PerovskiteCheckResult(
        is_perovskite=is_perovskite,
        x_site=x_symbol,
        b_sites=b_sites,
        a_sites=a_sites,
        b_site_cn=b_site_cn,
        a_site_cn=a_site_cn,
        corner_sharing_fraction=corner_fraction,
        avg_b_network_degree=degree,
        reasons=reasons,
    )


VERIFICATION_FIELDNAMES = [
    "verified_perovskite",
    "b_sites",
    "a_sites",
    "x_site",
    "b_site_cn",
    "a_site_cn",
    "corner_sharing_fraction",
    "avg_b_network_degree",
    "verification_reasons",
]


def verification_fields(cif_path) -> dict:
    """The VERIFICATION_FIELDNAMES columns for one structure file: the
    check_perovskite_structure verdict flattened to strings ("a;b" site lists,
    "Sym=12.00;..." coordination numbers -- the format
    composition_features.parse_site_cn reads). A structure that can't be read
    or checked gets verified_perovskite = "ERROR" with the reason."""
    try:
        structure = Structure.from_file(str(cif_path))
        result = check_perovskite_structure(structure)
    except Exception as exc:  # noqa: BLE001 -- log and keep going, don't abort a 12.5k-row pass over one bad CIF
        return {
            "verified_perovskite": "ERROR",
            "b_sites": "",
            "a_sites": "",
            "x_site": "",
            "b_site_cn": "",
            "a_site_cn": "",
            "corner_sharing_fraction": "",
            "avg_b_network_degree": "",
            "verification_reasons": f"exception: {exc}",
        }
    return {
        "verified_perovskite": result.is_perovskite,
        "b_sites": ";".join(result.b_sites),
        "a_sites": ";".join(result.a_sites),
        "x_site": result.x_site or "",
        "b_site_cn": ";".join(f"{k}={v:.2f}" for k, v in result.b_site_cn.items()),
        "a_site_cn": ";".join(f"{k}={v:.2f}" for k, v in result.a_site_cn.items()),
        "corner_sharing_fraction": (
            f"{result.corner_sharing_fraction:.3f}" if result.corner_sharing_fraction is not None else ""
        ),
        "avg_b_network_degree": (
            f"{result.avg_b_network_degree:.2f}" if result.avg_b_network_degree is not None else ""
        ),
        "verification_reasons": "|".join(result.reasons),
    }
