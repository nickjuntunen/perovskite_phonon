import re
from dataclasses import dataclass

from mendeleev import element as mendeleev_element

_ROMAN_TO_CN = {
    'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
    'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12,
}
_CN_PREFIX_RE = re.compile(r'^[IVX]+')


def shannon_ionic_radius(symbol: str, oxidation_state: int, coordination: int) -> float | None:
    """Look up a Shannon effective ionic radius (angstrom) from mendeleev.

    Matches on element and formal charge, preferring the exact requested
    coordination number and falling back to whichever coordination mendeleev
    reports that's closest to it. Halides are weak-field ligands, so when a
    transition-metal ion has both high- and low-spin radii at the same
    coordination, the high-spin value is preferred (appropriate for halide
    perovskites; revisit if used for oxide perovskites). Returns None if
    mendeleev has no entry for that element/charge combination at all (e.g.
    it has no Sn2+ entry despite Sn2+ being the common perovskite B-site ion).
    """
    try:
        candidates = [r for r in mendeleev_element(symbol).ionic_radii if r.charge == oxidation_state]
    except Exception:
        return None
    if not candidates:
        return None

    def _cn(radius):
        match = _CN_PREFIX_RE.match(radius.coordination)
        return _ROMAN_TO_CN.get(match.group()) if match else None

    annotated = [(_cn(r), r) for r in candidates]
    annotated = [(cn, r) for cn, r in annotated if cn is not None]
    if not annotated:
        return None

    exact = [r for cn, r in annotated if cn == coordination]
    if exact:
        pool = exact
    else:
        nearest_cn = min((cn for cn, _ in annotated), key=lambda cn: abs(cn - coordination))
        pool = [r for cn, r in annotated if cn == nearest_cn]

    high_spin = [r for r in pool if r.spin == 'HS']
    chosen = high_spin[0] if high_spin else pool[0]
    return chosen.ionic_radius / 100


@dataclass
class Ion:
    """Chemical species occupying a crystallographic site."""
    symbol: str
    oxidation_state: int
    ionic_radius_angstrom: float | None = None
    coordination: int | None = None

    @property
    def label(self) -> str:
        sign = "+" if self.oxidation_state > 0 else "-"
        return f"{self.symbol}{abs(self.oxidation_state)}{sign}"
