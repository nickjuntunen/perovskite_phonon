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


def expected_unpaired_electrons(symbol: str, oxidation_state: int) -> int | None:
    """Predicted number of unpaired electrons for an ion, via Hund's-rule filling
    of the mendeleev ground-state configuration ionized to `oxidation_state`.

    Returns None if mendeleev has no data for the element, or if
    oxidation_state <= 0 (mendeleev's ionize() only removes electrons, so
    anions -- always closed-shell in this pipeline's X-site halides/
    chalcogenides -- aren't handled here; they don't need to be).
    """
    if oxidation_state <= 0:
        return 0
    try:
        ion_ec = mendeleev_element(symbol).ec.ionize(oxidation_state)
    except (ValueError, Exception):
        return None
    return ion_ec.unpaired_electrons()


def expected_spin_only_moment_bohr(symbol: str, oxidation_state: int) -> float | None:
    """Spin-only magnetic moment (mu_B) predicted for an ion, sqrt(n(n+2))
    from its Hund's-rule unpaired-electron count. None if undetermined."""
    unpaired = expected_unpaired_electrons(symbol, oxidation_state)
    if unpaired is None:
        return None
    return (unpaired * (unpaired + 2)) ** 0.5


def is_open_shell(symbol: str, oxidation_state: int) -> bool:
    """True if this ion is predicted to carry unpaired electrons and therefore
    needs nspin=2 (+ a starting_magnetization guess) rather than a
    non-spin-polarized SCF."""
    unpaired = expected_unpaired_electrons(symbol, oxidation_state)
    return bool(unpaired)
