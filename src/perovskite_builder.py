"""Build bulk ABX3 perovskite structures via pyrovskite.

Ion identity and oxidation state always come from the NOMAD ion database
(data/nomad/nomad_ions.json), which covers far more ions than pyrovskite's bundled
table. Ionic radii prefer pyrovskite's curated literature values
(pyrovskite/data.py) and fall back to a mendeleev Shannon radius, at the ion's
real NOMAD-derived oxidation state, when pyrovskite has no entry for it (e.g.
Sn2+, which mendeleev's Shannon-radius table doesn't include at all).

Only monatomic (elemental) A-site cations are supported for now — organic
A-site cations need a real 3D geometry (e.g. from NOMAD's ions_database .xyz
files), which isn't wired up yet.
"""

import json
from pathlib import Path

from pyrovskite.builder import make_bulk
from pyrovskite.data import ionic_radii as PYROVSKITE_RADII

from chemistry import Ion, shannon_ionic_radius

NOMAD_IONS_PATH = Path(__file__).resolve().parent.parent / "data" / "nomad" / "nomad_ions.json"

# Conventional perovskite coordination numbers, used for the mendeleev fallback.
SITE_COORDINATION = {"A": 12, "B": 6, "X": 6}

# Elemental A-site cations found in the NOMAD ion database; organic cations
# (MA, FA, ...) are excluded until A-site geometry loading is wired up.
MONATOMIC_A_SITE_SYMBOLS = {"Cs", "Rb", "K", "Na", "Li", "Tl"}

_NOMAD_IONS = json.loads(NOMAD_IONS_PATH.read_text())


def lookup_ion(site: str, symbol: str) -> Ion:
    """Resolve an ion's oxidation state and radius for a perovskite site.

    Oxidation state comes from the NOMAD ion database. Radius prefers
    pyrovskite's curated table; if pyrovskite has no entry for this element
    (regardless of the charge pyrovskite's own key implies, e.g. its "Sb3+"
    naming), a mendeleev Shannon radius is looked up at the ion's real,
    NOMAD-derived oxidation state instead.
    """
    nomad_entry = _NOMAD_IONS.get(site, {}).get(symbol)
    if nomad_entry is None:
        raise ValueError(f"{symbol!r} is not a known {site}-site ion in the NOMAD database")
    oxidation_state = nomad_entry["oxidation_state"]

    radius = PYROVSKITE_RADII.get(site, {}).get(symbol)
    if radius is None:
        radius = shannon_ionic_radius(symbol, oxidation_state, SITE_COORDINATION[site])

    return Ion(
        symbol=symbol,
        oxidation_state=oxidation_state,
        ionic_radius_angstrom=radius,
        coordination=SITE_COORDINATION[site],
    )


def available_ions(site: str) -> dict[str, Ion]:
    """All ions the NOMAD database records for `site`, resolved via lookup_ion.

    Ions neither pyrovskite nor mendeleev can supply a radius for still appear
    here with ionic_radius_angstrom=None; filter those out before building.
    """
    return {symbol: lookup_ion(site, symbol) for symbol in _NOMAD_IONS.get(site, {})}


def build_monatomic_perovskite(a_symbol: str, b_symbol: str, x_symbol: str, bx_distance: float | None = None):
    """Build a bulk ABX3 perovskite with a monatomic A-site cation.

    `bx_distance` (the B-X bond length in angstrom, passed straight through to
    pyrovskite's make_bulk) defaults to the sum of the B and X ionic radii.

    Returns the ase.Atoms structure and the {"A", "B", "X"}: Ion mapping used
    to build it.
    """
    if a_symbol not in MONATOMIC_A_SITE_SYMBOLS:
        raise ValueError(
            f"{a_symbol!r} is not a supported monatomic A-site cation "
            f"(supported: {sorted(MONATOMIC_A_SITE_SYMBOLS)}); "
            "organic A-site cations aren't wired up yet."
        )

    ions = {
        "A": lookup_ion("A", a_symbol),
        "B": lookup_ion("B", b_symbol),
        "X": lookup_ion("X", x_symbol),
    }

    if bx_distance is None:
        b_radius = ions["B"].ionic_radius_angstrom
        x_radius = ions["X"].ionic_radius_angstrom
        if b_radius is None or x_radius is None:
            raise ValueError(
                f"No ionic radius available for B={ions['B'].label} (r={b_radius}) "
                f"or X={ions['X'].label} (r={x_radius}); pass bx_distance explicitly."
            )
        bx_distance = b_radius + x_radius

    structure = make_bulk(a_symbol, b_symbol, x_symbol, bx_distance)
    return structure, ions
