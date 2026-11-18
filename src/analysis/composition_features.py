"""Composition-side inputs (tier C in descriptor_selection.ipynb) for one
perovskite, from its formula and A/B/X site assignment: per-site Shannon
radius, mass, Pauling electronegativity and formal oxidation state, plus the
Goldschmidt tolerance and octahedral factors.

Same definitions as scripts/analysis/cluster_compositions.py's
mp_descriptor_vector (an unweighted mean over the species listed on a site,
radii at the measured site CN; pass weights for a mixed site whose species
aren't 1:1, e.g. the X site of CsPbBr2Cl), so for every structure that function could
handle the values here are identical. What differs is what happens when a
lookup fails. mp_descriptor_vector drops the whole structure; here each
feature is NaN only if that feature itself can't be resolved, and fallbacks
cover the common gaps. Oxidation states, in order (oxi_source):

    guess<i>   pymatgen's charge-balance guesses over common oxidation states,
               in rank order: the first whose states all have a radius (the
               top guess is sometimes an unphysical state with none -- In2+,
               W2+, Bi4+ -- while a lower-ranked one is fine)
    bv         bond-valence analysis of the structure, per site: resolves
               mixed valence (Tl+/Tl3+ in CsTlCl3, Fe3+/Fe4+) and states the
               common list lacks (Pd3+, Fe4+, I7+), with each species'
               radius averaged over its sites
    all_states pymatgen's guesses over every known oxidation state (Os5+ in
               NaOsO3); less reliable, so it comes last
    none       nothing charge-balances: radii and oxidation states are NaN

bv and all_states are used only when the X site is a conventional anion
(ANION_STATE), and only assignments that put X at its standard state and
every fixed-valence cation (FIXED_VALENCE: alkali, alkaline earth, Sc/Y/
lanthanides with one common state, Zn, Cd, Al, Ga) at its one state are
accepted -- both fallbacks otherwise sometimes balance charge with
O^-1.67 (BaBiO3) or Mg+. On intermetallics and antiperovskites with a metallic X site
(ScRh3C, Mn3ZnN) they return formal states like Sc+ / Rh+ that ionic radii
can't be looked up for meaningfully. Masses and electronegativities don't
need an oxidation state and are filled regardless.

x_frac_<halide> is each halide's share of the X site (0 for a non-halide
X), so mixed and pure halides sit on one Cl-Br-I simplex.

Radii: Shannon (mendeleev), then RADIUS_FALLBACK for perovskite ions
Shannon's table lacks. Electronegativity: mendeleev's Pauling value, then
pymatgen's (mendeleev has none for Eu, Tb, Yb). radius_source / en_source
give the worst case over a row's species, so a model can be checked with the
fallback rows left out.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path

import ase.data
import numpy as np
from mendeleev import element as mendeleev_element
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Composition, Element, Structure

from src.chemistry import Ion, octahedral_factor, shannon_ionic_radius, tolerance_factor
from src.data.mp_metadata import mp_run_name
from src.data.perovskite_builder import SITE_COORDINATION

HALIDES = ("F", "Cl", "Br", "I")

FEATURE_NAMES = [
    "r_A", "r_B", "r_X", "tolerance_factor", "octahedral_factor",
    "mass_A", "mass_B", "mass_X",
    "en_A", "en_B", "en_X",
    "oxidation_A", "oxidation_B", "oxidation_X",
] + [f"x_frac_{h}" for h in HALIDES]

ANION_STATE = {"O": -2, "S": -2, "Se": -2, "Te": -2, "N": -3, "F": -1, "Cl": -1, "Br": -1, "I": -1}

# (symbol, oxidation state) -> radius (A), any CN. Sn2+: the value commonly
# used for Sn(II) halide perovskite tolerance factors; Shannon (1976) has no
# Sn2+ entry.
RADIUS_FALLBACK = {("Sn", 2): 1.10}

MAX_OXI_GUESSES = 20  # pymatgen ranks guesses by likelihood; deep ones are rarely physical


@lru_cache(maxsize=None)
def radius(symbol: str, oxidation_state: int, coordination: int) -> tuple[float, str] | tuple[None, None]:
    """(radius in A, "shannon" | "fallback"), or (None, None)."""
    r = shannon_ionic_radius(symbol, oxidation_state, coordination)
    if r is not None:
        return r, "shannon"
    if (symbol, oxidation_state) in RADIUS_FALLBACK:
        return RADIUS_FALLBACK[(symbol, oxidation_state)], "fallback"
    return None, None


@lru_cache(maxsize=None)
def electronegativity(symbol: str) -> tuple[float, str] | tuple[None, None]:
    """(Pauling electronegativity, "mendeleev" | "pymatgen"), or (None, None)."""
    el = mendeleev_element(symbol)
    value = getattr(el, "en_pauling", None)
    if value is None:
        try:
            value = el.electronegativity(scale="pauling")
        except Exception:
            value = None
    if value is not None:
        return value, "mendeleev"
    value = Element(symbol).X
    if value is not None and not math.isnan(value):
        return value, "pymatgen"
    return None, None


@lru_cache(maxsize=None)
def oxidation_guesses(formula: str, all_oxi_states: bool = False) -> tuple[dict[str, float], ...]:
    try:
        guesses = Composition(formula).oxi_state_guesses(all_oxi_states=all_oxi_states)[:MAX_OXI_GUESSES]
    except Exception:
        guesses = ()
    return tuple(dict(g) for g in guesses)


@lru_cache(maxsize=None)
def fixed_valence(symbol: str) -> int | None:
    """The one oxidation state of a cation that has no other in perovskites, else None."""
    el = Element(symbol)
    common = el.common_oxidation_states
    if el.group in (1, 2) and len(common) == 1:
        return common[0]
    if len(common) == 1 and (el.is_rare_earth or symbol in {"Zn", "Cd", "Al", "Ga"}):
        return common[0]
    return None


def plausible(mean: dict[str, float], x_sites: list[str]) -> bool:
    """X at its standard state and fixed-valence cations at theirs (for the
    bv / all_states fallbacks)."""
    for s, v in mean.items():
        want = ANION_STATE[s] if s in x_sites else fixed_valence(s)
        if want is not None and abs(v - want) > 1e-6:
            return False
    return True


def bond_valence_states(structure: Structure) -> dict[str, list[int]] | None:
    """{element: integer valence of each of its sites}, or None if pymatgen's
    BVAnalyzer finds no charge-balanced assignment."""
    try:
        valences = BVAnalyzer().get_valences(structure)
    except Exception:
        return None
    out: dict[str, list[int]] = {}
    for site, v in zip(structure, valences):
        out.setdefault(site.specie.symbol, []).append(int(v))
    return out


def bond_valence_ions(structure: Structure) -> dict[str, Ion] | None:
    """Best-effort {element: Ion} from bond-valence sums, taking the first
    site's valence per element (true mixed-valence sites of one element are
    rare here, and this only feeds the smearing/magnetism decision of
    generate_qe_input.pbesol_config_for). None if BVAnalyzer can't resolve
    the structure; callers should then fall back to no magnetism detection
    rather than fail the structure over it."""
    states = bond_valence_states(structure)
    if states is None:
        return None
    return {symbol: Ion(symbol=symbol, oxidation_state=valences[0]) for symbol, valences in states.items()}


def parse_site_cn(cn_field: str) -> dict[str, float]:
    """"Sym=12.00;Sym2=8.50" (metadata_stable.csv's a_site_cn/b_site_cn) ->
    {symbol: measured coordination number}."""
    out = {}
    for part in (cn_field or "").split(";"):
        symbol, _, value = part.strip().partition("=")
        if symbol:
            out[symbol] = float(value)
    return out


def _species_radius(symbol: str, states: list[int], coordination: int) -> tuple[float | None, str | None]:
    """Radius of one species averaged over its sites' states."""
    looked_up = [radius(symbol, s, coordination) for s in states]
    if any(r is None for r, _ in looked_up):
        return None, None
    return float(np.mean([r for r, _ in looked_up])), "fallback" if any(src == "fallback" for _, src in looked_up) else "shannon"


def _radii(sites: dict[str, list[str]], cn: dict[str, dict[str, float]], states: dict[str, list[int]]) -> dict:
    """{site: [(radius, source) per species]} under one oxidation-state assignment."""
    return {site: [_species_radius(s, states[s], round(cn[site].get(s, SITE_COORDINATION[site])))
                   if s in states else (None, None) for s in symbols]
            for site, symbols in sites.items()}


def _resolves(radii: dict) -> bool:
    return all(r is not None for rs in radii.values() for r, _ in rs)


def composition_features(formula: str, a_sites: list[str], b_sites: list[str], x_sites: list[str],
                         a_cn: dict[str, float] | None = None, b_cn: dict[str, float] | None = None,
                         structure: Structure | None = None, weights: dict[str, float] | None = None) -> dict:
    """FEATURE_NAMES -> value (NaN where unresolvable), plus oxi_source,
    radius_source ("shannon" | "fallback" | "missing") and en_source
    ("mendeleev" | "pymatgen" | "missing"). structure is needed only for the
    bond-valence fallback. weights: {species: weight} for the site means
    (default 1 each)."""
    sites = {"A": list(a_sites), "B": list(b_sites), "X": list(x_sites)}
    cn = {"A": a_cn or {}, "B": b_cn or {}, "X": {}}
    w = {s: (weights or {}).get(s, 1.0) for s in sum(sites.values(), [])}
    mean_over = lambda symbols, values: float(np.average(values, weights=[w[s] for s in symbols]))
    out: dict = {name: np.nan for name in FEATURE_NAMES}
    x_total = sum(w[s] for s in sites["X"])
    for h in HALIDES:
        out[f"x_frac_{h}"] = w[h] / x_total if h in sites["X"] else 0.0
    if not all(sites.values()):
        return out | {"oxi_source": "none", "radius_source": "missing", "en_source": "missing"}

    for site, symbols in sites.items():
        out[f"mass_{site}"] = mean_over(symbols, [ase.data.atomic_masses[ase.data.atomic_numbers[s]] for s in symbols])
        ens = [electronegativity(s) for s in symbols]
        if all(v is not None for v, _ in ens):
            out[f"en_{site}"] = mean_over(symbols, [v for v, _ in ens])
    en_sources = {electronegativity(s)[1] for s in sum(sites.values(), [])}
    out["en_source"] = "missing" if None in en_sources else "pymatgen" if "pymatgen" in en_sources else "mendeleev"

    # candidate assignments, in order of preference: (source, {element: mean state}, {element: per-site states})
    def candidates():
        for i, g in enumerate(oxidation_guesses(formula)):
            yield f"guess{i}", g, {s: [round(v)] for s, v in g.items()}
        if not set(x_sites) <= ANION_STATE.keys():
            return
        if structure is not None and (bv := bond_valence_states(structure)) is not None:
            mean = {s: float(np.mean(v)) for s, v in bv.items()}
            if plausible(mean, x_sites):
                yield "bv", mean, bv
        for g in oxidation_guesses(formula, all_oxi_states=True):
            if plausible(g, x_sites):
                yield "all_states", g, {s: [round(v)] for s, v in g.items()}

    chosen = None
    for source, mean, states in candidates():
        radii = _radii(sites, cn, states)
        if chosen is None:
            chosen = source, mean, radii  # kept if nothing resolves every radius
        if _resolves(radii):
            chosen = source, mean, radii
            break
    if chosen is None:
        return out | {"oxi_source": "none", "radius_source": "missing"}
    out["oxi_source"], mean, radii = chosen

    for site, symbols in sites.items():
        if all(s in mean for s in symbols):
            out[f"oxidation_{site}"] = mean_over(symbols, [mean[s] for s in symbols])
        if all(r is not None for r, _ in radii[site]):
            out[f"r_{site}"] = mean_over(symbols, [r for r, _ in radii[site]])
    r_sources = {src for rs in radii.values() for _, src in rs}
    out["radius_source"] = "missing" if None in r_sources else "fallback" if "fallback" in r_sources else "shannon"

    out["tolerance_factor"] = tolerance_factor(out["r_A"], out["r_B"], out["r_X"])
    out["octahedral_factor"] = octahedral_factor(out["r_B"], out["r_X"])
    return out


def mp_feature_rows(metadata: Path) -> list[dict]:
    """composition_features for every row of an MP metadata CSV (structures
    read from its cif_path column, relative to the CSV). Rows whose metadata
    lacks an A/B assignment get one from design_descriptors.assign_sublattices
    (site_source 'econ')."""
    import pandas as pd
    from pymatgen.io.ase import AseAtomsAdaptor

    from src.analysis.design_descriptors import assign_sublattices, oxidation_states

    meta = pd.read_csv(metadata, keep_default_na=False)
    rows = []
    for _, m in meta.iterrows():
        structure = Structure.from_file(metadata.parent / m["cif_path"])
        a_sites, b_sites = ([s for s in m[field].split(";") if s] for field in ("a_sites", "b_sites"))
        x_sites = [m["x_site"]] if m["x_site"] else []
        site_source = "metadata"
        if x_sites and not (a_sites and b_sites):
            atoms = AseAtomsAdaptor.get_atoms(structure)
            try:
                roles = assign_sublattices(atoms, oxidation_states(m["formula_pretty"]), set(a_sites), set(b_sites), m["x_site"])
            except ValueError:
                roles = None
            if roles is not None:
                symbols = atoms.get_chemical_symbols()
                a_sites, b_sites = (sorted({s for s, r in zip(symbols, roles) if r == site}) for site in "AB")
                site_source = "econ"
        rows.append({"name": mp_run_name(m["material_id"], m["formula_pretty"]), "site_source": site_source}
                    | composition_features(m["formula_pretty"], a_sites, b_sites, x_sites,
                                           parse_site_cn(m["a_site_cn"]), parse_site_cn(m["b_site_cn"]), structure))
    return rows


def generated_feature_rows(compositions: Path) -> list[dict]:
    """composition_features for every structure of a generated mixed-halide
    set's compositions.csv (scripts/dataset/generate_mixed_halides.py)."""
    import pandas as pd

    comp = pd.read_csv(compositions).dropna(subset=["structures"])
    rows = []
    for _, c in comp.iterrows():
        cif = compositions.parent / "structures" / f"{c['structures']}.cif"
        structure = Structure.from_file(cif) if cif.exists() else None
        rows.append({"name": c["structures"], "site_source": "generated"} | composition_features(
            c["formula"], c["A"].split(";"), c["B"].split(";"), [c["X1"], c["X2"]],
            structure=structure, weights={x: Composition(c["formula"])[x] for x in (c["X1"], c["X2"])}))
    return rows


# --- the 14-feature vector scripts/analysis/cluster_compositions.py clusters on ---
#
# The strict predecessor of composition_features: same quantities, but Shannon
# radii and mendeleev electronegativities only (no fallbacks), the first
# oxidation-state guess only, and a composition missing any lookup is dropped
# whole instead of getting NaN features. data/mp_composition_clusters.csv, and
# everything keyed on its cluster labels, was built from this.

DESCRIPTOR_NAMES = FEATURE_NAMES[:14]


def _strict(value_and_source: tuple, source: str) -> float | None:
    value, got = value_and_source
    return value if got == source else None


def descriptor_vector(ions: dict) -> np.ndarray | None:
    """DESCRIPTOR_NAMES vector for one {"A": Ion, "B": Ion, "X": Ion}
    composition (perovskite_builder.charge_balanced_abx3), or None if any ion
    is missing an ionic radius or electronegativity -- generate_inputs.py
    would have skipped building that composition's structure anyway."""
    r = {site: ions[site].ionic_radius_angstrom for site in "ABX"}
    if any(v is None for v in r.values()):
        return None
    mass = {site: ase.data.atomic_masses[ase.data.atomic_numbers[ions[site].symbol]] for site in "ABX"}
    en = {site: _strict(electronegativity(ions[site].symbol), "mendeleev") for site in "ABX"}
    if any(v is None for v in en.values()):
        return None
    return np.array([
        r["A"], r["B"], r["X"], tolerance_factor(r["A"], r["B"], r["X"]), octahedral_factor(r["B"], r["X"]),
        mass["A"], mass["B"], mass["X"],
        en["A"], en["B"], en["X"],
        ions["A"].oxidation_state, ions["B"].oxidation_state, ions["X"].oxidation_state,
    ])


def _site_descriptor(symbols: list[str], oxi: dict[str, float], cn_by_symbol: dict[str, float], site: str) -> tuple | None:
    """Average (radius, mass, electronegativity, oxidation_state) across the
    species occupying one sublattice (a_sites/b_sites can list more than one
    element, e.g. a mixed/double-perovskite site). None if any species is
    missing an oxidation-state guess, a Shannon radius, or an
    electronegativity value."""
    radii, masses, ens, oxis = [], [], [], []
    for symbol in symbols:
        oxidation_state = oxi.get(symbol)
        if oxidation_state is None:
            return None
        coordination = round(cn_by_symbol.get(symbol, SITE_COORDINATION[site]))
        r = _strict(radius(symbol, round(oxidation_state), coordination), "shannon")
        en = _strict(electronegativity(symbol), "mendeleev")
        if r is None or en is None:
            return None
        radii.append(r)
        masses.append(ase.data.atomic_masses[ase.data.atomic_numbers[symbol]])
        ens.append(en)
        oxis.append(oxidation_state)
    return sum(radii) / len(radii), sum(masses) / len(masses), sum(ens) / len(ens), sum(oxis) / len(oxis)


def mp_descriptor_vector(row: dict) -> np.ndarray | None:
    """DESCRIPTOR_NAMES vector for one metadata_verified.csv/metadata_stable.csv
    row, built from its structural site assignment (a_sites/b_sites/x_site,
    a_site_cn/b_site_cn) and a formula-level oxidation-state guess. None if
    any site's chemistry can't be resolved."""
    guesses = oxidation_guesses(row["formula_pretty"])
    if not guesses:
        return None
    oxi = guesses[0]

    a_symbols = [s for s in row["a_sites"].split(";") if s]
    b_symbols = [s for s in row["b_sites"].split(";") if s]
    x_symbols = [row["x_site"]] if row["x_site"] else []
    if not a_symbols or not b_symbols or not x_symbols:
        return None

    a = _site_descriptor(a_symbols, oxi, parse_site_cn(row["a_site_cn"]), "A")
    b = _site_descriptor(b_symbols, oxi, parse_site_cn(row["b_site_cn"]), "B")
    x = _site_descriptor(x_symbols, oxi, {}, "X")
    if a is None or b is None or x is None:
        return None
    r_a, mass_a, en_a, oxi_a = a
    r_b, mass_b, en_b, oxi_b = b
    r_x, mass_x, en_x, oxi_x = x
    return np.array([
        r_a, r_b, r_x, tolerance_factor(r_a, r_b, r_x), octahedral_factor(r_b, r_x),
        mass_a, mass_b, mass_x,
        en_a, en_b, en_x,
        oxi_a, oxi_b, oxi_x,
    ])


def mp_descriptor_table(csv_path: Path) -> tuple[list[str], list[np.ndarray], list[str]]:
    """names (<material_id>_<formula_pretty>), descriptor vectors, and skipped
    names for every row of an MP metadata CSV (metadata_verified.csv or
    metadata_stable.csv)."""
    names, vectors, skipped = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            vec = mp_descriptor_vector(row)
            name = mp_run_name(row["material_id"], row["formula_pretty"])
            if vec is None:
                skipped.append(name)
                continue
            names.append(name)
            vectors.append(vec)
    return names, vectors, skipped