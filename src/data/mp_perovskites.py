"""Curate a perovskite-compound dataset from the Materials Project.

Traversal strategy follows the discussion in
https://matsci.org/t/traversing-the-materials-project-database-for-perovskite-compounds/49331/7:
no single MP query is a complete catalog of perovskites, so results from
several independent searches are merged and deduplicated by material_id.

That thread's second leg -- a keyword search on the `provenance` endpoint --
no longer exists in the current MP API (mp-api 0.46.5): `ProvenanceRester.search`
only accepts `material_ids`/`deprecated`, not free-text keywords (verified
against both the installed client and the live /materials/provenance/ OpenAPI
schema). It's replaced here with an anonymized-formula search on the
`summary` endpoint (e.g. formula="ABC3", the MP API's mechanism for
stoichiometry-pattern queries), which is a strict, current-API-compatible
stand-in for "give me every literal ABX3 perovskite" -- at the cost of
missing anything robocrys's text search would catch by structural
description alone (e.g. vacancy-ordered or off-stoichiometry perovskites).
Combining both, as the thread does, trades off recall against precision in
opposite directions, so results are reported per-source before merging.
"""

from __future__ import annotations

import os
from pathlib import Path

from mp_api.client import MPRester

# Anonymized-formula patterns for the summary-search leg: simple ABX3
# perovskites and A2BB'X6 double perovskites. Ruddlesden-Popper phases
# (AB2C4, ...) and other layered variants are deliberately left out -- they're
# a large, mostly non-perovskite population that would swamp true hits.
DEFAULT_FORMULA_PATTERNS = ["ABC3", "ABC2D6"]

DEFAULT_KEYWORDS = ["perovskite"]

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "elements",
    "nsites",
    "symmetry",
    "energy_above_hull",
    "is_stable",
    "theoretical",
    "band_gap",
    "is_gap_direct",
    "is_metal",
    "density",
    "volume",
    "deprecated",
    "structure",
]


def resolve_api_key(api_key: str | None = None) -> str:
    """API key precedence: explicit arg > MP_API_KEY env var > ~/.mp_api_key file.

    Never hardcode a key in a script; get one at
    https://next-gen.materialsproject.org/api#api-key
    """
    if api_key:
        return api_key
    env_key = os.environ.get("MP_API_KEY")
    if env_key:
        return env_key
    key_file = Path.home() / ".mp_api_key"
    if key_file.exists():
        stripped = key_file.read_text().strip()
        if stripped:
            return stripped
    raise RuntimeError(
        "No Materials Project API key found. Pass --api-key, set the MP_API_KEY "
        "environment variable, or save your key (and nothing else) to ~/.mp_api_key. "
        "Get a key at https://next-gen.materialsproject.org/api#api-key"
    )


def search_by_keyword(mpr: MPRester, keywords: list[str]) -> set[str]:
    """material_ids whose robocrystallographer description text matches `keywords`.

    Normalized to plain str: the client returns MPID objects, whose __lt__
    assumes both sides are MPID-formatted and raises on comparison against a
    non-MPID-like value, which breaks a plain sorted()/set() over mixed
    sources downstream.
    """
    docs = mpr.materials.robocrys.search(keywords=keywords)
    return {str(d.material_id) for d in docs}


def search_by_anonymous_formula(mpr: MPRester, pattern: str) -> set[str]:
    """material_ids whose reduced composition matches an anonymized stoichiometry
    pattern such as "ABC3" (simple perovskite) or "ABC2D6" (double perovskite)."""
    docs = mpr.materials.summary.search(formula=pattern, fields=["material_id"])
    return {str(d.material_id) for d in docs}


def collect_candidate_ids(
    mpr: MPRester,
    keywords: list[str] = DEFAULT_KEYWORDS,
    formula_patterns: list[str] = DEFAULT_FORMULA_PATTERNS,
) -> dict[str, set[str]]:
    """Run every traversal strategy independently, keyed by source name, so
    overlap and per-source yield can be inspected before merging."""
    sources: dict[str, set[str]] = {}
    if keywords:
        sources["robocrys_keywords"] = search_by_keyword(mpr, keywords)
    for pattern in formula_patterns:
        sources[f"formula:{pattern}"] = search_by_anonymous_formula(mpr, pattern)
    return sources


def fetch_summary_docs(mpr: MPRester, material_ids: list[str], chunk_size: int = 1000):
    """Fetch full summary docs (incl. structure) for material_ids.

    Requests are chunked so a large candidate set doesn't get flattened into
    one unbounded comma-separated query string.
    """
    docs = []
    ids = sorted(material_ids)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        docs.extend(mpr.materials.summary.search(material_ids=chunk, fields=SUMMARY_FIELDS))
    return docs


METADATA_FIELDNAMES = [
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "nsites",
    "spacegroup_symbol",
    "spacegroup_number",
    "crystal_system",
    "energy_above_hull",
    "is_stable",
    "theoretical",
    "band_gap",
    "is_metal",
    "density",
    "volume",
    "deprecated",
    "found_via",
    "cif_path",
]


def _spacegroup_fields(doc):
    symmetry = doc.symmetry
    if symmetry is None:
        return None, None, None
    crystal_system = getattr(symmetry.crystal_system, "value", symmetry.crystal_system)
    return symmetry.symbol, symmetry.number, crystal_system


def summary_row(doc, found_via: list[str], cif_path: str) -> dict:
    """One metadata.csv row (METADATA_FIELDNAMES) from a summary document."""
    spacegroup_symbol, spacegroup_number, crystal_system = _spacegroup_fields(doc)
    return {
        "material_id": doc.material_id,
        "formula_pretty": doc.formula_pretty,
        "formula_anonymous": doc.formula_anonymous,
        "nsites": doc.nsites,
        "spacegroup_symbol": spacegroup_symbol,
        "spacegroup_number": spacegroup_number,
        "crystal_system": crystal_system,
        "energy_above_hull": doc.energy_above_hull,
        "is_stable": doc.is_stable,
        "theoretical": doc.theoretical,
        "band_gap": doc.band_gap,
        "is_metal": doc.is_metal,
        "density": doc.density,
        "volume": doc.volume,
        "deprecated": doc.deprecated,
        "found_via": ";".join(found_via),
        "cif_path": cif_path,
    }
