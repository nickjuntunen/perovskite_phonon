"""Fetch halide-perovskite ion data from the NOMAD perovskite solar cell database.

Queries https://nomad-lab.eu for PerovskiteAIon/BIon/XIon entries and converts
each one's SMILES charge notation (e.g. "[Pb+2]", "[I-]", "C[NH3+]") into an `Ion`.
"""

import json
import re
import time
from pathlib import Path

import requests

from ..chemistry import Ion, shannon_ionic_radius

BASE_URL = "http://nomad-lab.eu/prod/v1/api/v1"
SITE_SECTIONS = {
    "A": "PerovskiteAIon",
    "B": "PerovskiteBIon",
    "X": "PerovskiteXIon",
}
# Conventional perovskite coordination numbers, used to pick which Shannon
# radius to report for each site.
SITE_COORDINATION = {"A": 12, "B": 6, "X": 6}
CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "nomad" / "nomad_ions.json"

# NOMAD only allows one in-flight request to this endpoint, with 5s between requests.
REQUEST_INTERVAL_SECONDS = 5

# `molecular_formula` (e.g. "BF4-", "C8H21N2+") is ambiguous: a trailing digit
# can be a charge magnitude ("Pb2+") or just an atom-count subscript that
# happens to sit next to the sign ("BF4-" has one negative charge, not four;
# "C8H21N2+" is singly protonated, not +2). SMILES charge notation only ever
# appears inside brackets, so it doesn't have that ambiguity.
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_BRACKET_CHARGE_RE = re.compile(r"([+-])(\d*)$")


def _smiles_charge(smiles: str) -> int:
    """Sum the formal charges of every bracketed atom in a SMILES string."""
    total = 0
    for bracket in _BRACKET_RE.findall(smiles):
        match = _BRACKET_CHARGE_RE.search(bracket)
        if not match:
            continue
        sign_char, digits = match.groups()
        if digits:
            magnitude = int(digits)
        else:
            magnitude = len(re.search(rf"{re.escape(sign_char)}+$", bracket).group())
        total += magnitude if sign_char == "+" else -magnitude
    return total


def parse_oxidation_state(data: dict) -> int:
    """Extract an ion's net charge from its NOMAD entry, preferring `smiles`."""
    smiles = data.get("smiles")
    if smiles:
        return _smiles_charge(smiles)
    molecular_formula = data.get("molecular_formula")
    if not molecular_formula:
        raise ValueError(f"No smiles or molecular_formula to parse charge from: {data!r}")
    match = re.match(r"^.+?(?P<sign>[+-])(?P<magnitude>\d*)$", molecular_formula)
    if not match:
        raise ValueError(f"Could not parse charge from formula: {molecular_formula!r}")
    magnitude = int(match.group("magnitude") or 1)
    return magnitude if match.group("sign") == "+" else -magnitude


def _fetch_site_ions(section: str) -> dict[str, dict]:
    """Page through all NOMAD entries for one ion site (A, B, or X)."""
    json_body = {
        "owner": "visible",
        "query": {"results.eln.sections:any": [section]},
        "pagination": {"page_size": 10},
    }
    entries: dict[str, dict] = {}
    while True:
        response = requests.post(f"{BASE_URL}/entries/archive/query", json=json_body, timeout=30)
        response.raise_for_status()
        payload = response.json()
        for entry in payload["data"]:
            data = entry["archive"]["data"]
            entries[data["abbreviation"]] = data
        next_value = payload["pagination"].get("next_page_after_value")
        if not next_value:
            break
        json_body["pagination"]["page_after_value"] = next_value
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return entries


def fetch_all_ions() -> dict[str, dict[str, Ion]]:
    """Fetch A/B/X site ions from NOMAD and convert them into Ion instances."""
    ions_by_site: dict[str, dict[str, Ion]] = {}
    for site, section in SITE_SECTIONS.items():
        raw = _fetch_site_ions(section)
        coordination = SITE_COORDINATION[site]
        ions_by_site[site] = {}
        for abbreviation, data in raw.items():
            if not (data.get("smiles") or data.get("molecular_formula")):
                continue
            oxidation_state = parse_oxidation_state(data)
            radius = shannon_ionic_radius(abbreviation, oxidation_state, coordination)
            ions_by_site[site][abbreviation] = Ion(
                symbol=abbreviation,
                oxidation_state=oxidation_state,
                ionic_radius_angstrom=radius,
                coordination=coordination if radius is not None else None,
            )
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return ions_by_site


if __name__ == "__main__":
    ions_by_site = fetch_all_ions()

    CACHE_PATH.parent.mkdir(exist_ok=True)
    serializable = {
        site: {abbr: ion.__dict__ for abbr, ion in ions.items()}
        for site, ions in ions_by_site.items()
    }
    CACHE_PATH.write_text(json.dumps(serializable, indent=2))

    for site, ions in ions_by_site.items():
        print(f"{site}-site: {len(ions)} ions -> {sorted(ions)}")
    print(f"Saved to {CACHE_PATH}")
