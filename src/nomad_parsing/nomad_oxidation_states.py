"""Determine which oxidation states a B-site element actually takes in reported perovskites.

The small NOMAD ion reference table (see nomad_ions.py) assigns exactly one
fixed oxidation state per element abbreviation, so it can't reveal whether an
element is reported with more than one oxidation state across the literature.
This module instead scans the underlying `PerovskiteSolarCell` entries (the
migrated Perovskite Database Project, ~51k device records), each of which
carries its own embedded ion list with a molecular formula per reported ion,
and tallies the distinct oxidation states actually declared for each element.
"""

import json
import time
from collections import Counter
from pathlib import Path

import requests

from .nomad_ions import parse_oxidation_state

BASE_URL = "http://nomad-lab.eu/prod/v1/api/v1"
CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "nomad" / "nomad_b_site_oxidation_states.json"
)

# NOMAD rate-limits this endpoint to one in-flight request, with 5s between requests.
REQUEST_INTERVAL_SECONDS = 5
PAGE_SIZE = 100
MAX_RECORDS_PER_ELEMENT = 1000
EXAMPLES_PER_OXIDATION_STATE = 3
MAX_RETRIES = 3


def _post_with_retries(json_body: dict) -> dict:
    """POST to the archive query endpoint, retrying on transient network errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(f"{BASE_URL}/entries/archive/query", json=json_body, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(REQUEST_INTERVAL_SECONDS * attempt)


def _fetch_oxidation_states(element: str, ion_type: str = "B") -> dict:
    """Tally the distinct oxidation states declared for `element` across real device records."""
    json_body = {
        "owner": "visible",
        "query": {
            "entry_type": "PerovskiteSolarCell",
            "results.material.elements:all": [element],
        },
        "pagination": {"page_size": PAGE_SIZE},
    }
    counts = Counter()
    examples = {}
    seen_records = 0
    total_records = None

    while seen_records < MAX_RECORDS_PER_ELEMENT:
        payload = _post_with_retries(json_body)
        total_records = payload["pagination"].get("total", total_records)

        for entry in payload["data"]:
            perovskite = entry["archive"]["data"].get("perovskite", {})
            for ion in perovskite.get("ions", []):
                if ion["name"] != element or ion["ion_type"] != ion_type:
                    continue
                oxidation_state = parse_oxidation_state(
                    {"smiles": ion.get("smile"), "molecular_formula": ion.get("molecular_formula")}
                )
                counts[oxidation_state] += 1
                bucket = examples.setdefault(oxidation_state, [])
                long_form = perovskite.get("composition_long_form")
                if long_form and len(bucket) < EXAMPLES_PER_OXIDATION_STATE:
                    bucket.append(long_form)
        seen_records += len(payload["data"])

        next_value = payload["pagination"].get("next_page_after_value")
        if not next_value or seen_records >= total_records:
            break
        json_body["pagination"]["page_after_value"] = next_value
        time.sleep(REQUEST_INTERVAL_SECONDS)

    return {
        "total_records": total_records,
        "sampled_records": seen_records,
        "oxidation_states": {
            str(state): {"count": count, "examples": examples[state]}
            for state, count in counts.most_common()
        },
    }


def survey_b_site_oxidation_states(elements: list[str], results: dict[str, dict]) -> dict[str, dict]:
    """Survey declared oxidation states for each B-site element across real device records.

    Mutates and returns `results` in place, saving to CACHE_PATH after each element so
    progress survives a transient failure partway through the (slow, rate-limited) survey.
    """
    for element in elements:
        if element in results:
            continue
        results[element] = _fetch_oxidation_states(element)
        CACHE_PATH.write_text(json.dumps(results, indent=2))
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return results


if __name__ == "__main__":
    ions_cache = json.loads(
        (Path(__file__).resolve().parent.parent.parent / "data" / "nomad" / "nomad_ions.json").read_text()
    )
    b_site_elements = sorted(ions_cache["B"])

    results = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    results = survey_b_site_oxidation_states(b_site_elements, results)

    for element, data in results.items():
        states = data["oxidation_states"]
        flag = " <-- MULTIPLE OXIDATION STATES" if len(states) > 1 else ""
        print(
            f"{element}: sampled {data['sampled_records']}/{data['total_records']} records, "
            f"states={ {s: v['count'] for s, v in states.items()} }{flag}"
        )
    print(f"Saved to {CACHE_PATH}")
