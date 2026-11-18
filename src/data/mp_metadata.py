"""Conventions of the MP perovskite metadata CSVs (data/mp_perovskites/
metadata*.csv) that more than one script depends on. No mp-api import here:
src/data/mp_perovskites.py is the part that talks to the Materials Project.
"""

from __future__ import annotations


def mp_run_name(material_id: str, formula_pretty: str) -> str:
    """<material_id>_<formula>: the CIF stem, QE input directory, MLIP run
    directory and clustering row name of one MP entry."""
    return f"{material_id}_{formula_pretty.replace(' ', '')}"


def split_sites(field) -> list[str]:
    """Species of an a_sites / b_sites field ("Cs;Rb"); [] for an empty or
    missing (NaN) one."""
    return [s for s in field.split(";") if s] if isinstance(field, str) else []


def filter_stable_rows(rows: list[dict], max_e_above_hull: float) -> tuple[list[dict], dict[str, int]]:
    """Rows verified as perovskites (verified_perovskite == "True") within
    max_e_above_hull (eV/atom) of the hull, and how many each test dropped."""
    stats = {"not_verified": 0, "above_threshold": 0}

    kept = []
    for row in rows:
        if str(row["verified_perovskite"]) != "True":
            stats["not_verified"] += 1
            continue
        if float(row["energy_above_hull"]) > max_e_above_hull:
            stats["above_threshold"] += 1
            continue
        kept.append(row)

    return kept, stats
