"""Look up pseudopotential files and their recommended plane-wave cutoffs.

Pulls from data/pseudopotentials/pbesol_library (UPF files) and
data/pseudopotentials/cutoffs.json (per-element cutoff_wfc/cutoff_rho, in Ry),
an SSSP-style PBEsol pseudopotential set.
"""

import json
from upf_tools import UPFDict

from src.paths import PSEUDO_DIR


PSEUDO_ROOT = PSEUDO_DIR
LIBRARY_DIR = PSEUDO_ROOT / "pbesol_library"
CUTOFFS_PATH = PSEUDO_ROOT / "cutoffs.json"

_CUTOFFS = json.loads(CUTOFFS_PATH.read_text())


def _library_files() -> dict[str, str]:
    """Map element symbol -> UPF filename, keyed by the filename's leading '<Symbol>.' token."""
    return {path.name.split(".", 1)[0]: path.name for path in LIBRARY_DIR.iterdir()}


_FILES_BY_ELEMENT = _library_files()

# Elements whose only library pseudopotential crashes pw.x on this project's QE
# build. Eu used to be here: under QE 7.3.1 its legacy atompaw PAW dataset hit
# 'Error in routine ylmr2: l too large' (then segfaulted at higher ecutrho),
# but it runs cleanly under QE 7.5 (checked on CsEuCl3, mp-1068377).
BROKEN_ELEMENTS: dict[str, str] = {}


def pseudopotential_filename(symbol: str) -> str:
    """Look up the PBEsol UPF filename for an element symbol."""
    if symbol in BROKEN_ELEMENTS:
        raise ValueError(f"{symbol} pseudopotential is known broken: {BROKEN_ELEMENTS[symbol]}")
    filename = _FILES_BY_ELEMENT.get(symbol)
    if filename is None:
        raise ValueError(f"No PBEsol pseudopotential found for element {symbol!r} in {LIBRARY_DIR}")
    return filename


def pseudopotentials_for(symbols: list[str]) -> dict[str, str]:
    """Map each element symbol to its PBEsol UPF filename."""
    return {symbol: pseudopotential_filename(symbol) for symbol in dict.fromkeys(symbols)}


def suggested_cutoffs(symbols: list[str]) -> tuple[float, float]:
    """Recommended (ecutwfc, ecutrho) in Ry for a structure containing `symbols`.

    A single plane-wave cutoff applies to the whole calculation, so this is the
    max over each element's individually recommended cutoff, per SSSP convention.
    """
    missing = [s for s in symbols if s not in _CUTOFFS]
    if missing:
        raise ValueError(f"No cutoff data for: {missing}")
    ecutwfc = max(_CUTOFFS[s]["cutoff_wfc"] for s in symbols)
    ecutrho = max(_CUTOFFS[s]["cutoff_rho"] for s in symbols)
    return ecutwfc, ecutrho


def valence_electrons(symbol: str) -> int:
    """Look up the number of valence electrons from the pseudopotential file."""
    upf_file = LIBRARY_DIR / pseudopotential_filename(symbol)
    return UPFDict.from_upf(upf_file)["header"]["z_valence"]
