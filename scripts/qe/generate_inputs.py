"""Generate cubic ABX3 structures and Quantum ESPRESSO SCF+DFPT inputs for every
sensible (A, B, X) combination of monatomic ions from the NOMAD ion database.

For a 1:1:3 corner-sharing perovskite to be charge-neutral, A + B + 3*X must
equal zero, so rather than fixing each site to one oxidation state, every
(A, B, X) combination is checked against that condition directly. This is what
lets chalcogenide X-sites (O2-, S2-) in: they need A + B = +6 instead of the
+3 halides need, which e.g. admits Nb5+ (previously excluded when B was
hardcoded to +2) paired with a +1 A-site -- NaNbO3 and KNbO3 are real,
well-known perovskites.

Site scope:
- A-site: monatomic cations only (organic "hybrid" A-site cations like MA/FA
  aren't wired up yet, see src/data/perovskite_builder.py).
- B-site: no restriction beyond what NOMAD reports; the charge-balance check
  above does the filtering.
- X-site: monatomic anions only. This excludes polyatomic pseudohalides
  (SCN-, BF4-, PF6-) for a geometric reason, not a chemical one: make_bulk
  places one ASE atom per site, and has no way to place a rotated multi-atom
  group there. Supporting those would need the same kind of work as organic
  A-site cations -- a real 3D geometry, an anchor atom, and an orientation --
  not just a looser filter.

Combinations pyrovskite/mendeleev can't supply an ionic radius for, or that
data/pseudopotentials has no PBEsol pseudopotential/cutoff for, are skipped
and reported rather than silently dropped.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ase.io

from src.paths import CUBIC_STRUCTURES_DIR
from src.qe.generate_qe_input import QE_INPUTS_DIR, generate_inputs_for_structure, pbesol_config_for
from src.data.perovskite_builder import build_monatomic_perovskite, candidate_ions, charge_balanced_abx3



def generate_all() -> None:
    a_ions, b_ions, x_ions = candidate_ions()
    print(
        f"A-site candidates ({len(a_ions)}): {sorted(a_ions)}\n"
        f"B-site candidates ({len(b_ions)}): {sorted(b_ions)}\n"
        f"X-site candidates ({len(x_ions)}): {sorted(x_ions)}\n"
    )

    CUBIC_STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    built = []
    skipped = []
    for name, site_ions in sorted(charge_balanced_abx3().items()):
        try:
            structure, ions = build_monatomic_perovskite(*(site_ions[site].symbol for site in "ABX"))
            cif_path = CUBIC_STRUCTURES_DIR / f"{name}.cif"
            ase.io.write(cif_path, structure)
            config = pbesol_config_for(structure, ions=ions)
            generate_inputs_for_structure(cif_path, config, QE_INPUTS_DIR / name)
        except ValueError as error:
            skipped.append((name, str(error)))
            continue
        built.append(name)

    print(f"Built {len(built)} structures + QE input pairs.")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    generate_all()
