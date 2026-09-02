"""Diagnose and recover from Quantum ESPRESSO pw.x SCF failures across a
batch of generated perovskite structures.

The one failure mode this handles is QE's

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         from electrons : error #         1
         charge is wrong: smearing is needed
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

which happens because generate_qe_input.QEInputConfig defaults to
occupations='fixed': that requires an unambiguous integer count of filled
bands with a real gap above them, and plain PBEsol -- with no Hubbard U
correction and no spin polarization -- frequently comes out gapless or has
near-degenerate states at the Fermi level for open d-shell transition-metal
(Co, Cu, Mn, Cr, ...) and open f-shell lanthanide (Eu, Sm, ...) B-site ions,
which generate_inputs.py sweeps through with no element-specific electronic-
structure handling. Which exact ion/halide combinations hit this isn't a
clean per-element rule -- e.g. Fe- and Ni-containing analogues commonly do
NOT hit it -- it depends on where the (possibly spurious, self-interaction-
error-driven) Fermi level happens to land for that particular structure. A
handful of failures with no open d/f shell at all are more likely an extreme
A/B ionic-radius mismatch (see perovskite_builder.py's sum-of-radii bond
length heuristic) producing a badly strained, accidentally metallic proxy
structure rather than a correlated-electron issue.

The fix here, occupations='smearing', lets pw.x's SCF loop converge either
way. It does NOT fix the deeper issue for genuinely correlated d/f-electron
materials: PBEsol without Hubbard U or explicit spin-polarization may still
describe them as non-magnetic and/or metallic when the real material is an
insulating antiferromagnet -- that's a real physics decision (which U, which
magnetic ordering to try) this module deliberately leaves to you rather than
guessing. It's also worth knowing that if a structure turns out genuinely
metallic once smearing lets you see it, the DFPT step's epsil=True/zeu=True
(Born effective charges + LO-TO splitting, on by default in QEInputConfig)
is only valid for insulators -- expect that step to need epsil=False, or to
fail outright, for any of these that come out metallic.
"""

from __future__ import annotations

from pathlib import Path

import ase.io

from generate_qe_input import generate_inputs_for_structure, pbesol_config_for

SCF_CHARGE_ERROR = "charge is wrong: smearing is needed"


def find_scf_charge_errors(scf_root: Path, pattern: str = "*/*.scf.out") -> list[str]:
    """Structure names whose *.scf.out log (under scf_root, one subdirectory
    per structure, matching `pattern`) contains QE's fixed-occupations charge
    error. Scanning the actual logs (rather than e.g. inferring failure from
    an empty downstream ph_first/<name>/ directory) also correctly excludes
    structures that are simply still queued, or failed for an unrelated
    reason (OOM, walltime) that smearing won't fix.
    """
    names = []
    for out_path in sorted(Path(scf_root).glob(pattern)):
        text = out_path.read_text(errors="ignore")
        if SCF_CHARGE_ERROR in text:
            names.append(out_path.parent.name)
    return names


def regenerate_with_smearing(
    names: list[str],
    structures_dir: Path,
    qe_inputs_dir: Path,
    smearing: str = "gaussian",
    degauss: float = 0.01,
    structure_suffix: str = ".cif",
) -> list[str]:
    """Re-render each name's SCF+DFPT inputs in place, at qe_inputs_dir/<name>/,
    with occupations='smearing' instead of the default 'fixed', reusing the
    same structure geometry already in structures_dir.

    Returns the names actually regenerated, skipping (with a printed
    warning) any whose structure file isn't found rather than raising, so
    one missing file doesn't abort the whole batch.
    """
    done = []
    for name in names:
        structure_path = Path(structures_dir) / f"{name}{structure_suffix}"
        if not structure_path.exists():
            print(f"Skipping {name}: no structure file at {structure_path}")
            continue
        atoms = ase.io.read(structure_path)
        config = pbesol_config_for(atoms, occupations="smearing", smearing=smearing, degauss=degauss)
        generate_inputs_for_structure(structure_path, config, Path(qe_inputs_dir) / name)
        done.append(name)
    return done