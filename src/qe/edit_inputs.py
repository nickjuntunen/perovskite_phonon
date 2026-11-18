"""Derive a new QE input from the text of an existing one, changing only what
a follow-up run needs -- for inputs whose settings (smearing, magnetism,
k-points) were fixed up after generation and must carry over, where
re-rendering from the templates (generate_qe_input.py) would lose them.
"""

from __future__ import annotations

import re

# pw.x's default nstep for 'relax' is 50; see relax_input_from_scf.
RELAX_NSTEP = 500


def set_namelist_value(text: str, namelist: str, key: str, value: str) -> str:
    """Set `key = value` inside &NAMELIST ... /, replacing it if present."""
    block = re.search(rf"&{namelist}\b.*?\n/", text, flags=re.S | re.I)
    if block is None:
        raise ValueError(f"no &{namelist} namelist")
    body = block.group(0)
    line = f"  {key:<11} = {value}"
    if re.search(rf"^\s*{key}\s*=", body, flags=re.M | re.I):
        body = re.sub(rf"^\s*{key}\s*=.*$", line, body, flags=re.M | re.I)
    else:
        body = body[: -len("\n/")] + f"\n{line}\n/"
    return text[: block.start()] + body + text[block.end():]


def relax_input_from_scf(scf_text: str, atoms, prefix: str, forc_conv_thr: float, nstep: int = RELAX_NSTEP) -> str:
    """A pw.x fixed-cell 'relax' input from an existing scf input's text:
    same settings, positions replaced by `atoms`' (same atom order), prefix
    renamed, BFGS with force threshold `forc_conv_thr` (Ry/Bohr)."""
    text = set_namelist_value(scf_text, "CONTROL", "calculation", "'relax'")
    text = set_namelist_value(text, "CONTROL", "prefix", f"'{prefix}'")
    text = set_namelist_value(text, "CONTROL", "forc_conv_thr", f"{forc_conv_thr:g}")
    # pw.x's default nstep for 'relax' is 50, and it then stops *as if* done
    # ("End of BFGS Geometry Optimization" is printed either way). Descending
    # off a flat saddle with a tight forc_conv_thr takes more: CsPbBr3 hit the
    # cap at 0.015 eV/A, ~50x the threshold.
    text = set_namelist_value(text, "CONTROL", "nstep", str(nstep))
    if not re.search(r"&IONS\b", text, flags=re.I):
        electrons = re.search(r"&ELECTRONS\b.*?\n/\n", text, flags=re.S | re.I)
        text = text[: electrons.end()] + "&IONS\n  ion_dynamics = 'bfgs'\n/\n" + text[electrons.end():]

    header = re.search(r"^ATOMIC_POSITIONS\s+\{?crystal\}?\s*$\n", text, flags=re.M | re.I)
    if header is None:
        raise ValueError("expected ATOMIC_POSITIONS crystal (as templates/qe/pw_scf.in.j2 writes)")
    lines = text[header.end():].split("\n")
    nat = len(atoms)
    new_lines = [
        f"  {symbol:<4} {p[0]:14.10f} {p[1]:14.10f} {p[2]:14.10f}"
        for symbol, p in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=False))
    ]
    return text[: header.end()] + "\n".join(new_lines + lines[nat:])


def renamed_ph_input(ph_text: str, name: str, prefix: str) -> str:
    """A ph.x input's text re-pointed from run `name` to the pw.x run `prefix`
    (prefix, fildyn, a local outdir, and the title line)."""
    text = re.sub(r"(prefix\s*=\s*)'[^']*'", rf"\1'{prefix}'", ph_text)
    text = re.sub(r"(fildyn\s*=\s*)'[^']*'", rf"\1'{prefix}.dyn'", text)
    # The first run's outdir points at *its* save directory; run_ph.sbatch rewrites
    # this anyway, but a by-hand ph.x run must not read the unrattled wavefunctions.
    text = re.sub(r"(outdir\s*=\s*)'[^']*'", r"\1'./out'", text)
    return text.replace(f"for {name}\n", f"for {prefix}\n", 1)
