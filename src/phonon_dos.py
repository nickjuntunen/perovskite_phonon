"""Generate Quantum ESPRESSO q2r.x + matdyn.x inputs to Fourier-interpolate a
full-Brillouin-zone phonon density of states (vDOS) from a completed ph.x
DFPT dispersion run, and read the result back.

This is the piece the existing pipeline doesn't have yet: generate_qe_input.py
+ ph_dfpt.in.j2 already run ph.x with ldisp over an nq1 x nq2 x nq3 grid,
which produces <prefix>.dyn1 .. <prefix>.dynN (the irreducible q-points) plus
<prefix>.dyn0 (the grid manifest -- see unstable_modes.is_numbered_dyn_file)
-- exactly the `fildyn` root q2r.x needs to build a real-space interatomic
force-constant file (`flfrc`). matdyn.x then Fourier-interpolates that force-
constant file onto an arbitrarily dense q-mesh (nk1 x nk2 x nk3, decoupled
from the coarse DFPT grid) and, with dos=.true., bins the resulting
frequencies into a histogram -- the vDOS this project's fitness functions and
clustering (see vdos_features.py) operate on.

Both q2r.x and matdyn.x are cheap (pure linear algebra on already-computed
force constants, no fresh electronic-structure work), so a dense nk mesh
costs almost nothing next to the DFPT run that produced fildyn's inputs.

Note matdyn.x's dos=.true. mode is a plain per-bin histogram count at bin
width deltaE (cm-1) -- it has no Gaussian-smearing knob of its own (unlike an
electronic DOS calculation). If you want a smoothed vDOS for feature
extraction or plotting, apply Gaussian smoothing after reading the file back
in -- see vdos_features.smooth_dos.
"""

from pathlib import Path

import jinja2
import numpy as np

from generate_qe_input import TEMPLATES_DIR

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def render_q2r_input(prefix: str, zasr: str = "simple") -> str:
    """Render the q2r.x input that turns <prefix>.dyn0..dynN (from a completed
    ldisp DFPT run) into a real-space force-constant file <prefix>.fc.

    `zasr` enforces the acoustic sum rule on the Born effective charges --
    the same physical quantity DFPT's `zeu=True` computes. 'simple' matches
    this pipeline's default epsil=True/zeu=True for insulating samples. For a
    structure generated with epsil=False/zeu=False (the open-shell/magnetic
    B-site branch in generate_qe_input.pbesol_config_for, or anything you ran
    with --metallic in scripts/fix_smearing_crashes.py), pass zasr='no' --
    there's no Born-charge sum rule to enforce if none were ever computed.
    """
    template = _env.get_template("q2r.in.j2")
    return template.render(fildyn=f"{prefix}.dyn", zasr=zasr, flfrc=f"{prefix}.fc")


def render_matdyn_dos_input(
    prefix: str,
    nk: tuple[int, int, int] = (20, 20, 20),
    delta_e_cm1: float = 0.5,
    asr: str = "simple",
) -> str:
    """Render the matdyn.x input that Fourier-interpolates <prefix>.fc onto a
    dense nk1 x nk2 x nk3 mesh and bins the resulting frequencies into a vDOS
    histogram (<prefix>.dos), at `delta_e_cm1`-wide bins.

    `nk` should be dense relative to the coarse DFPT q-grid (nq1/nq2/nq3 in
    ph_dfpt.in.j2/generate_qe_input.QEInputConfig.qpts) -- that's the entire
    point of the q2r/matdyn interpolation step, and it's cheap since it's
    linear algebra on the already-interpolated dynamical matrix, not a fresh
    electronic-structure calculation. `asr` should match whatever `zasr` was
    passed to render_q2r_input (matdyn.x re-applies the acoustic sum rule at
    its own interpolated q-points).
    """
    template = _env.get_template("matdyn_dos.in.j2")
    return template.render(
        asr=asr,
        flfrc=f"{prefix}.fc",
        fldos=f"{prefix}.dos",
        flfrq=f"{prefix}.freq_dos",
        nk=nk,
        deltaE=delta_e_cm1,
    )


def generate_dos_inputs(
    out_dir: Path,
    prefix: str,
    zasr: str = "simple",
    nk: tuple[int, int, int] = (20, 20, 20),
    delta_e_cm1: float = 0.5,
) -> tuple[Path, Path]:
    """Write q2r.in + matdyn_dos.in into an existing structure directory --
    wherever generate_qe_input.generate_inputs_for_structure already wrote
    <prefix>.scf.in/<prefix>.ph.in -- ready to run once that structure's ph.x
    ldisp run (producing <prefix>.dyn0..dynN there) has finished.

    Pass zasr='no' here (and it'll be threaded through to the matdyn.x input
    as asr='no' automatically) for a structure whose DFPT run had
    epsil=False/zeu=False.
    """
    out_dir = Path(out_dir)
    q2r_path = out_dir / f"{prefix}.q2r.in"
    matdyn_path = out_dir / f"{prefix}.matdyn_dos.in"
    q2r_path.write_text(render_q2r_input(prefix, zasr=zasr))
    matdyn_path.write_text(
        render_matdyn_dos_input(prefix, nk=nk, delta_e_cm1=delta_e_cm1, asr=zasr)
    )
    return q2r_path, matdyn_path


def parse_matdyn_dos(dos_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a matdyn.x DOS file (fldos): a one-line comment header, then two
    columns -- frequency (cm-1) and DOS (states per cm-1 per unit cell, QE's
    normalization) -- one row per deltaE-wide bin.

    Frequencies below QE's numerical-zero tolerance for the acoustic branch
    near Gamma print as small negative values; these are left in as-is
    (vdos_features.extract_features is what interprets negative frequencies
    as imaginary/unstable, using the same tolerance as
    unstable_modes.find_unstable_modes).
    """
    lines = Path(dos_path).read_text().splitlines()
    rows = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    freq_cm1 = np.array([float(row[0]) for row in rows])
    dos = np.array([float(row[1]) for row in rows])
    return freq_cm1, dos
