"""Turn a short model spec string into an ASE calculator -- the one place every
MLIP workflow in this project (phonon benchmark, pre-DFT relaxations,
soft-mode relaxations) gets its calculator from.

Spec format: `<backend>:<model>`, or a bare path to a `.model` file (MACE).

    models/MACE-matpes-r2scan-omat-ft.model       local MACE checkpoint
    mace:models/2023-12-10-mace-128-L0_energy_epoch-249.model
    mace:medium-mpa-0                             a named MACE foundation model
    models/mace-mh-1.model@omat_pbe               one head of a multi-head MACE model
    orb:orb_v3_conservative_inf_omat              an Orb pretrained model (orb-models)

Dispersion: every calculator is the MLIP plus DFT-D3(BJ) (src/mlip/dftd3.py).
The foundation models used here are trained on dispersion-free DFT (MPtrj,
OMat24, MatPES), so the correction is on by default, with the damping
parameters of the model's training functional inferred from its name (see
infer_xc). A suffix overrides that:

    <spec>+d3-<xc>    D3(BJ) with this functional's damping (pbe, pbesol, r2scan)
    <spec>+nod3       no correction -- only for a model whose training data
                      already includes dispersion, or to reproduce pre-D3 results

A spec whose functional can't be inferred, or that looks like it already has
dispersion in it, is an error until one of the two suffixes says what to do.
The output label (model_label) carries `-D3BJ-<xc>`, so D3 results never land
in, or get skipped because of, a directory of uncorrected ones.

To add another MLIP family, write a `_load_<name>(model, device)` function
returning an ASE calculator and add it to BACKENDS -- imports stay inside the
loader so an uninstalled package only fails when actually used.

Every loader should use float64 where the package allows it: finite-
displacement phonons difference forces at 0.01 A displacements, and float32
round-off is visible in the resulting low-frequency (acoustic/tilt) branches.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from ase.calculators.calculator import Calculator

from src.mlip.dftd3 import BJ_PARAMS


def _load_mace(model: str, device: str) -> Calculator:
    """`model` is a foundation-model name or checkpoint path, optionally
    suffixed `@<head>` for a multi-head model (e.g. mace-mh-1.model@omat_pbe):
    those have no default head, and the head picks the level of theory."""
    from mace.calculators import mace_mp

    path, _, head = model.partition("@")
    kwargs = {"head": head} if head else {}
    # cuequivariance kernels on GPU: without them MACE runs the e3nn tensor
    # products, whose memory OOMed a 40 GB A100 on ~2000-3200-atom supercells.
    # Opt-in in mace-torch, and it raises (not falls back) if the package is missing.
    if device.startswith("cuda"):
        kwargs["enable_cueq"] = True
    return mace_mp(model=path, device=device, default_dtype="float64", **kwargs)


def _load_orb(model: str, device: str) -> Calculator:
    """`model` is an orb_models.forcefield.pretrained loader name, with
    hyphens or underscores, e.g. orb_v3_conservative_inf_omat. Only the
    conservative (forces = -dE/dx) variants belong in a phonon benchmark: the
    'direct' ones predict forces without an energy, so their force constants
    aren't guaranteed to be symmetric."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    loader = getattr(pretrained, model.replace("-", "_"), None)
    if loader is None:
        raise ValueError(f"orb_models.forcefield.pretrained has no model {model!r}")
    orbff, adapter = loader(device=device, precision="float64")
    return ORBCalculator(orbff, adapter, device=device)


BACKENDS: dict[str, Callable[[str, str], Calculator]] = {
    "mace": _load_mace,
    "orb": _load_orb,
}


D3_SUFFIX = re.compile(r"\+(?:d3-([A-Za-z0-9]+)|(nod3))$", re.IGNORECASE)

# Checkpoints whose file name says nothing about the functional.
KNOWN_XC = {
    "2023-12-10-mace-128-l0_energy_epoch-249": "pbe",  # MACE-MP-0 medium (MPtrj, PBE/PBE+U)
}
# Dataset names that imply PBE when no functional is spelled out.
_PBE_DATASETS = {"mp", "mpa", "mptrj", "omat", "oam", "matpes"}
# Training sets / levels of theory that already contain dispersion.
_HAS_DISPERSION = {"d3", "d4", "vv10", "omol", "spice", "wb97m", "off"}


def _split_suffix(spec: str) -> tuple[str, str | None]:
    """(base spec, suffix choice): a functional name for `+d3-<xc>`, 'nod3'
    for `+nod3`, None when there is no suffix."""
    match = D3_SUFFIX.search(spec)
    if match is None:
        return spec, None
    return spec[: match.start()], (match.group(1) or match.group(2)).lower()


def parse_spec(spec: str) -> tuple[str, str]:
    """(backend, model) from a spec string (see module docstring)."""
    spec, _ = _split_suffix(spec)
    backend, sep, model = spec.partition(":")
    if sep and backend in BACKENDS:
        return backend, model
    if spec.partition("@")[0].endswith(".model"):
        return "mace", spec
    raise ValueError(
        f"Can't parse MLIP spec {spec!r}: expected '<backend>:<model>' with backend in "
        f"{sorted(BACKENDS)}, or a path to a MACE .model file"
    )


def infer_xc(spec: str) -> str:
    """The D3(BJ) functional for a spec without a suffix, from the model's
    name: the head of a multi-head model if given (it picks the level of
    theory), else the checkpoint/model name. An explicit functional in the
    name wins (r2scan, pbesol, pbe); otherwise a PBE dataset name (mp, mpa,
    omat, ...) or a named MACE foundation model (all MP-family PBE) gives pbe.
    """
    backend, model = parse_spec(spec)
    path, _, head = model.partition("@")
    is_file = path.endswith(".model")
    name = (head or (Path(path).stem if is_file else path)).lower()
    hint = f"append '+d3-<xc>' (one of {sorted(BJ_PARAMS)}) or '+nod3' to the spec"

    tokens = set(re.split(r"[^a-z0-9]+", name))
    if tokens & _HAS_DISPERSION:
        raise ValueError(f"MLIP spec {spec!r} looks like it already includes dispersion; {hint}")
    if name in KNOWN_XC:
        return KNOWN_XC[name]
    for xc in ("r2scan", "pbesol", "pbe"):
        if xc in name:
            return xc
    if tokens & _PBE_DATASETS or (backend == "mace" and not is_file and not head):
        return "pbe"
    raise ValueError(f"Can't infer the training functional of MLIP spec {spec!r} for its D3 correction; {hint}")


def dispersion_xc(spec: str) -> str | None:
    """The functional whose D3(BJ) damping this spec's calculator adds, or
    None if the spec opts out with `+nod3`."""
    _, choice = _split_suffix(spec)
    if choice == "nod3":
        return None
    xc = choice or infer_xc(spec)
    if xc not in BJ_PARAMS:
        raise ValueError(f"MLIP spec {spec!r}: no D3(BJ) parameters for {xc!r}; known: {sorted(BJ_PARAMS)}")
    return xc


def model_label(spec: str) -> str:
    """Filesystem-safe short name for a model, used as its output directory
    name and plot legend entry (e.g. 'MACE-matpes-r2scan-omat-ft-D3BJ-r2scan')."""
    backend, model = parse_spec(spec)
    path, _, head = model.partition("@")
    stem = Path(path).stem if path.endswith(".model") else f"{backend}-{path}"
    if head:
        stem = f"{stem}-{head}"
    d3_xc = dispersion_xc(spec)
    if d3_xc:
        stem = f"{stem}-D3BJ-{d3_xc}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def default_device() -> str:
    """'cuda' if torch sees a GPU, else 'cpu'."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_calculator(spec: str, device: str | None = None) -> Calculator:
    """The MLIP named by `spec` plus its D3(BJ) correction (see module
    docstring), on `device` (default: default_device())."""
    device = device or default_device()
    backend, model = parse_spec(spec)
    d3_xc = dispersion_xc(spec)  # before the (slow) model load: a bad spec fails fast
    path = model.partition("@")[0]
    if path.endswith(".model") and not Path(path).exists():
        raise FileNotFoundError(f"MLIP checkpoint not found: {path}")
    calc = BACKENDS[backend](model, device)
    if d3_xc is None:
        return calc
    from ase.calculators.mixing import SumCalculator

    from src.mlip.dftd3 import DFTD3Calculator

    return SumCalculator([calc, DFTD3Calculator(xc=d3_xc, device=device)])
