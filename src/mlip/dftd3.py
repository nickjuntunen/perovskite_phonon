"""DFT-D3(BJ) dispersion as an ASE calculator, on NVIDIA ALCHEMI's GPU kernels
(nvalchemi-toolkit-ops), to add to an MLIP trained on dispersion-free DFT
(MatPES r2SCAN/PBE, OMat24 PBE). mlip_calculators.load_calculator adds it to
every MLIP by default, picking the functional from the model's name or from a
`+d3-<xc>` spec suffix, e.g. models/MACE-matpes-r2scan-omat-ft.model+d3-r2scan.

nvalchemiops ships the kernels but not the D3 reference tables. Those are
parsed here from Grimme's reference implementation (dftd3.tgz: C6 grid in
pars.f, covalent radii and <r4>/<r2> in dftd3.f), pinned by MD5, and cached
at ~/.cache/nvalchemiops/dftd3_parameters.pt -- the file and layout
nvalchemiops' own examples use, so either can build it for the other.

Two-body C6 + C8 terms with Becke-Johnson damping, no three-body (ATM) term,
like mace_mp(dispersion=True). The kernels work in atomic units and always
return float32 -- harmless for finite-displacement phonons: summed D3 force
round-off is ~1e-8 eV/A against ~1e-3 eV/A force differences at 0.01 A.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
from hashlib import md5
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

DFTD3_TGZ_URL = "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz"
DFTD3_TGZ_MD5 = "a76c752e587422c239c99109547516d2"
PARAM_CACHE = Path.home() / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"

# D3(BJ) damping parameters (s6, a1, s8, a2 / Bohr), as in simple-dftd3's
# parameters.toml. Pick the functional the MLIP's training data used.
BJ_PARAMS: dict[str, dict[str, float]] = {
    "pbe": {"s6": 1.0, "a1": 0.4289, "s8": 0.7875, "a2": 4.4407},
    "pbesol": {"s6": 1.0, "a1": 0.4466, "s8": 2.9491, "a2": 6.1742},
    "r2scan": {"s6": 1.0, "a1": 0.49484001, "s8": 0.78981345, "a2": 5.73083694},
}

# 40 Bohr (21.2 A): torch-dftd's default as used by mace_mp(dispersion=True).
# The C6/r^6 tail beyond it is ~1e-4 eV/atom and cancels in crystal forces.
DEFAULT_CUTOFF_BOHR = 40.0


def _fortran_data_array(source: str, name: str) -> np.ndarray:
    """Values of a `data <name> / ... /` statement in fixed-form Fortran."""
    code = "\n".join(line.split("!")[0] for line in source.splitlines() if not re.match(r"[cC!*]", line[:1]))
    match = re.search(rf"data\s+{name}\s*/(.*?)/", code, re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError(f"no 'data {name}' block in dftd3.f")
    return np.array([float(v.replace("_wp", "")) for v in re.findall(r"[-+]?\d+\.\d+(?:_wp)?", match.group(1))])


def _c6_grid(pars_source: str) -> tuple[np.ndarray, np.ndarray]:
    """(c6ab, cn_ref), each [95, 95, 5, 5], from pars.f's flat records of
    (C6, Z_i', Z_j', CN_i, CN_j), where Z' = Z + 100 * (reference index - 1).
    Unset cn_ref entries are -1 (no such reference), as nvalchemiops expects."""
    code = "\n".join(line.split("!")[0] for line in pars_source.splitlines())
    blocks = re.findall(r"pars\(\s*\d+\s*:\s*\d+\s*\)\s*=\s*\(/(.*?)/\)", code, re.DOTALL)
    if not blocks:
        raise ValueError("no pars(...)=(/ ... /) blocks in pars.f")
    values = [float(v.replace("D", "e").replace("d", "e"))
              for block in blocks for v in re.findall(r"[-+]?\d+\.\d+[eEdD][-+]?\d+", block)]
    records = np.array(values).reshape(-1, 5)

    c6ab = np.zeros((95, 95, 5, 5))
    cn_ref = np.full((95, 95, 5, 5), -1.0)
    cn_of = {}  # (Z, reference index) -> CN
    for c6, zi, zj, cni, cnj in records:
        iref, iat = divmod(int(zi) - 1, 100)
        jref, jat = divmod(int(zj) - 1, 100)
        iat, jat = iat + 1, jat + 1
        c6ab[iat, jat, iref, jref] = c6ab[jat, iat, jref, iref] = c6
        cn_of.setdefault((iat, iref), cni)
        cn_of.setdefault((jat, jref), cnj)
    for (z, ref), cn in cn_of.items():
        cn_ref[z, 1:, ref, :] = cn
    return c6ab, cn_ref


def build_parameters(cache: Path = PARAM_CACHE) -> Path:
    """Download dftd3.tgz, check its MD5, parse the reference tables and save
    them (float64, atomic units, index 0 = padding) to `cache`."""
    import requests
    import torch

    response = requests.get(DFTD3_TGZ_URL, timeout=60)
    response.raise_for_status()
    digest = md5(response.content, usedforsecurity=False).hexdigest()
    if digest != DFTD3_TGZ_MD5:
        raise ValueError(f"{DFTD3_TGZ_URL}: MD5 {digest}, expected {DFTD3_TGZ_MD5}")
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and Path(member.name).name in ("dftd3.f", "pars.f"):
                sources[Path(member.name).name] = tar.extractfile(member).read().decode("utf-8", errors="ignore")

    rcov, r4r2 = np.zeros(95), np.zeros(95)
    rcov[1:] = _fortran_data_array(sources["dftd3.f"], "rcov")  # already scaled by k2 = 4/3, Bohr
    r4r2[1:] = _fortran_data_array(sources["dftd3.f"], "r2r4")
    c6ab, cn_ref = _c6_grid(sources["pars.f"])

    # Written to a temporary name and renamed: with D3 on for every model, the
    # tasks of a Slurm array can all reach this at once on a fresh cache, and a
    # reader must never see a half-written file.
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    torch.save({k: torch.from_numpy(v) for k, v in
                {"rcov": rcov, "r4r2": r4r2, "c6ab": c6ab, "cn_ref": cn_ref}.items()}, partial)
    os.replace(partial, cache)
    return cache


class DFTD3Calculator(Calculator):
    """D3(BJ) energy, forces and stress for periodic ASE Atoms."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, xc: str = "pbe", cutoff_bohr: float = DEFAULT_CUTOFF_BOHR, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        import torch
        from nvalchemiops.torch.interactions.dispersion import D3Parameters

        if xc not in BJ_PARAMS:
            raise ValueError(f"no D3(BJ) parameters for xc={xc!r}; known: {sorted(BJ_PARAMS)}")
        if not PARAM_CACHE.exists():
            build_parameters()
        tables = torch.load(PARAM_CACHE, map_location="cpu", weights_only=True)
        self.d3_params = D3Parameters(**tables).to(device=device, dtype=torch.float64)
        self.xc = xc
        self.damping = BJ_PARAMS[xc]
        self.cutoff_bohr = cutoff_bohr
        self.device = device

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        import torch
        from nvalchemiops.torch.interactions.dispersion import dftd3
        from nvalchemiops.torch.neighbors import neighbor_list

        super().calculate(atoms, properties, system_changes)
        if not self.atoms.pbc.all():
            raise ValueError("DFTD3Calculator handles fully periodic cells only")
        kw = {"dtype": torch.float64, "device": self.device}
        positions = torch.tensor(self.atoms.positions / Bohr, **kw)
        cell = torch.tensor(self.atoms.cell.array / Bohr, **kw).unsqueeze(0)
        numbers = torch.tensor(self.atoms.numbers, dtype=torch.int32, device=self.device)
        n = len(self.atoms)

        nbr, _, shifts = neighbor_list(
            positions, self.cutoff_bohr, cell=cell,
            pbc=torch.ones(1, 3, dtype=torch.bool, device=self.device), fill_value=n,
        )
        energy, forces, _, virial = dftd3(
            positions=positions, numbers=numbers, d3_params=self.d3_params, **self.damping,
            cell=cell, neighbor_matrix=nbr, neighbor_matrix_shifts=shifts, fill_value=n,
            compute_virial=True, num_systems=1,
        )
        energy = float(energy[0]) * Hartree
        # virial W = -sum_ij r_ij (x) F_ij  =>  stress = -W / V (ASE sign: tensile positive)
        stress = -virial[0].double().cpu().numpy() * Hartree / self.atoms.get_volume()
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces.double().cpu().numpy() * (Hartree / Bohr),
            "stress": stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]],
        }
