#!/bin/bash
#SBATCH --job-name=install-d3
#SBATCH --partition=owners,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.err
# One-off: install the DFT-D3(BJ) kernels (nvalchemi-toolkit-ops, pinned to the
# local versions) into the project venv, build the D3 reference-table cache
# (~/.cache/nvalchemiops/dftd3_parameters.pt), and check the D3 energy of cubic
# CsPbBr3 against the local value (-0.40713 eV, r2scan BJ).
# Ran once (job 52858500, 2026-11-18). The check structure is a scratch file,
# $SCRATCH/perovskite_phonon/jobs/CsPbBr3.cif (cubic, the local relaxed cell).
set -eo pipefail
P=$HOME/perovskite_phonon
# sourceme.sh isn't written for strict mode (a find that exits nonzero on an absent optional dir)
set +eo pipefail
source $P/sourceme.sh
set -eo pipefail
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
apptainer exec "$SIF_PATH" "$VENV_PATH/bin/python" -m pip install "nvalchemi-toolkit-ops==0.3.1" "warp-lang==1.17.0"
cd $P
apptainer-python - <<'PY'
import sys; sys.path.insert(0, ".")
import ase.io, nvalchemiops, warp
from importlib.metadata import version
print("nvalchemi-toolkit-ops", version("nvalchemi-toolkit-ops"), "warp", warp.__version__)
from src.mlip.dftd3 import PARAM_CACHE, build_parameters, DFTD3Calculator
if not PARAM_CACHE.exists():
    build_parameters()
print("cache", PARAM_CACHE, PARAM_CACHE.stat().st_size, "bytes")
import os
a = ase.io.read(os.path.expandvars("$SCRATCH/perovskite_phonon/jobs/CsPbBr3.cif"))
a.calc = DFTD3Calculator(xc="r2scan", device="cpu")
e = a.get_potential_energy()
print(f"E_D3(CsPbBr3) = {e:.6f} eV (local -0.407128)")
assert abs(e + 0.4071278) < 1e-4, "D3 energy differs from the local value"
print("D3 INSTALL OK")
PY
