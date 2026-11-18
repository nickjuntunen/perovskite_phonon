#!/bin/bash
#SBATCH --job-name=mlip-phonons
#SBATCH --partition=gpu,owners,rotskoff
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.err

# Finite-displacement phonons (dispersion + vDOS) with one or more MLIPs, for
# every completed DFPT run under ph_first/ -- see scripts/mlip/run_mlip_phonons.py.
# Afterwards, sync ${SCRATCH}/perovskite_phonon/mlip_phonons back and run
# scripts/mlip/compare_phonons.py against the synced ph_first/.
#
# Each model runs with DFT-D3(BJ) for its training functional and writes to
# <model>-D3BJ-<xc>/ (src/mlip/mlip_calculators.py; +d3-<xc> / +nod3 on a
# spec override that).
#
# Reuses the pytorch/mace apptainer container + venv from mat_tmmc (see
# run_mace_relax.sbatch). That venv also needs phonopy and the D3 kernels, once:
#   apptainer exec ${SIF_PATH} ${VENV_PATH}/bin/python -m pip install phonopy nvalchemi-toolkit-ops
#
# Usage:
#   sbatch --export=ALL,PROJECT_ROOT=$(pwd)/../../..,MODELS="/path/a.model /path/b.model" run_mlip_phonons.sbatch
#   optionally STRUCTURES="CsPbBr3 KNbO3" (default: every run with a .fc file)
#   and EXTRA_ARGS="--relax full" (any other run_mlip_phonons.py flags)

set -euo pipefail
: "${SCRATCH:?\$SCRATCH is not set}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this
# script on the compute node -- must come in from the submitting shell.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,PROJECT_ROOT=\$(pwd)/../../..}"
: "${MODELS:?Set MODELS to one or more MLIP specs, see src/mlip/mlip_calculators.py}"

SIF_PATH="${GROUP_HOME}/nick/pytorch_2.11.0-cuda12.8-cudnn9-runtime.sif"
VENV_PATH="${HOME}/mat_tmmc/.venv"

# Word-splitting of MODELS/STRUCTURES/EXTRA_ARGS into separate arguments is intended.
# shellcheck disable=SC2086
apptainer exec --nv "${SIF_PATH}" "${VENV_PATH}/bin/python" \
    "${PROJECT_ROOT}/scripts/mlip/run_mlip_phonons.py" ${STRUCTURES:-} \
    --qe-ph-root "${SCRATCH}/perovskite_phonon/ph_first" \
    --out-root "${SCRATCH}/perovskite_phonon/mlip_phonons" \
    --models ${MODELS} \
    --device cuda \
    ${EXTRA_ARGS:-}
