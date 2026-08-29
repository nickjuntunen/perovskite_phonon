#!/bin/bash
#SBATCH --job-name=mace-relax
#SBATCH --partition=normal,owners,rotskoff
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Relaxes one structure with a MACE-MPA-0 foundation potential (CPU-only --
# a few-atom cell relaxation doesn't need a GPU) before handing it to
# pw.x/ph.x. See scripts/relax_mace.py for why.
#
# Reuses the pytorch/mace apptainer container + venv already built for the
# mat_tmmc project (has ase/mace/torch installed) -- nothing under
# ~/mat_tmmc is read-write here, only its .sif and .venv are referenced.
#
# Usage: sbatch --export=ALL,STRUCTURE=CsPbI3 run_mace_relax.sbatch
# (submit_all.sh does this for you, chaining into run_scf.sbatch)

set -euo pipefail
: "${STRUCTURE:?Set STRUCTURE, e.g. sbatch --export=ALL,STRUCTURE=CsPbI3 run_mace_relax.sbatch}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this
# script on the compute node, so BASH_SOURCE there points into Slurm's spool
# dir, not the repo -- must come in from the submitting shell instead.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,STRUCTURE=CsPbI3,PROJECT_ROOT=\$(pwd)/../../..}"

SIF_PATH="${GROUP_HOME}/nick/pytorch_2.11.0-cuda12.8-cudnn9-runtime.sif"
VENV_PATH="${HOME}/mat_tmmc/.venv"

apptainer exec "${SIF_PATH}" "${VENV_PATH}/bin/python" \
    "${PROJECT_ROOT}/scripts/relax_mace.py" "${STRUCTURE}" --fmax 0.01
