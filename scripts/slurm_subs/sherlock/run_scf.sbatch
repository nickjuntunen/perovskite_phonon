#!/bin/bash
#SBATCH --job-name=qe-scf
#SBATCH --partition=normal,owners,rotskoff
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Runs the pw.x SCF calculation for one structure from data/qe_inputs/.
# Usage: sbatch --export=ALL,STRUCTURE=CsPbI3 run_scf.sbatch
# (submit_all.sh does this for you, including chaining to run_ph.sbatch)
#
# Listing multiple partitions lets Slurm start the job on whichever of them
# frees up first; --time is kept <= 2 days so it stays valid on all of them
# (owners caps at 2 days even though the others allow up to 7). "hns" is
# deliberately left out: submitting with both rotskoff and hns listed together
# gets rejected with reason=BadConstraints (verified against this account) --
# a scheduler-side incompatibility between those two partitions, not a
# resource-request problem. Submit standalone to hns manually if needed:
#   sbatch --partition=hns --export=ALL,STRUCTURE=<name> run_scf.sbatch

set -euo pipefail
: "${STRUCTURE:?Set STRUCTURE, e.g. sbatch --export=ALL,STRUCTURE=CsPbI3 run_scf.sbatch}"
: "${SCRATCH:?\$SCRATCH is not set}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this
# script on the compute node, so BASH_SOURCE there points into Slurm's spool
# dir, not the repo -- must come in from the submitting shell instead.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,STRUCTURE=CsPbI3,PROJECT_ROOT=\$(pwd)/../../..}"

ml purge
ml chemistry
ml quantum-espresso/7.1

SRC_INPUT_DIR="${PROJECT_ROOT}/data/qe_inputs/${STRUCTURE}"
PSEUDO_DIR="${PROJECT_ROOT}/data/pseudopotentials/pbesol_library"
RUN_DIR="${SCRATCH}/perovskite_phonon/scf_first/${STRUCTURE}"

mkdir -p "${RUN_DIR}/out"

# Rendered inputs may carry a pseudo_dir/outdir from wherever they were
# generated -- rewrite both to this run's actual paths on $SCRATCH.
sed -e "s|pseudo_dir *= *'.*'|pseudo_dir  = '${PSEUDO_DIR}'|" \
    -e "s|outdir *= *'.*'|outdir      = './out'|" \
    "${SRC_INPUT_DIR}/${STRUCTURE}.scf.in" > "${RUN_DIR}/${STRUCTURE}.scf.in"

cd "${RUN_DIR}"
srun pw.x -in "${STRUCTURE}.scf.in" > "${STRUCTURE}.scf.out"
