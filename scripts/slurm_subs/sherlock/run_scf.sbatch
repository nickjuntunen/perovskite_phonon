#!/bin/bash
#SBATCH --job-name=qe-scf
#SBATCH --partition=normal,owners
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# --nodes/--ntasks/--mem/--time above are sized for the common <=80-atom
# case. scripts/orchestrate_pipeline.py's scf_ph_resource_overrides() passes
# CLI overrides (which beat these #SBATCH defaults) for larger soft-mode-tilt
# supercells, spreading more MPI ranks over more nodes rather than asking for
# a single bigmem node -- see its docstring for why (NaCaCl3_tilted's OOM).

# Runs the pw.x SCF calculation for one structure from
# $SCRATCH/perovskite_phonon/qe_inputs/.
# Usage: sbatch --export=ALL,STRUCTURE=CsPbI3 run_scf.sbatch
# (submit_all.sh does this for you, including chaining to run_ph.sbatch)
#
# Listing multiple partitions lets Slurm start the job on whichever of them
# frees up first; --time is kept <= 2 days so it stays valid on all of them
# (owners caps at 2 days even though normal allows up to 7). "rotskoff" is
# deliberately excluded (small owner partition -- shouldn't be monopolized by
# this batch); "hns" is left out too since submitting with both rotskoff and
# hns listed together gets rejected with reason=BadConstraints (verified
# against this account) -- a scheduler-side incompatibility between those two
# partitions, not a resource-request problem.

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

SRC_INPUT_DIR="${SCRATCH}/perovskite_phonon/qe_inputs/${STRUCTURE}"
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
