#!/bin/bash
#SBATCH --job-name=qe-ph
#SBATCH --partition=normal,owners
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# --nodes/--ntasks/--mem/--time above are sized for the common <=80-atom
# case. scripts/orchestrate_pipeline.py's scf_ph_resource_overrides() passes
# CLI overrides (which beat these #SBATCH defaults) for larger soft-mode-tilt
# supercells, spreading more MPI ranks over more nodes rather than asking for
# a single bigmem node -- see its docstring for why (NaCaCl3_tilted's OOM).

# Runs the ph.x DFPT phonon calculation for one structure from
# $SCRATCH/perovskite_phonon/qe_inputs/, then post-processes it with q2r.x +
# matdyn.x to get an interpolated phonon dispersion along the cubic-perovskite
# G-X-M-G-R-X path (all structures here are the same 5-atom ibrav=0
# simple-cubic ABX3 cell on a 4x4x4 q-grid, so one fixed path in crystal
# coordinates applies to every structure regardless of lattice constant).
# Must run after run_scf.sbatch has completed for the same structure (ph.x
# reads the SCF run's wavefunctions via matching prefix/outdir).
#
# Usage: sbatch --export=ALL,STRUCTURE=CsPbI3 run_ph.sbatch
# (submit_all.sh does this for you, with a --dependency on the SCF job)
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
: "${STRUCTURE:?Set STRUCTURE, e.g. sbatch --export=ALL,STRUCTURE=CsPbI3 run_ph.sbatch}"
: "${SCRATCH:?\$SCRATCH is not set}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this
# script on the compute node, so BASH_SOURCE there points into Slurm's spool
# dir, not the repo -- must come in from the submitting shell instead.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,STRUCTURE=CsPbI3,PROJECT_ROOT=\$(pwd)/../../..}"

ml purge
ml chemistry
ml quantum-espresso/7.1

SRC_INPUT_DIR="${SCRATCH}/perovskite_phonon/qe_inputs/${STRUCTURE}"
SCF_OUTDIR="${SCRATCH}/perovskite_phonon/scf_first/${STRUCTURE}/out"
RUN_DIR="${SCRATCH}/perovskite_phonon/ph_first/${STRUCTURE}"

if [[ ! -d "${SCF_OUTDIR}" ]]; then
    echo "SCF outdir ${SCF_OUTDIR} not found -- run run_scf.sbatch for ${STRUCTURE} first" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}"

# outdir must point back at the SCF run's save directory so ph.x can read its
# wavefunctions; fildyn/ph.out stay in this run's own ph_first directory.
sed -e "s|outdir *= *'.*'|outdir      = '${SCF_OUTDIR}'|" \
    "${SRC_INPUT_DIR}/${STRUCTURE}.ph.in" > "${RUN_DIR}/${STRUCTURE}.ph.in"

cd "${RUN_DIR}"
# ph.x's MPI ranks sometimes exit via Fortran STOP 1, which srun reports as
# exit code 1 even though the calculation completed normally -- with set -e
# that aborted the script before q2r.x/matdyn.x ran, discarding a good result.
# Tolerate the exit code here and check ph.out's own status markers instead.
# Note "JOB DONE" alone is not sufficient: ph.x also prints it after a
# self-consistency divergence ("No convergence has been achieved"), so check
# for that failure explicitly too.
srun ph.x -in "${STRUCTURE}.ph.in" > "${STRUCTURE}.ph.out" || true
if ! grep -q "JOB DONE" "${STRUCTURE}.ph.out"; then
    echo "ph.x did not complete (no JOB DONE marker in ${STRUCTURE}.ph.out)" >&2
    exit 1
fi
if grep -q "No convergence has been achieved" "${STRUCTURE}.ph.out"; then
    echo "ph.x did not converge (see ${STRUCTURE}.ph.out)" >&2
    exit 1
fi

# q2r.x: real-space force constants from the dyn0..dynN files ph.x just wrote.
cat > q2r.in <<EOF
&input
  fildyn = '${STRUCTURE}.dyn',
  zasr = 'simple',
  flfrc = '${STRUCTURE}.fc'
/
EOF
q2r.x < q2r.in > q2r.out

# matdyn.x: interpolate onto the G-X-M-G-R-X path. amass values are pulled
# from this run's own ph.in so they always match the ph.x calculation.
AMASS_LINES=$(grep -E "amass\([0-9]+\)" "${STRUCTURE}.ph.in")
cat > matdyn.in <<EOF
&input
  asr = 'simple',
  ${AMASS_LINES}
  flfrc = '${STRUCTURE}.fc',
  flfrq = '${STRUCTURE}.freq',
  q_in_band_form = .true.,
  q_in_cryst_coord = .true.,
/
6
0.0 0.0 0.0   20
0.0 0.0 0.5   20
0.5 0.5 0.0   20
0.0 0.0 0.0   20
0.5 0.5 0.5   20
0.0 0.0 0.5   0
EOF
matdyn.x < matdyn.in > matdyn.out
