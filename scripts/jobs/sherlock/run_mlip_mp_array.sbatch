#!/bin/bash
#SBATCH --job-name=mlip-mp-phonons
#SBATCH --partition=owners,hns,gpu
#SBATCH --array=0-98
#SBATCH --gpus=1
#SBATCH --constraint=GPU_CC:8.0|GPU_CC:9.0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/%x-%A_%a.err

# Logs: #SBATCH lines can't expand $SCRATCH, hence the literal path. Slurm
# doesn't create it -- a missing directory fails the job with no log at all:
#   mkdir -p $SCRATCH/perovskite_phonon/logs

# The full MLIP phonon pipeline (relax -> dispersion + vDOS -> rattle
# retry -> soft-mode freeze-in, see scripts/mlip/run_mlip_phonons.py) over
# every verified, near-hull MP perovskite (metadata_stable.csv; see
# scripts/dataset/write_mp_structure_list.py -- the list no longer depends on
# the composition clustering), one MLIP. For only the structures a previous
# sweep hasn't finished, pass STRUCTURE_LIST=.../mlip_structures_new.txt
# (write_mp_structure_list.py --exclude-done).
#
# Why ~99 tasks of ~28 structures each, not one task per structure (2759):
# EnforcePartLimits=ALL makes a job satisfy every listed partition's limits
# at submit time, and the gpu QOS caps a user at 100 submitted jobs (each
# array task counts as one). `sbatch --test-only` with gpu listed accepts
# --array=0-99 and rejects 0-999; separately, max_array_tasks=1000 caps any
# single array. Tasks stride through the list (task i takes lines i, i+99,
# ...), which write_mp_structure_list.py sorts by cell size, so each slice
# gets the same cost mix. For finer-grained parallelism drop gpu:
#   sbatch -p owners,hns --array=0-919 --export=ALL,...,NUM_TASKS=920 ...
# (~3 structures per task; owners allows 3000 submitted jobs, and Sherlock
# also caps submissions at 1000 jobs/user/hour).
#
# Resumable: a structure with a meta.json is skipped, so preemption on owners
# (PreemptMode=REQUEUE, hence --requeue) or a 24 h timeout loses at most the
# structure in progress -- resubmit the same array (same NUM_TASKS) to
# finish. Failed structures are listed at the end of each task's .out.
#
# --constraint: the gauntlet runs the MLIP in float64 (src/mlip/mlip_calculators.py),
# and only the data-center GPUs have full-rate FP64 -- compute capability
# 8.0 (A100), 9.0 (H100/H200). The L40S/A40/RTX parts (8.6/8.9/7.5) run FP64
# at 1/64 rate. 7.0 (V100, TITAN V) also has full-rate FP64, but the
# container's torch 2.11+cu128 ships no sm_70 kernels: job 46355821's V100
# tasks all died with "no kernel image is available"; cuequivariance needs
# >= 8.0 too. 32G host memory covers the largest supercells in the set
# (~1600-3200 atoms after even rounding); on a 40 GB A100 those few may still
# OOM on the GPU -- they'll show up as failures (rerun on 80 GB cards).
# Organic cations need --mem=64G: their N-H/C-H stretches make the vDOS
# tetrahedron weights ~34 GiB (job 52574151).
#
# Dispersion: the MLIP runs with DFT-D3(BJ) for its training functional
# (src/mlip/mlip_calculators.py), so results land in
# ${OUT_ROOT}/<model>-D3BJ-<xc>/ -- next to, not on top of, a pre-D3 sweep's
# <model>/. Append +d3-<xc> to MODEL to pick the functional, +nod3 to turn it
# off. The D3 reference tables are downloaded once to
# ~/.cache/nvalchemiops/ (src/mlip/dftd3.py).
#
# Environment: ${PROJECT_ROOT}/sourceme.sh (not in git) -- the CUDA container,
# the venv at $GROUP_HOME/nick/python_venvs/perovskite_phonon (mace-torch,
# nvalchemi-toolkit-ops, phonopy, spglib), and apptainer-python, which also sets the LD_LIBRARY_PATH
# cuequivariance needs inside the container.
#
# Data: data/mp_perovskites/structures/ and mlip_structures.txt
# (scripts/dataset/write_mp_structure_list.py), in the same directory.
#
# Usage:
#   sbatch --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon run_mlip_mp_array.sbatch
#   optionally STRUCTURE_LIST=..., MODEL=..., OUT_ROOT=..., NUM_TASKS=... (must match --array's
#   size), EXTRA_ARGS="..." (other run_mlip_phonons.py flags)
#   rerun specific tasks with: sbatch --array=5,17 ... (same NUM_TASKS)

set -euo pipefail
: "${SCRATCH:?\$SCRATCH is not set}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array job (--array)}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this
# script on the compute node -- must come in from the submitting shell.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,PROJECT_ROOT=\$HOME/perovskite_phonon}"
STRUCTURE_LIST="${STRUCTURE_LIST:-${PROJECT_ROOT}/data/mp_perovskites/mlip_structures.txt}"

MODEL="${MODEL:-/home/groups/rotskoff/nick/MACE-matpes-r2scan-omat-ft.model}"
NUM_TASKS="${NUM_TASKS:-99}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH}/perovskite_phonon/mlip_phonons_mp}"

# sourceme.sh isn't written for strict mode: unset variables, and $(find ...)
# assignments that exit nonzero when an optional library dir is absent.
set +euo pipefail
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/sourceme.sh"
set -euo pipefail

# The 40 GB A100 OOMs in job 46355821 had ~10-13 GiB reserved but unallocated
# (fragmentation). apptainer passes the host environment into the container.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail here, once per task, rather than as a "could not load model" buried in the log.
# The checkpoint path is MODEL minus any +d3-<xc>/+nod3 and @<head> suffix.
MODEL_PATH="${MODEL%+d3-*}"; MODEL_PATH="${MODEL_PATH%+nod3}"; MODEL_PATH="${MODEL_PATH%@*}"
[[ -f "${MODEL_PATH}" ]] || { echo "MLIP checkpoint not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${STRUCTURE_LIST}" ]] || { echo "structure list not found: ${STRUCTURE_LIST}" >&2; exit 1; }
if (( SLURM_ARRAY_TASK_ID >= NUM_TASKS )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} >= NUM_TASKS=${NUM_TASKS}: make --array and NUM_TASKS agree" >&2
    exit 1
fi

# -u: unbuffered, so the per-structure progress lines show up in the .out as they happen.
# shellcheck disable=SC2086  # word-splitting EXTRA_ARGS into separate flags is intended
apptainer-python -u \
    "${PROJECT_ROOT}/scripts/mlip/run_mlip_phonons.py" \
    --structure-list "${STRUCTURE_LIST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" --num-tasks "${NUM_TASKS}" \
    --models "${MODEL}" \
    --relax full \
    --out-root "${OUT_ROOT}" \
    --device cuda \
    ${EXTRA_ARGS:-}
