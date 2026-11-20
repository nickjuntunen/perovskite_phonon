#!/bin/bash
#SBATCH --job-name=gp-disorder
#SBATCH --partition=normal,owners,hns
#SBATCH --array=0-8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/%x-%A_%a.err

# gp_vs_tree/gp_disorder.py, one target per array task (its TARGETS list, 9):
# GP and trees with / without X-site disorder features and with the bowing-only
# target, scored on held-out halide pairs and held-out series. CPU only. Each
# (variant, scheme, fold) is checkpointed under gp_vs_tree/disorder/parts/, so
# a preempted (owners, --requeue) task resumes where it stopped.
#
# Environment: the analysis venv, not the MLIP one -- scikit-learn, mendeleev,
# pymatgen, pyrovskite on Sherlock's python/3.12.1 module, at
#   $GROUP_HOME/nick/python_venvs/perovskite_analysis
# Inputs (git-ignored, copied from the local tree): data/mlip_stable_polymorphs/,
# data/mp_perovskites/composition_features.csv, data/mixed_halides/analysis/.
#
# Usage:
#   sbatch --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon run_gp_disorder.sbatch
#   # then, after every task: the summary table
#   sbatch --dependency=afterok:<array id> --time=00:10:00 --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon \
#       --wrap 'ml python/3.12.1 && source $GROUP_HOME/nick/python_venvs/perovskite_analysis/bin/activate && \
#               python $PROJECT_ROOT/gp_vs_tree/gp_disorder.py --summarize'
#   # refit on SQS-ensemble labels (needs data/mixed_halides_ensemble/analysis/ensemble_targets.csv copied over;
#   # summarize with --out gp_vs_tree/disorder_ens):
#   sbatch --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon,GP_ARGS="--labels ensemble --variants base" \
#       run_gp_disorder.sbatch

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit as an array job (--array)}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,PROJECT_ROOT=\$HOME/perovskite_phonon}"

ml python/3.12.1
# shellcheck source=/dev/null
source "${GROUP_HOME}/nick/python_venvs/perovskite_analysis/bin/activate"
# One BLAS thread per fit: joblib runs SLURM_CPUS_PER_TASK fits side by side.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

python -u "${PROJECT_ROOT}/gp_vs_tree/gp_disorder.py" \
    --task-index "${SLURM_ARRAY_TASK_ID}" --jobs "${SLURM_CPUS_PER_TASK}" ${GP_ARGS:-}
