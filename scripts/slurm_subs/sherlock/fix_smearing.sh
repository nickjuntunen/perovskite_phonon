#!/bin/bash
#SBATCH --job-name=fix-smearing
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/$x-$j.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/$x-$j.err
#SBATCH --partition=normal,owners,rotskoff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Runs scripts/fix_smearing_crashes.py: scans a scf_first/ tree for QE's
# "charge is wrong: smearing is needed" error and regenerates just the
# affected structures' SCF+DFPT inputs with occupations='smearing' (see
# src/scf_diagnostics.py for why this shows up and what it does/doesn't fix).
# This is a lightweight single-CPU Python job -- no QE module, no MPI -- so
# the resource request here is intentionally small, unlike run_scf.sbatch/
# run_ph.sbatch. "hns" is left off the partition list for the same reason as
# those two scripts: submitting with both rotskoff and hns listed together
# gets rejected with reason=BadConstraints on this account.
#
# Usage (auto-detect from existing SCF logs -- the common case; defaults
# SCF_DIR to $SCRATCH/perovskite_phonon/scf_first):
#   PROJECT_ROOT="$root_directory" sbatch --export=ALL run_fix_smearing.sbatch
#   PROJECT_ROOT="$root_directory" SCF_DIR="/other/scf_first" sbatch --export=ALL run_fix_smearing.sbatch
#
# Usage (fix an explicit list instead of scanning):
#   sbatch --export=ALL,PROJECT_ROOT=$(pwd),NAMES="CsCoBr3 CsCuCl3 CsEuF3" run_fix_smearing.sbatch
#
# Dry run first (recommended before touching a large batch at once):
#   sbatch --export=ALL,PROJECT_ROOT=$(pwd),DRY_RUN=1 run_fix_smearing.sbatch
#
# Optional overrides: DEGAUSS (default 0.01), SMEARING (default gaussian).

ml devel gcc/14.2.0

set -euo pipefail
export PROJECT_ROOT=/home/users/juntunen/perovskite_phonon

NAMES="${NAMES:-}"
SCF_DIR="${SCF_DIR:-}"
if [[ -z "${SCF_DIR}" && -n "${SCRATCH:-}" ]]; then
    SCF_DIR="${SCRATCH}/perovskite_phonon/scf_first"
fi
DEGAUSS="${DEGAUSS:-0.01}"
SMEARING="${SMEARING:-gaussian}"
DRY_RUN="${DRY_RUN:-0}"
METALLIC="${METALLIC:-0}"

if [[ -z "${NAMES}" && -z "${SCF_DIR}" ]]; then
    echo "Set NAMES=\"<space-separated structure names>\", or SCF_DIR=<path> (or SCRATCH, which SCF_DIR defaults from)." >&2
    exit 1
fi

ARGS=(--degauss "${DEGAUSS}" --smearing "${SMEARING}")
if [[ -n "${NAMES}" ]]; then
    # Intentionally unquoted: NAMES is a space-separated list and --names
    # takes multiple arguments, same pattern submit_all.sh uses for its own
    # space-separated structure list.
    ARGS+=(--names ${NAMES})
else
    ARGS+=(--scf-dir "${SCF_DIR}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    ARGS+=(--dry-run)
fi
if [[ "${METALLIC}" == "1" ]]; then
    ARGS+=(--metallic)
fi

cd $PROJECT_ROOT
VENV_PATH="${GROUP_HOME}/nick/uv_venvs/perovskite_phonon"
# fix_smearing_crashes.py only needs ase + jinja2 -- no QE module load needed.

"${VENV_PATH}/bin/python" -m \
    "scripts.fix_smearing_crashes" "${ARGS[@]}"

