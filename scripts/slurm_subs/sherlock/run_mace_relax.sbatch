#!/bin/bash
#SBATCH --job-name=mace-relax
#SBATCH --partition=normal,owners
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Relaxes one structure with a MACE-MPA-0 foundation potential (CPU-only --
# a few-atom cell relaxation doesn't need a GPU) before handing it to
# pw.x/ph.x. See scripts/relax_mace.py for why.
#
# Uses this project's own dedicated venv (built via
# scripts/slurm_subs/sherlock/setup_venv.sh), which has ase/mace/torch
# installed -- not ~/mat_tmmc's (that project's .venv was never actually
# built on this cluster).
#
# Usage: sbatch --export=ALL,STRUCTURE=CsPbI3 run_mace_relax.sbatch
# (submit_all.sh does this for you, chaining into run_scf.sbatch)

set -euo pipefail
: "${STRUCTURE:?Set STRUCTURE, e.g. sbatch --export=ALL,STRUCTURE=CsPbI3 run_mace_relax.sbatch}"
: "${GROUP_HOME:?\$GROUP_HOME is not set}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this
# script on the compute node, so BASH_SOURCE there points into Slurm's spool
# dir, not the repo -- must come in from the submitting shell instead.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,STRUCTURE=CsPbI3,PROJECT_ROOT=\$(pwd)/../../..}"

VENV_PATH="${GROUP_HOME}/nick/uv_venvs/perovskite_phonon"

# The venv's portable Python (python-build-standalone) has OpenSSL compiled
# with Debian-style default cert paths (/etc/ssl/cert.pem) that don't exist on
# this RHEL-based system, so HTTPS (mace_mp's foundation-model download/cache
# check) fails cert verification unless pointed at the real cert bundle.
export SSL_CERT_FILE=/etc/pki/tls/cert.pem

"${VENV_PATH}/bin/python" \
    "${PROJECT_ROOT}/scripts/relax_mace.py" "${STRUCTURE}" --fmax 0.01
