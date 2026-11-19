#!/bin/bash
#SBATCH --job-name=setup-analysis-venv
#SBATCH --partition=normal,owners
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.err

# The CPU analysis venv for gp_vs_tree/ and scripts/analysis/ (not the MLIP venv,
# which lives in the CUDA container), on Sherlock's python/3.12.1 module:
#   $GROUP_HOME/nick/python_venvs/perovskite_analysis
#
# Binary wheels only. Sherlock is CentOS 7 (glibc 2.17): recent numpy/scipy ship
# no manylinux2014 wheel, and building them from source fails (the system
# assembler predates `endbr64`). The pins below are the newest versions with
# glibc-2.17 wheels (resolved 2026-11-19); they trail the local uv lock
# (numpy 2.5.2, sklearn 1.9.1, ...) without any API difference gp_vs_tree uses.
# The installed set is frozen to $V/requirements.lock.
#
# Usage: sbatch scripts/jobs/sherlock/setup_analysis_venv.sbatch   (rebuilds from scratch)

set -euo pipefail
ml python/3.12.1
V=$GROUP_HOME/nick/python_venvs/perovskite_analysis
P=$HOME/perovskite_phonon
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

rm -rf "${V:?}"
python3 -m venv "$V"
source "$V/bin/activate"
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: \
    numpy==2.2.6 pandas==2.3.2 scipy==1.16.3 scikit-learn==1.7.2 joblib==1.6.0 threadpoolctl==3.8.0 \
    matplotlib==3.11.2 ase==3.29.0 mendeleev==1.3.0 pymatgen==2026.10.24
# pyrovskite from GitHub, the commit the local venv has: PyPI's 1.0.0 has no
# pyrovskite.data, which src/data/perovskite_builder.py imports (job 52979192
# failed on that). Pure Python, built from source, so not under --only-binary;
# its dependencies (pymatgen, ase) are already installed above.
python -m pip install --no-deps "pyrovskite @ git+https://github.com/r2stanton/pyrovskite.git@3c89f4b55308bff6230178d3168af5b876611a52"
python -m pip freeze > "$V/requirements.lock"

cd "$P"
python - <<'PY'
import sys; sys.path.insert(0, ".")
import pandas as pd
from src.analysis.composition_features import x_site_disorder
r = x_site_disorder(pd.DataFrame({"x_frac_F": [0.0], "x_frac_Cl": [0.5], "x_frac_Br": [0.0], "x_frac_I": [0.5]}))
print(r.round(4).to_string())
assert abs(r["mass_cv_X"].iloc[0] - 0.5633) < 1e-3
import numpy, sklearn, joblib, pyrovskite.data
print("numpy", numpy.__version__, "sklearn", sklearn.__version__, "ANALYSIS VENV OK")
PY
