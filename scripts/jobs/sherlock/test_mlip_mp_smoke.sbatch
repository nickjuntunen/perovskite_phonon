#!/bin/bash
#SBATCH --job-name=mlip-mp-smoke
#SBATCH --partition=owners,hns,gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_CC:8.0|GPU_CC:9.0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.out
#SBATCH --error=/scratch/users/juntunen/perovskite_phonon/logs/%x-%j.err

# Logs: #SBATCH lines can't expand $SCRATCH, hence the literal path. Slurm
# doesn't create it -- a missing directory fails the job with no log at all:
#   mkdir -p $SCRATCH/perovskite_phonon/logs

# Smoke test for run_mlip_mp_array.sbatch: same environment, model and flags,
# on a handful of structures, before spending 99 GPU allocations. Job 46355821
# lost every structure to two environment errors (a phonopy/phonors API
# mismatch, no sm_70 kernels on V100s) that one structure per GPU type would
# have shown.
#
# 1. Preflight, in the container on the allocated GPU: torch has kernels for
#    this GPU's compute capability, PYTORCH_CUDA_ALLOC_CONF reaches the
#    container, phonopy/phonors agree, cuequivariance imports. Exits 1 on
#    the first problem, before loading the model.
# 2. The real pipeline (run_mlip_phonons.py) on SMOKE_SET:
#      small (default)  BaSnO3 (5 atoms, 4x4x4), Ba2MgWO6 (10 atoms),
#                       CsPbBr3 (cubic, imaginary R/M tilts: exercises the
#                       rattle retry and soft-mode steps)
#      large            the two job-46355821 CUDA OOMs on a 40 GB A100:
#                       La4Ni3O10 (2176-atom supercell), Ba10Ti8NbNiO30 (3200);
#                       slow -- give it more --time
#    Output goes to a separate OUT_ROOT, so nothing here is mistaken for (or
#    skipped by) the production array; --overwrite reruns it from scratch.
#
# Usage, from anywhere -- one job per GPU type you want the array to use:
#   sbatch -C GPU_CC:8.0 --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon test_mlip_mp_smoke.sbatch
#   sbatch -C GPU_CC:9.0 --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon test_mlip_mp_smoke.sbatch
#   sbatch -C GPU_CC:8.0 --time=08:00:00 \
#       --export=ALL,PROJECT_ROOT=$HOME/perovskite_phonon,SMOKE_SET=large test_mlip_mp_smoke.sbatch
# Pass: the .out ends with "SMOKE TEST PASSED", and each structure has a
# meta.json under OUT_ROOT (printed at the end).

set -euo pipefail
: "${SCRATCH:?\$SCRATCH is not set}"
# NOT derived from ${BASH_SOURCE[0]}: sbatch runs a spooled copy of this script.
: "${PROJECT_ROOT:?Set PROJECT_ROOT, e.g. --export=ALL,PROJECT_ROOT=\$HOME/perovskite_phonon}"

# Keep in sync with run_mlip_mp_array.sbatch.
MODEL="${MODEL:-/home/groups/rotskoff/nick/MACE-matpes-r2scan-omat-ft.model}"
SMOKE_SET="${SMOKE_SET:-small}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH}/perovskite_phonon/mlip_phonons_mp_smoke}"
STRUCTURE_DIR="${PROJECT_ROOT}/data/mp_perovskites/structures"

case "${SMOKE_SET}" in
    small) NAMES=(mp-3163_BaSnO3 mp-18986_Ba2MgWO6 mp-1014168_CsPbBr3) ;;
    large) NAMES=(mp-558316_La4Ni3O10 mp-1228979_Ba10Ti8NbNiO30) ;;
    *) echo "SMOKE_SET must be small or large, got '${SMOKE_SET}'" >&2; exit 1 ;;
esac
STRUCTURES=()
for name in "${NAMES[@]}"; do
    f="${STRUCTURE_DIR}/${name}.cif"
    [[ -f "$f" ]] || { echo "structure not found: $f" >&2; exit 1; }
    STRUCTURES+=("$f")
done

set +euo pipefail
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/sourceme.sh"
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The checkpoint path is MODEL minus any +d3-<xc>/+nod3 and @<head> suffix.
MODEL_PATH="${MODEL%+d3-*}"; MODEL_PATH="${MODEL_PATH%+nod3}"; MODEL_PATH="${MODEL_PATH%@*}"
[[ -f "${MODEL_PATH}" ]] || { echo "MLIP checkpoint not found: ${MODEL_PATH}" >&2; exit 1; }

echo "=== job ${SLURM_JOB_ID} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, SMOKE_SET=${SMOKE_SET}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== preflight"
apptainer-python -u - <<'EOF'
import os, sys

problems = []

import torch
if not torch.cuda.is_available():
    sys.exit("PREFLIGHT FAILED: torch.cuda.is_available() is False")
major, minor = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)} (sm_{major}{minor}), built for {archs}")
# sm_XY runs on a binary built for sm_XW with W <= Y (same major), or PTX compute_XW.
if not any(a.startswith(f"sm_{major}") and int(a[len(f"sm_{major}"):]) <= minor for a in archs if a.startswith("sm_")):
    problems.append(f"torch has no kernels for sm_{major}{minor} (built for {archs})")
else:
    # Actually launch one: the arch list is a proxy, this is the real check.
    try:
        (torch.ones(8, dtype=torch.float64, device="cuda") * 2).sum().item()
    except Exception as exc:
        problems.append(f"CUDA kernel launch failed: {type(exc).__name__}: {exc}")

conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print(f"PYTORCH_CUDA_ALLOC_CONF in container: {conf!r}")
if conf is None or "expandable_segments:True" not in conf:
    problems.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True did not reach the container")

from importlib.metadata import version
print(f"phonopy {version('phonopy')}, phonors {version('phonors')}")
# Our own force constants -> mesh -> tetrahedron DOS path, on fcc Cu with
# ASE's EMT (CPU, seconds): the mesh is where job 46355821's phonopy called
# a function phonors had renamed.
try:
    sys.path.insert(0, os.environ["PROJECT_ROOT"])
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from src.mlip import mlip_phonons
    run = mlip_phonons.compute_force_constants(bulk("Cu", "fcc", a=3.6), EMT(), (2, 2, 2))
    mlip_phonons.total_dos_cm1(run.phonon, mesh=(8, 8, 8))
    print(f"phonopy mesh/DOS OK (EMT Cu, min {mlip_phonons.min_mesh_frequency_cm1(run.phonon):.1f} cm-1)")
except Exception as exc:
    problems.append(f"phonopy force constants/mesh/DOS failed: {type(exc).__name__}: {exc}")

try:
    import cuequivariance_torch  # noqa: F401
    from mace.calculators.mace import CUEQQ_AVAILABLE
    print(f"cuequivariance_torch {version('cuequivariance_torch')}, mace sees it: {CUEQQ_AVAILABLE}")
    if not CUEQQ_AVAILABLE:
        problems.append("mace-torch cannot import its cuequivariance converter")
except Exception as exc:
    problems.append(f"cuequivariance_torch import failed: {type(exc).__name__}: {exc}")

if problems:
    print("PREFLIGHT FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("preflight OK")
EOF

echo "=== pipeline on ${#STRUCTURES[@]} structure(s)"
start=$SECONDS
status=0
apptainer-python -u \
    "${PROJECT_ROOT}/scripts/mlip/run_mlip_phonons.py" \
    "${STRUCTURES[@]}" \
    --models "${MODEL}" \
    --relax full \
    --out-root "${OUT_ROOT}" \
    --device cuda \
    --overwrite \
    ${EXTRA_ARGS:-} || status=$?
echo "=== pipeline exit ${status} after $(( SECONDS - start )) s"

# The output directory name, as run_mlip_phonons.py derives it (it carries -D3BJ-<xc> unless +nod3).
label=$(apptainer-python -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.mlip.mlip_calculators import model_label; print(model_label('${MODEL}'))" | tail -1)
missing=0
for name in "${NAMES[@]}"; do
    meta="${OUT_ROOT}/${label}/${name}/meta.json"
    if [[ -f "$meta" ]]; then
        echo "  OK      ${meta}"
    else
        echo "  MISSING ${meta}"
        missing=$(( missing + 1 ))
    fi
done

if (( status != 0 || missing > 0 )); then
    echo "SMOKE TEST FAILED (${missing} structure(s) without meta.json)"
    exit 1
fi
echo "SMOKE TEST PASSED"
