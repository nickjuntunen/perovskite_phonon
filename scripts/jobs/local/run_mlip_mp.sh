#!/bin/bash
# Local counterpart of scripts/jobs/sherlock/run_mlip_mp_array.sbatch: the MLIP
# phonon pipeline over the MP perovskite list, serially. Resumable (finished
# structures, i.e. with a meta.json, are skipped). The model runs with
# DFT-D3(BJ) (src/mlip/mlip_calculators.py): output is under
# ${OUT_ROOT}/<model>-D3BJ-<xc>/. Optional TASK_INDEX/NUM_TASKS
# split the list the same way the array does, e.g. to run two halves at once.
#
# Usage: ./run_mlip_mp.sh [DEVICE]   (default: cuda)

DEVICE="${1:-cuda}"
PROJECT_ROOT="/home/rotskofflab6/Desktop/research/perovskite_phonon"
MODEL="${PROJECT_ROOT}/models/MACE-matpes-r2scan-omat-ft.model"
STRUCTURE_LIST="${PROJECT_ROOT}/data/mp_perovskites/mlip_structures.txt"
OUT_ROOT="${PROJECT_ROOT}/data/mlip_phonons_mp"

cd "$PROJECT_ROOT"
uv run python3 -u scripts/mlip/run_mlip_phonons.py \
    --structure-list "$STRUCTURE_LIST" \
    --task-index "${TASK_INDEX:-0}" --num-tasks "${NUM_TASKS:-1}" \
    --models "$MODEL" \
    --relax full \
    --out-root "$OUT_ROOT" \
    --device "$DEVICE" \
    2>&1 | tee -a "${OUT_ROOT}.log"
