#!/bin/bash
# Submit MACE-relax -> SCF -> DFPT job chains on Sherlock for structures in
# $SCRATCH/perovskite_phonon/qe_inputs/.
#
# Usage:
#   ./submit_all.sh CsPbI3 NaMgF3     # submit specific structures
#   ./submit_all.sh --all             # submit every structure in $SCRATCH/perovskite_phonon/qe_inputs/
#                                      # (currently 300+ -- check your allocation
#                                      # before doing this)
#
# For each structure, submits the MACE relax job, then SCF, then DFPT, each
# with --dependency=afterok on the previous one, so a step only starts (and
# only runs at all) if the one before it succeeded.

set -euo pipefail
: "${SCRATCH:?\$SCRATCH is not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
QE_INPUTS_DIR="${SCRATCH}/perovskite_phonon/qe_inputs"
RELAX_FIRST_DIR="${SCRATCH}/perovskite_phonon/mace_relax_first"
SCF_FIRST_DIR="${SCRATCH}/perovskite_phonon/scf_first"
PH_FIRST_DIR="${SCRATCH}/perovskite_phonon/ph_first"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <structure> [<structure> ...] | --all" >&2
    exit 1
fi

if [[ "$1" == "--all" ]]; then
    structures=$(cd "${QE_INPUTS_DIR}" && ls -d */)
    structures=${structures//\//}
else
    structures="$*"
fi

for name in ${structures}; do
    run_dir="${QE_INPUTS_DIR}/${name}"
    if [[ ! -f "${run_dir}/${name}.scf.in" || ! -f "${run_dir}/${name}.ph.in" ]]; then
        echo "Skipping ${name}: missing .scf.in/.ph.in in ${run_dir}" >&2
        continue
    fi

    mkdir -p "${RELAX_FIRST_DIR}/${name}" "${SCF_FIRST_DIR}/${name}" "${PH_FIRST_DIR}/${name}"

    relax_id=$(sbatch --parsable --job-name="relax-${name}" --export=ALL,STRUCTURE="${name}",PROJECT_ROOT="${PROJECT_ROOT}" \
        --output="${RELAX_FIRST_DIR}/${name}/%x-%j.out" --error="${RELAX_FIRST_DIR}/${name}/%x-%j.err" \
        "${SCRIPT_DIR}/run_mace_relax.sbatch")
    scf_id=$(sbatch --parsable --job-name="scf-${name}" --dependency="afterok:${relax_id}" \
        --export=ALL,STRUCTURE="${name}",PROJECT_ROOT="${PROJECT_ROOT}" \
        --output="${SCF_FIRST_DIR}/${name}/%x-%j.out" --error="${SCF_FIRST_DIR}/${name}/%x-%j.err" \
        "${SCRIPT_DIR}/run_scf.sbatch")
    ph_id=$(sbatch --parsable --job-name="ph-${name}" --dependency="afterok:${scf_id}" \
        --export=ALL,STRUCTURE="${name}",PROJECT_ROOT="${PROJECT_ROOT}" \
        --output="${PH_FIRST_DIR}/${name}/%x-%j.out" --error="${PH_FIRST_DIR}/${name}/%x-%j.err" \
        "${SCRIPT_DIR}/run_ph.sbatch")

    echo "${name}: relax job ${relax_id} -> scf job ${scf_id} -> ph job ${ph_id}"
done
