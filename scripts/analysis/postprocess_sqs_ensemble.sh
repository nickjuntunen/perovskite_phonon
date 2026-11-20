#!/bin/bash
# Post-process the SQS ensemble (data/mixed_halides_ensemble; MLIP array
# mlip-mixed-ens, OUT_ROOT=$SCRATCH/perovskite_phonon/mlip_phonons_mixed_ensemble)
# into ensemble-averaged targets and re-scored series models. Reads the sweep
# through the sshfs mount mnt/. Rerunnable: the descriptor step resumes, so a
# rerun after more runs finish only describes the new ones.
#
#   1. link_ensemble_runs.py      one run root: new cells + the reused production / noise-check runs (reuse.csv)
#   2. select_stable_polymorphs   per run (--group-by run), as the noise check
#   3. extract_design_descriptors (--resume)
#   4. sqs_ensemble_average.py    per-composition descriptor means / spreads
#   5. sqs_ensemble_rescore.py    averaged targets; R2(series) of the existing models vs
#                                 single-SQS and averaged targets; figures
#
# Usage: bash scripts/analysis/postprocess_sqs_ensemble.sh [workers]
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${1:-10}
E=data/mixed_halides_ensemble
R=$E/runs
ls mnt/mlip_phonons_mixed_ensemble >/dev/null 2>&1 || { echo "mnt/ not mounted" >&2; exit 1; }

uv run python3 scripts/analysis/link_ensemble_runs.py --set-dir $E
uv run python3 scripts/analysis/select_stable_polymorphs.py --group-by run --root $R \
    --out-dir $E/analysis/selection --no-copy-structures
uv run python3 -u scripts/analysis/extract_design_descriptors.py --root $R --workers "$W" --resume \
    --polymorphs-csv $E/analysis/selection/stable_polymorphs.csv \
    --candidates-csv $E/analysis/selection/candidates.csv --out-csv $E/analysis/design_descriptors.csv
uv run python3 scripts/analysis/sqs_ensemble_average.py --noise-dir $E
uv run python3 scripts/analysis/sqs_ensemble_rescore.py --set-dir $E
