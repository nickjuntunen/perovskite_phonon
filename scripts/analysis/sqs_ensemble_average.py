"""Configurational averages over a mixed-halide noise-check set: several SQS
realizations per composition (generate_mixed_halides.py --mode noise-check),
reduced to one row per composition.

Each realization is a separate MLIP run, selected per run
(select_stable_polymorphs.py --group-by run) and put through
extract_design_descriptors.py like any other polymorph. Here, for every
composition, each numeric descriptor gets its manifest-weighted mean over the
dynamically stable realizations (structure_gen.ensemble_average; SQS cells
that came out symmetry-equivalent were pooled into one with the summed
weight), the weighted spread between realizations (std), the effective sample
size and the standard error of the mean. The whole-cell vDOS quantile
functions are averaged the same way (the 1D Wasserstein barycenter), so shape
descriptors can be computed from the averaged spectrum.

The spread is the disorder noise of a single-SQS production value; the mean is
what production would hold with an ensemble average instead. Unstable
realizations are left out of both and counted (n_unstable), since the
production set keeps only dynamically stable compositions.

Inputs, in --noise-dir:
    manifest.csv                                  realization weights
    analysis/selection/stable_polymorphs.csv      (select_stable_polymorphs.py --group-by run)
    analysis/design_descriptors.csv (+ _quantiles.npz)   (extract_design_descriptors.py)
Outputs, in --noise-dir/analysis:
    ensemble_average.csv      formula, n_realizations, n_stable, n_unstable, weight_stable,
                              <descriptor>_mean / _std / _sem, n_eff
    ensemble_average_quantiles.npz   levels, quantiles_cm1 (weighted mean), keys (= formula)

Usage, after the noise-check MLIP sweep (OUT_ROOT=$SCRATCH/perovskite_phonon/mlip_phonons_mixed_noise_check_mp):
    R=mnt/mlip_phonons_mixed_noise_check_mp/MACE-matpes-r2scan-omat-ft
    N=data/mixed_halides_noise_check_mp
    uv run python3 scripts/analysis/select_stable_polymorphs.py --group-by run --root $R \\
        --out-dir $N/analysis/selection --no-copy-structures
    uv run python3 scripts/analysis/extract_design_descriptors.py --root $R \\
        --polymorphs-csv $N/analysis/selection/stable_polymorphs.csv \\
        --candidates-csv $N/analysis/selection/candidates.csv --out-csv $N/analysis/design_descriptors.csv
    uv run python3 scripts/analysis/sqs_ensemble_average.py --noise-dir $N
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.paths import MIXED_HALIDES_NOISE_CHECK_MP_DIR
from src.analysis.design_descriptors import load_quantiles
from src.analysis.sqs_ensemble import average, realizations

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-dir", type=Path, default=MIXED_HALIDES_NOISE_CHECK_MP_DIR)
    args = parser.parse_args()
    out = args.noise_dir / "analysis"

    try:
        d = realizations(args.noise_dir)
    except ValueError as exc:
        raise SystemExit(str(exc))
    qz = np.load(out / "design_descriptors_quantiles.npz")
    table, q_mean, keys = average(d, load_quantiles(out / "design_descriptors_quantiles.npz"))
    table.to_csv(out / "ensemble_average.csv", index=False)
    np.savez(out / "ensemble_average_quantiles.npz", levels=qz["levels"], quantiles_cm1=q_mean, keys=np.array(keys))
    print(f"{len(d)} realizations ({int(d['dynamically_stable'].astype(bool).sum())} stable), "
          f"{len(table)} compositions -> {out / 'ensemble_average.csv'}")
    print("stable realizations per composition:", table["n_stable"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
