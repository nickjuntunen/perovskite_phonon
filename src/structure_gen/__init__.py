"""Structure generation for substitutionally disordered perovskites (mixed
halides and mixed A/B sites): parent lattices and mixing rules (parent),
finite-cell disorder ensembles -- SQS, random, exact enumeration -- (disorder),
and writing them as MLIP-sweep inputs (io).

    from src.structure_gen import cubic_parent, sqs, write_realizations
    parent = cubic_parent("Cs", "Pb", {"Br": 2/3, "I": 1/3})
    # production: one SQS per composition, treated downstream like an MP entry
    write_realizations(sqs(parent, "2x2x2", n=1), "data/mixed_halides")
    # disorder noise check: several seeds, in its own directory
    write_realizations(sqs(parent, "2x2x2", n=5), "data/mixed_halides_noise_check",
                       subset="noise_check", prefix="nc-")
"""

from src.structure_gen.disorder import (
    Realization,
    cluster_space,
    correlation_error,
    default_cutoffs,
    ensemble_average,
    enumerate_orderings,
    n_decorations,
    random_alloy_correlations,
    random_configurations,
    sqs,
)
from src.structure_gen.io import realization_name, write_realizations
from src.structure_gen.parent import (
    SUPERCELL_PRESETS,
    ParentLattice,
    binary_fractions,
    commensurate_tilt_points,
    conventional_cell,
    cubic_parent,
    formability,
    halide_pairs,
    scaled_to_volume,
    supercell_for_sites,
    vegard_lattice_parameter,
)
