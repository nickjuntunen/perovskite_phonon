"""Where things live in the repository -- the one definition of each
directory more than one module or script reads or writes. Script-specific
outputs stay in their script.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = PROJECT_ROOT / "templates" / "qe"

# QE chain: structures in, inputs out, local pw.x/ph.x scratch
CUBIC_STRUCTURES_DIR = DATA_DIR / "cubic_structures"  # Shannon-radii cubic ABX3 guesses
RELAXED_STRUCTURES_DIR = DATA_DIR / "relaxed_structures"  # ... after scripts/mlip/relax_mace.py
TILTED_STRUCTURES_DIR = DATA_DIR / "tilted_structures"  # soft modes frozen in + MLIP-relaxed
RATTLED_STRUCTURES_DIR = DATA_DIR / "rattled_structures"
QE_INPUTS_DIR = DATA_DIR / "qe_inputs"
QE_OUT_DIR = PROJECT_ROOT / "out"
PSEUDO_DIR = DATA_DIR / "pseudopotentials"
CONVERGENCE_TESTS_DIR = DATA_DIR / "convergence_tests"
NOMAD_IONS_PATH = DATA_DIR / "nomad" / "nomad_ions.json"

# Materials Project perovskite set (scripts/dataset/)
MP_DIR = DATA_DIR / "mp_perovskites"
MP_STRUCTURES_DIR = MP_DIR / "structures"
MP_METADATA = MP_DIR / "metadata.csv"  # fetch_mp_perovskites.py
MP_METADATA_VERIFIED = MP_DIR / "metadata_verified.csv"  # verify_mp_perovskites.py
MP_METADATA_STABLE = MP_DIR / "metadata_stable.csv"  # filter_stable_mp_perovskites.py
MP_QE_INPUTS_DIR = DATA_DIR / "mp_qe_inputs"
# The MP structure list the sweep ran (write_mp_structure_list.py, generate_mixed_halides.py
# --source mp-clusters read it as a membership list); its 2026-09 KMeans labels are superseded
# by MP_CHEM_CLUSTERS_CSV
MP_COMPOSITION_CLUSTERS_CSV = DATA_DIR / "mp_composition_clusters.csv"
MP_COMPOSITION_FEATURES_CSV = MP_DIR / "composition_features.csv"  # extract_composition_features.py
# Chemical (composition) clusters: cluster_compositions.py --features-csv; select_stable_polymorphs.py
# attaches them as `cluster` (chem_cluster downstream)
MP_CHEM_CLUSTERS_CSV = MP_DIR / "composition_clusters.csv"

# MLIP phonons
MLIP_PHONONS_DIR = DATA_DIR / "mlip_phonons"  # run_mlip_phonons.py default --out-root
STABLE_POLYMORPHS_DIR = DATA_DIR / "mlip_stable_polymorphs"  # select_stable_polymorphs.py and downstream
# The MP sweep the polymorph selection / descriptor analysis was built on (sshfs of
# Sherlock scratch; pre-D3 -- a D3 sweep lands in <model>-D3BJ-<xc>/ next to it)
MP_SWEEP_DIR = PROJECT_ROOT / "mnt" / "mlip_phonons_mp" / "MACE-matpes-r2scan-omat-ft"

# Mixed-halide SQS sets (scripts/dataset/generate_mixed_halides.py)
MIXED_HALIDES_DIR = DATA_DIR / "mixed_halides"
MIXED_HALIDES_NOISE_CHECK_DIR = DATA_DIR / "mixed_halides_noise_check"
MIXED_HALIDES_NOISE_CHECK_MP_DIR = DATA_DIR / "mixed_halides_noise_check_mp"

# Synced Sherlock QE results, one directory per structure
PH_FIRST_DIR = PROJECT_ROOT / "sherlock_outputs" / "ph_first"
SCF_FIRST_DIR = PROJECT_ROOT / "sherlock_outputs" / "scf_first"
