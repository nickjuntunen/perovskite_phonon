# Claude session log

Progress record for Claude Code sessions on this repo (CLAUDE.md points new sessions here).

**Conventions**
- One entry per session, newest first. Each entry has: date range, what was done, where the outputs are, current state, open threads.
- Write facts verified in the session; mark anything not checked as such.
- Keep standing constraints (Sherlock policy, workflow rules) in the section below, not in entries.
- At the end of a session, add an entry or update the current one before stopping.

## Standing constraints

- **Sherlock:** read `/etc/agents/AGENTS.md` at the start of every session and after any context compaction, plus the relevant topic files (`slurm.md`, `storage.md`, `software.md`, `policy.md`).
- **Login nodes** are for light inspection and job preparation only.
- **Jobs:**
  - Every job gets an explicit `--time`.
  - Wait at least 60 s between status checks, with no polling loops.
  - Never resubmit a failed job without finding the cause.
- **Who submits:** the user normally submits sbatch jobs. Draft the command and check it with `sbatch --test-only`.
- **Sherlock code vs local:** since 2026-11-18 the local tree contains every Sherlock code edit (see that day's entry), so local is the superset. Sherlock still runs its own older copy.
  - Never sync local code over Sherlock without the user's explicit go-ahead.
  - Local makes DFT-D3(BJ) the default, which would change Sherlock's output directories and physics.
  - Copying new data directories to a new path is fine.
- **`mnt/`** is an sshfs mount of `sherlock:$SCRATCH/perovskite_phonon`. If it's empty, it's unmounted; the data isn't missing.
- **Running notebooks:** run `descriptor_selection.ipynb` with nbclient (`uv run --with nbformat --with nbclient --with ipykernel python run_nb.py <in> <out>`). Don't set `MPLBACKEND=Agg` for notebooks; do set it for scripts.

---

## 2026-11-18: Sherlock → local code merge, SQS noise-check post-processing, commit

- **Merge** (local only; Sherlock untouched).
  - Both trees sit on commit 272d1e1 with uncommitted work. Sherlock was overwritten with the local tree on 2026-10-02. After that:
    - On Sherlock, 12 files were edited (Oct 2–7).
    - Locally, the user refactored on 2026-11-11, from the Oct 2 tree: `phonon_pipeline.py` split out, D3 default, `src/paths.py`, `src/analysis/`, `src/utils/`.
  - Files with Sherlock mtimes ≤ Oct 2 are unchanged since the sync, so the local version stands for all of them. That includes `src/visualize.py` and `src/phonon_benchmark.py`: the refactor moved them, and dropped `parse_qe_modes`, which nothing uses.
  - **Ported from Sherlock into the local layout:**
    - `--mesh-length` (q-density vDOS mesh, `PhononRunConfig.mesh_length`) and `_dos_mesh`, which now also documents the organic-cation tetrahedron-weight memory.
    - `--reuse-fc` (`_reused_attempt`, `mlip_phonons.load_force_constant_run`, `ForceConstantRun.wall_time_s` may be None).
    - Saving `phonopy_params.yaml` before the DOS/band step (`save_dir`), and `_spectra_attempt`.
    - The soft-mode DOS mesh now comes from the first attempt's `dos_mesh`.
    - meta: `force_constants_reused`, per-attempt `dos_mesh`, `dos_mesh_length_A`.
    - `--max-env-failures`, `_is_environment_error`.
    - MACE `enable_cueq` on CUDA.
    - The array sbatch: GPU_CC 8.0/9.0 only (no sm_70 kernels), `PYTORCH_CUDA_ALLOC_CONF`, the real default MODEL path, and absolute `/scratch` log paths. The four other sbatches got the same log paths.
    - `test_mlip_mp_smoke.sbatch`, adapted to the D3 suffixes: MODEL_PATH stripping, and the label via `model_label`.
  - **Not copied:** `sourceme.sh` (environment, not in git).
  - **Tested locally** on CsPbBr3 with MACE-matpes `+nod3`, 2×2×2: a fresh run, then `--reuse-fc`, which gives `force_constants_reused: true`.
- **Commit:** `caf4727` on the new branch `merge-sherlock-refactor`, not pushed.
  - `pubchem_*.parquet` and `pubchem_*.csv` were added to `.gitignore`.
  - `get_sh_molecules.ipynb` outputs were cleared; there were 23 MB in cell 1.
- **D3 deployed to Sherlock** (user request):
  - Backup of the old Sherlock code: `$GROUP_HOME/nick/perovskite_phonon_code_backup_20261118.tar.gz`.
  - Synced the code, the top-level tracked files and `.git`. Sherlock's checkout is now exactly `caf4727` plus `sourceme.sh`. The only deletions were the moved `src/visualize.py` and `src/phonon_benchmark.py`.
  - **Zwitterions:** the user chose to redo all 90 with D3.
    - Cancelled pending tasks 52692210_[3-14].
    - The 18 no-D3 results stay in `mlip_phonons_zwitterions/MACE-matpes-r2scan-omat-ft/`.
  - **Job chain:**

    | Job | Name | What | Dependency |
    |---|---|---|---|
    | 52857069 | `install-d3` | owners/normal, 30 min. pip installs `nvalchemi-toolkit-ops==0.3.1` and `warp-lang==1.17.0` in the venv, builds `~/.cache/nvalchemiops/dftd3_parameters.pt`, and checks E_D3(cubic CsPbBr3, r2scan) = −0.407128 eV against local. Script: `$SCRATCH/perovskite_phonon/jobs/install_d3.sbatch`. | — |
    | 52857094 | `mlip-mp-smoke` | `test_mlip_mp_smoke.sbatch` with the default model, so D3 r2scan | afterok:52857069 |
    | 52857102 | `mlip-zw-d3` | `--array=0-14`, `NUM_TASKS=15`, `--mem=64G`, 6 h. Output in `mlip_phonons_zwitterions/MACE-matpes-r2scan-omat-ft-D3BJ-r2scan/`. | afterok:52857094 |

  - If any job in the chain fails, the ones after it never start (DependencyNeverSatisfied): find the cause, then scancel them.
  - **First attempt failed.** 52857069 FAILED after 8 s. The install script's `set -e` was active while sourcing `sourceme.sh`, whose `find` exits nonzero on an absent `nvidia/cuda_nvrtc`. Slurm cancelled 52857094 and 52857102 itself.
    - Fixed: `set +eo pipefail` around the source, as the array sbatch does.
    - **Resubmitted:** install **52858500** → smoke **52858505** → `mlip-zw-d3` **52858514**. Same settings as before.
- **SQS noise check:**
  - 528/529 realizations finished. The exception is `nc-sqs4-1x1x1_Cs2NaIn(BrCl)3`, which fails deterministically with "Failed to recognize lattice" (ASE bandpath) after its force constants are saved.
  - Post-processing: the three steps from `sqs_ensemble_average.py`'s docstring, with production's default `--imag-tol 1`. Log: `data/mixed_halides_noise_check_mp/analysis/postprocess.log`.
  - **Done** (23 min, 0 failures): 855 states from 528 runs, 115 compositions. 509 of 528 realizations are stable; stable realizations per composition: {2: 2, 3: 9, 4: 42, 5: 62}.
  - The notebook ran to `<scratchpad>/descriptor_selection_run.ipynb`; the user's copy is unchanged. It wrote `data/mlip_stable_polymorphs/descriptor_sqs_noise.csv`.
  - `sqs0_repro` = 0 for every target, so the noise-check seed 0 reproduces production.
  - **Ceilings on R²(series):**
    - ≥ 0.96: rel_centroid_X, cv, ω_log, S, log_k_X.
    - 0.85–0.93: low_enrich_A/X, shape_pc1–4, log_k_B, high_enrich_B, rel_centroid_A/B.
    - Disorder-limited (SNR_series ≤ 1.9): ω_max 0.73, rel_width_X 0.77, rel_width_A 0.62, rel_width_B 0.67, log_msd_A/B/X 0.35–0.62, log_k_A 0.28.
  - Re-scoring against the ensemble mean barely changes any model's R². Where models fall short, the x-dependence is unlearned, not drowned in noise; most targets have models well below their ceiling.
  - GP beats trees and linear on almost every target.
  - **Core targets, seed noise vs gauntlet MLIP noise:** ω_log 0.29×, rel_centroid_A 1.02×, rel_width_A 0.78×, shape_pc1 0.59×, rel_centroid_B 1.47×. So single-SQS noise doesn't swamp the MLIP-noise units of `cluster_targets.py`, except it inflates rel_centroid_B.

- **D3 chain result:**
  - Install **52858500** COMPLETED. E_D3(CsPbBr3) = −0.407128 eV matches local, and the cache was built.
  - Smoke **52858505** PASSED: BaSnO3, Ba2MgWO6 and CsPbBr3, label `-D3BJ-r2scan`.
  - Note: with D3, cubic CsPbBr3 is only −3.4 cm⁻¹ unstable. The soft-mode step reached Cm at +0.00 meV/atom with −7.1 cm⁻¹ left, not a tilted phase. D3 visibly softens the tilt instability; compare with no-D3 before trusting either.
  - The zwitterion D3 array **52858514** is now released and queued.
- **Why the models fail along series** (diagnostic, trees, production mixed set):
  - Within-series std is 0.2–0.47 of the total.
  - The poor targets bow: a line fits 42–63%, a quadratic 85–91%.
  - Held-out pair is much worse than held-out series (rel_centroid_B 0.51 → −0.60).
  - The C tier has no X-site disorder feature.
- **Experiment for it:**
  - `src/analysis/composition_features.py`: `x_site_disorder()` and `X_DISORDER_FEATURES` (x_mix = 1−Σx², mass_cv_X, r_spread_X, en_spread_X), from x_frac only.
  - `gp_vs_tree/gp_disorder.py`: 9 targets × {base, +dis, +dis with the bowing-only Δ target} × {held-out pair, GroupKFold(5) series} × {GP, tree}, checkpointed.
  - The quick local test reproduces the notebook's linear R²(series) for rel_centroid_A (0.1868).
  - `scripts/jobs/sherlock/run_gp_disorder.sbatch` (array 0–8, 16 CPU, 12 h, requeue).
  - Its environment is a new analysis venv, `$GROUP_HOME/nick/python_venvs/perovskite_analysis` (python/3.12.1 module, pinned to local versions), built by `$SCRATCH/perovskite_phonon/jobs/setup_analysis_venv.sbatch`.
  - Inputs copied to new Sherlock paths: `data/mlip_stable_polymorphs/` (4 files), `data/mp_perovskites/composition_features.csv`, `data/mixed_halides/analysis/` (4 files), `data/nomad/nomad_ions.json`.
  - **Jobs:** venv **52863595** → array **52863609** → summary **52863615**. Output: `~/perovskite_phonon/gp_vs_tree/disorder/gp_disorder.csv` on Sherlock.
  - **2026-11-19: venv job 52863595 FAILED** (12 min); Slurm cancelled the array and the summary.
    - Cause: Sherlock is CentOS 7 (glibc 2.17). numpy 2.5.2 has no manylinux2014 wheel, so pip built it from source, and the system assembler rejects `endbr64`.
    - Fix: `scripts/jobs/sherlock/setup_analysis_venv.sbatch` (now in the repo) installs with `--only-binary=:all:`. The pins are the newest versions with glibc-2.17 wheels: numpy 2.2.6, pandas 2.3.2, scipy 1.16.3, scikit-learn 1.7.2, matplotlib 3.11.2, mendeleev 1.3.0, pymatgen 2026.10.24. It rebuilds the venv from scratch and writes `requirements.lock` into it.
    - `install_d3.sbatch` (job 52858500, worked) is also in the repo now.

## 2026-11-17: target-space clustering (redundant compositions)

- **`scripts/analysis/cluster_targets.py`** (new) clusters compositions on the six core targets of `descriptor_selection.md` §6.
  - **Data:** the MP stable polymorphs (2,657) and the mixed halides (993); 3,650 polymorphs and 3,606 formulas in all. These match the notebook's current counts. The `.md`'s figure of 1,984 is stale.
  - **Units:** each target is divided by its gauntlet MLIP noise, as in notebook cell 11. `log_distortion_energy` has no gauntlet value, so its noise is set to std / median SNR of the others (10.3). This is an assumption.
  - **Method:** complete linkage with the Chebyshev metric, cut at tol = 1, 2 and 3 noise units. Every member pair is then within tol on every target.
  - **Chemical spread:** median pairwise distance in standardized C descriptors, divided by the random-pair median.
- **Results** (in `data/target_clusters/`):

  | tol | Formulas with a partner | Redundant clusters | Max formulas per cluster | Clusters with chem spread > 0.5 |
  |---|---|---|---|---|
  | 1 | 18% | 276 | 5 | 1 |
  | 2 | 48% | 660 | 10 | 19 |
  | 3 | 72% | 892 | 17 | 55 |

  - Most redundancy is chemically trivial (median spread 0.14): rare-earth swaps in oxides, and different orderings or supercells of one parent. Oxides have the most: 71% of them have a partner at tol 2.
  - The non-trivial cases are almost all mixed-halide double perovskites, e.g. Cs2AgSb(I2Cl)2 ≈ CsEuBr3 and Rb2NaY(I2Cl)2 ≈ Rb2TlCu(Br2Cl)2.
- **Caveats:**
  - The noise unit is measured on CsPbBr3 only.
  - For mixed halides, the single-SQS disorder noise hasn't been measured yet (noise-check post-processing is pending), so their clusters at tol ≤ 2 aren't reliable.
  - ~~The "other" family is still included.~~ **Fixed the same day:** `X_ANIONS` now keeps only MP polymorphs whose X site is O, S, Se, Te, N, F, Cl, Br or I.
    - Dropped 212 antiperovskite formulas with a metal on the X site. This includes ionic ones such as Ca3BiN and Sr3PbO.
    - Rerun: 3,438 polymorphs (2,445 MP).

      | tol | Formulas with a partner | Max formulas per cluster | Clusters with spread > 0.5 |
      |---|---|---|---|
      | 1 | 19% | 5 | 1 |
      | 2 | 51% | 11 | 28 |
      | 3 | 75% | 16 | 63 |

    - More clusters now clear spread > 0.5 because the random-pair reference got smaller. The antiperovskites had widened it.
    - At tol 2, 11 of the 28 are oxides, e.g. TiCdO3 ≈ La10MgTi7O30. The rest are mixed-halide double perovskites.
  - `log_omega_log` keeps clusters mostly within one anion family. Drop it to look for scale-free redundancy across anions.

---

## 2026-11-16: thiol zwitterions for the X/A-site replacement

- **`get_sh_molecules.ipynb`:**
  - Cell 1: the single GET of all 808,870 `[SH]` CIDs failed (SSLEOFError, because the URL was too long). It now sends batched POSTs.
  - Cell 2: keeps molecules that can be +1/−1 zwitterions. Also adds zwitterions drawn as charged, from a second search (`MatchCharges=true`). Keeps 3–4 bonds between N and S and drops weak bases. Molecules are deduplicated by their neutral parent.
  - Cell 3: aliphatic, stable, single-basic-site "simple" class, giving 1,109 molecules. Picks 40 of them, stratified by cation class × bond count, with cysteamine forced in.
  - Outputs: `pubchem_zwitterion_picks.csv`, `pubchem_zwitterions_aliphatic.parquet`.
- **`scripts/dataset/build_zwitterion_perovskites.py`** (new) builds from the cysteamine prototype `data/organic_halide_replacement_structures/CYS-PbCl-ZW.cif`.
  - Each structure is 6 × PbX2(zwitterion), for X = Cl, Br, I. The lattice is scaled by the CsPbX3 lattice parameter, and the cell is expanded isotropically until the molecules are clash-free.
  - Output: `data/organic_halide_replacement_structures/zwitterion_picks/`, holding 120 CIFs, `manifest.csv` and `mlip_structures.txt`.
  - Cysteamine-Cl reproduces the prototype (heavy-atom match).
- **First build: most picks don't fit the 3D cage.** This is the A-site size limit behind the Goldschmidt tolerance factor.
  - The median expansion was 1.28 for Cl, 1.22 for Br and 1.14 for I. Pb–X gets stretched to as much as 4.0 Å.
  - 9 structures still clashed at the 1.40 cap.
  - Only cysteamine and 2-(methylamino)propane-1-thiol fit at ≤ 1.06 for all three halides.
  - Those 120 CIFs were deleted and replaced by the fit-screened set (below).
- **Builder changes made after that build:**
  - Conformers now come from rotating each rotatable bond to 60°, 180° and 300°. ETKDG alone gave only the folded N–H···S shape, so cysteamine needed a 6% expansion; with the change it needs none.
  - The S→N axis can tilt up to 25° (6 azimuths). 24 spin angles.
  - Placement starts from the best symmetric packing (the same pose on all 6 sites, as in the prototype), then refines site by site with a soft score (0.3 Å margin).
  - New options: `--screen` (writes only `screen.csv`), `--max-expansion`, `--jobs`, and `.parquet` input.
  - Check: cysteamine-Cl builds at 1.00. 3-aminobutane-2-thiol (6 heavy atoms) still needs 1.16, so even C4 branched molecules are too big.
- **Steric-fit screen, done:**
  - Command: `--picks pubchem_zwitterions_aliphatic.parquet --halides Cl --max-expansion 1.10 --screen --jobs 10 --out data/organic_halide_replacement_structures/fit_screen`. Run locally, about 45 min.
  - Output: `fit_screen/screen.csv`.
  - Only 34 of 1,109 fit at ≤ 1.10 with Cl, and only 9 at ≤ 1.04. None have more than 7 heavy atoms, and no quaternary ammonium fits.
- **Notebook cell 4** (`get_sh_molecules.ipynb`) keeps every molecule that fits at ≤ 1.10 with Cl.
  - It drops 2 N,O-acetal ethers and 2 fused bicyclics (strained), leaving 30 molecules.
  - Cation classes: 11 primary, 11 secondary, 5 tertiary, 3 amidine. Includes cysteamine.
  - Output: `pubchem_zwitterion_fit_picks.csv`.
  - `pubchem_zwitterion_picks.csv` (the 40-molecule pick) is now only an intermediate.
- **Final build:** `zwitterion_picks/` holds 90 structures (30 × Cl/Br/I), 84–138 atoms each, all clash-free.
  - Expansion medians: Cl 1.10 (max 1.10), Br 1.04 (max 1.06), I 1.00 (max 1.02).
  - Files: `manifest.csv`, `mlip_structures.txt`.
- **Submitted 2026-11-16 16:08** (the user asked Claude to transfer and submit):
  - Data copied to `$HOME/perovskite_phonon/data/organic_halide_replacement_structures/`: `CYS-PbCl-ZW.cif` and `zwitterion_picks/`.
  - Both jobs use `run_mlip_mp_array.sbatch` with the default model (MACE-matpes-r2scan-omat-ft), `--relax full`.
  - `OUT_ROOT=$SCRATCH/perovskite_phonon/mlip_phonons_zwitterions`.
  - Jobs:
    - **52574151**, `mlip-zw-test`: `--array=0`, `NUM_TASKS=90`, 1 h. Structure 0 only (cysteamine-PbCl2).
    - **52574153**, `mlip-zw`: `--array=0-14`, `NUM_TASKS=15`, 6 h, `--dependency=afterok:52574151`. Resumable, so it skips the structure the test finished.
  - If the test fails, 52574153 stays pending (DependencyNeverSatisfied). Find the cause, then `scancel` it before resubmitting.
  - **2026-11-17: the test failed with OUT_OF_MEMORY** after 3 min 54 s at the 32G limit (MaxRSS 33.5 GB). Slurm cancelled 52574153 on its own.
    - Relax and forces finished: `phonopy_params.yaml` was written with 168 displacements on a 2×2×1 supercell, P1.
    - Reproduced locally from that file. The OOM is phonopy's tetrahedron DOS, `get_integration_weights`, which allocates (n_q, n_bins, n_bands) float64: (2268, 8004, 252) = 34.1 GiB.
    - n_bins is 8004 because the organic N–H/C–H stretches put the frequency range near 4000 cm⁻¹ at 0.5 cm⁻¹ pitch. Inorganic cells top out near 300 cm⁻¹, which is why other sets never hit this.
    - Estimated over all 90 structures: at the default mesh-length of 120, median 31.6 GiB, max 37.9. At 80, max 11.9. At 60, max 5.2. A 1 cm⁻¹ pitch halves each figure.
    - Fix options: `--mem=64G` with unchanged DOS settings, so it stays comparable with the production vDOS; or `EXTRA_ARGS="--mesh-length 60"`. Add `--reuse-fc` so the test structure's forces aren't recomputed.
  - **2026-11-17: resubmitted with option 1**, `--mem=64G` and `EXTRA_ARGS=--reuse-fc`. Everything else is as before.
    - **52692197**, `mlip-zw-test`: `--array=0`, `NUM_TASKS=90`, 1 h.
    - **52692210**, `mlip-zw`: `--array=0-14`, `NUM_TASKS=15`, 6 h, `afterok:52692197`.
    - Logs: `$SCRATCH/perovskite_phonon/logs/mlip-zw-test-52692197_0.*`.
  - **52692197 (test) COMPLETED** in 12 min 40 s. Cysteamine-PbCl2 is stable (soft-mode Pna2_1, −1.1 cm⁻¹).
    - Checked locally: S stays on the bridging X site (Pb–S 2.83–2.89 Å), NH3+ is intact, ΔV +0.3%. Its input was not expanded.
  - **2026-11-18, array task 52692210_0:** 5 of 6 done; marked FAILED because one structure failed. Ran 52 min, MaxRSS 45 GB, so the 64G request was needed.
    - **zw-15137414-PbCl** (NC(=[NH2+])CC[S-]): the full relaxation didn't reach fmax 0.001 in 1000 steps. No output.
    - The other four relaxed and are stable to within about −1.6 cm⁻¹. zw-167773 needed the soft-mode step: P1, 432 atoms, min 0.0 cm⁻¹.
    - **Structure check** (scratchpad `check_zw.py`, `oct.py`): every molecule is intact, with no proton transfer and no S–H. But the framework reconstructs:

      | Structure | ΔV vs input | S still bridging 2 Pb | Notes |
      |---|---|---|---|
      | zw-89561906 | −14% | 6/6 | Octahedra distorted; 6th Pb–X median 3.72 Å |
      | zw-29580 | −3% | 3/6 | |
      | zw-167773 | −9% | 3/6 | |
      | zw-23525521 | −2% | 0/6 | All S terminal: Pb–S 2.67 Å, 2nd Pb 4.29 Å; Pb effectively 4–5-coordinated. No longer a corner-sharing perovskite. |

      For comparison, cysteamine has regular octahedra (all Pb–X ≤ 2.99 Å).
    - Takeaway: in the expanded cells, thiolate prefers a short terminal Pb–S. Run a corner-sharing check (`src/data/perovskite_structure_check.py` or an S-bridging test) on every relaxed output before using its phonons.
  - **52692210_1** was preempted on owners after 8 min (07:07–07:15) and requeued, as expected. It is resumable; zw-6058-PbBr had already finished.
  - **2026-11-18 (orchestrator queue check, sacct only):** tasks 1 and 2 COMPLETED; tasks 3–14 PENDING (Priority). Outputs not inspected.
- **Next:**
  - Check the test's `.out`/`.err` in `$SCRATCH/perovskite_phonon/logs/mlip-zw-test-52574151_0.*`.
  - Expect the Cl cells to contract back. Check that each relaxed structure still has the S on its X site.
  - Caveat: the model is trained on inorganic data plus OMat. Organic and H-bond energetics are weaker, and there is no D3 (consistent with the MP sweep).
- Larger molecules would need a layered (2D) prototype; not started.

---

## 2026-10-22 – 2026-11-03: composition features, halide interpolation, GP vs trees, SQS noise check

Earlier parts of this session are summarized from its compaction notes.

### Done

**Clustering at k=2 and k=3** (`scripts/analysis/cluster_polymorphs.py --n-clusters {2,3}`)
- Output: `data/mlip_stable_polymorphs/clusters_k{2,3}`.
- k=2 has silhouette 0.615 and splits soft (Cl/Br/I plus part of the F) from stiff (oxides).
- k=10 remains the default `polymorph_clusters.csv`.

**Composition features** (new `src/analysis/composition_features.py`, `scripts/analysis/extract_composition_features.py`)
- Oxidation states come from charge-balance guesses, then bond valence, then all oxidation states, with a plausibility check. Radii are Shannon with a small fallback table.
- Each feature is NaN only when that feature itself can't be resolved.
- `x_frac_{F,Cl,Br,I}` gives each halide's share of the X site.
- Outputs:
  - `data/mp_perovskites/composition_features.csv` (3,949 rows). It matches the old `mp_composition_clusters.csv` exactly where both have a row.
  - `data/mixed_halides/analysis/composition_features.csv` (1,160 rows, no NaN).
- Within-family R² for composition-only inputs rose a lot: rel_centroid_B 0.39 → 0.95, s_300K −0.36 → 0.86.

**Halide interpolation section** in `descriptor_selection.ipynb`, using 993 kept mixed compositions in 231 series. The output is `data/mlip_stable_polymorphs/descriptor_halide_interpolation.csv`.
- **Test 1, MP → mixed:** trees trained on MP only get the framework offsets right but not the trend along x (R²(series) is negative).
- **Linear end-member baseline:** strong for the frequency-scale targets (s_300K 0.93), poor for the width and shape targets.
- **Test 2, held-out pair:** adding the other pairs' mixtures helps.
- **Test 3, held-out pure family:** adding mixed data bridges the families.

**GP vs trees** (`gp_vs_tree/`)
- **Model:** ARD-RBF + white-noise GP on the C+G inputs. Kernels are fitted on a 1,000-row subsample of each split's training set. The full run takes about 3.3 h on 12 cores.
- **Files:** scripts `gp_setup.py`, `gp_try.py` and `plot_gp_vs_tree.py`; results `gp_all.csv` and `pred_<target>.csv`; figures `fig_r2_by_test.png`, `fig_r2_scatter.png` and `fig_series.png`.
- **Results** (medians over 23 targets, tree → GP):

  | Test | Tree | GP | GP better on |
  |---|---|---|---|
  | MP within-family R² | 0.87 | 0.86 | 11/23 |
  | Held-out pair, R²(series) | −0.30 | 0.02 | 20/23 |
  | Held-out chloride + mixed | 0.49 | 0.80 | 22/23 |
  | Held-out Br/I + mixed | 0.68 | 0.79 | 17/23 |

  For s_300K and log_omega_log the GP reaches 0.95, matching the linear baseline. It beats the baseline where the mixtures bow (log_k_A/B, rel_centroid_A/B).
- **Width, MSD and shape targets** stay negative for every model.
- **Notebook:** GP section added as cells 31–34. It reads the saved results and does not refit.

**SQS disorder-noise check, set up**
- `scripts/dataset/select_noise_check_series.py` → `data/mixed_halides_noise_check_mp/series.csv`. It picks 23 series, stratified by halide pair and single vs double perovskite. K|Ca is excluded because its KCaCl₃ parent changed after the production set was generated.
- `generate_mixed_halides.py` has a new `--series-csv` option. With `--mode noise-check --source mp-clusters`, it produced 529 SQS cells for 115 compositions, 5 seeds each; symmetry-equivalent cells are pooled. Every nc-sqs0 cell is identical to its production structure.
- The data directory was copied to Sherlock: `$HOME/perovskite_phonon/data/mixed_halides_noise_check_mp/`.
- Submit command (checked with `sbatch --test-only`): `run_mlip_mp_array.sbatch` with `--array=0-11 --time=03:00:00`, `NUM_TASKS=12`, `OUT_ROOT=$SCRATCH/perovskite_phonon/mlip_phonons_mixed_noise_check_mp`.
- New `scripts/analysis/sqs_ensemble_average.py` computes manifest-weighted per-composition mean, std and sem for every descriptor, plus an averaged vDOS quantile function.
- **Notebook:** noise section added as cells 35–38. It computes the seed spread, along-series SNR, the R²(series) noise ceiling, and models re-scored against the ensemble mean. It skips itself until `data/mixed_halides_noise_check_mp/analysis/design_descriptors.csv` exists.
- Tested on the old CsPb noise check; the test CSV was removed afterwards.

**Correction about failed tasks.** The "FAILED" production mixed tasks were not memory failures; peak memory was 7–9 GB of 32 GB. Each task finished nearly all of its structures, but the script exits 1 if any single run fails. There were 7 such structures in all:
- 5 had MLIP relaxations that didn't reach fmax = 0.001 in 1000 steps.
- 2 failed Niggli reduction.

### State as of 2026-11-16 (checked at the start of the next session)

- The noise-check MLIP sweep has run: 528 of 529 runs have a `meta.json` under `mnt/mlip_phonons_mixed_noise_check_mp/`.
- **Post-processing not yet done:** `data/mixed_halides_noise_check_mp/analysis/` doesn't exist.
- `scripts/analysis/sqs_ensemble_average.py` and `scripts/dataset/select_noise_check_series.py` were modified on 2026-11-11 after this session wrote them. Not reviewed here; read them before relying on them.
- Nothing from this session is committed.

### Next

1. Post-process the noise check (the three steps in the `sqs_ensemble_average.py` docstring):
   - select with `--group-by run`;
   - extract descriptors over `mnt/`;
   - average.

   Then re-run the notebook; its noise section turns on by itself.
2. Depending on the result:
   - If the width and shape targets turn out to be disorder-limited, switch production to ensemble averages for those targets.
   - Otherwise, look at end-member polymorph mismatch.

### Open threads (not started)

- Modelling:
  - a joint multi-output model;
  - hyperparameter tuning;
  - a two-stage model (composition → geometry → phonons);
  - stoichiometry-weighted site averages for MP.
- Regenerate `chem_cluster`.
- Mixing energies.
- The failed and still-imaginary mixed structures.
- The DOS memory fix in Sherlock's `run_mlip_phonons.py`.
- Merge the Sherlock code back into the local tree.
- Update `descriptor_selection.md`.
- Commit the new work.
