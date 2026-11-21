# Descriptor selection for perovskite phonon inverse design

**Question.** Which phonon descriptors can an inverse-design model target, and which composition and structure descriptors predict them?

**Short answer.** The approach is viable, with a narrower scope than "design an arbitrary vDOS." Across 2,657 MLIP-relaxed, dynamically stable Materials Project perovskite polymorphs and 993 mixed-halide compositions:

- The phonon side has about **4–5 independent, learnable directions**: frequency scale, A-site rattling, B–X stretching-band separation, B-sublattice weight, and lattice instability.
- All five are well above MLIP noise and predictable within an anion family (within-family R² ≈ 0.75–0.96).
- The A-site and instability descriptors also transfer to an anion family the model never saw (R² 0.75–0.84).
- The absolute frequency scale does not transfer, but the anion is a discrete choice made by enumeration anyway.
- **Mixed halides** (Section 7) are harder. Along a halide series, the frequency scale is close to linear in x and is easy to predict. The widths and shapes bow, and no model yet predicts the bowing for a halide pair it hasn't seen. A Gaussian process (GP) does much better than trees there (Section 8). For some targets, one SQS cell per composition is itself a noise limit (Section 9).
- Finer spectral shape is only weakly predictable.
- Many compositions reach the same core-target values (Section 10). Most such pairs are trivial chemical neighbours, but a non-trivial set exists, mostly mixed-halide double perovskites.
- Every result rests on MLIP data. A prospective DFPT check is the decisive next step (see [Caveats](#caveats-and-next-step)).

Reproduce with `descriptor_selection.ipynb`. Its inputs are built by `scripts/analysis/extract_design_descriptors.py` and `scripts/analysis/extract_composition_features.py`. The descriptor definitions are in `src/analysis/design_descriptors.py` and `src/analysis/composition_features.py`. The notebook reads the GP results of Section 8 from `gp_vs_tree/` and does not refit them.

Numbers below are from the notebook run of 2026-11-18.

---

## 1. Data

| | |
|---|---|
| Structures | Materials Project perovskites that passed the corner-sharing-octahedra check (`src/data/perovskite_structure_check.py`) |
| Phonons | MACE-matpes-r2scan-omat-ft, finite displacements (phonopy), from `scripts/mlip/run_mlip_phonons.py`, **without** DFT-D3 (see Caveats) |
| Polymorph per formula | The lowest-energy dynamically stable state among *symmetric*, *rattled* and *soft-mode* relaxations (`select_stable_polymorphs.py`) |
| Kept (MP) | 2,657 of 2,762 polymorphs, 2,613 formulas: 1,617 oxides, 320 fluorides, 258 chlorides, 191 bromides/iodides, 271 other |
| Dropped | Formulas with no dynamically stable state, and the antiperovskites with no A/B sublattice |
| Mixed halides | 993 of 1,154 SQS compositions kept, in 231 series (Cl/Br 564, Br/I 218, Cl/I 211). 672 of them have both end members in the MP set (Section 7). |

All descriptors come from the saved force constants (`phonopy_params.yaml`), so no new MLIP calls are needed. Those files total 224 GB on Sherlock scratch. They are read with `mlip_phonons.load_compact`, which keeps only the unit-cell force-constant rows and gives bit-identical frequencies to `phonopy.load` at 10–70× the speed.

The "other" family includes intermetallic antiperovskites (a metal on the X site), 212 formulas in all. The notebook keeps them. `cluster_targets.py` (Section 10) drops them.

---

## 2. Metrics: phonon-side targets

Each atom is assigned to the **A**, **B** or **X** sublattice. X is the anion. Cations follow the structural check's site lists, or fall back to their effective coordination to X (ECoN ≥ 7.5 → A). Sublattices are unions of whole symmetry orbits, so every per-sublattice quantity below is exact on a symmetry-reduced q-mesh.

The mesh is a Γ-free Monkhorst–Pack grid with a q-point count proportional to 1/N_atoms. Mean-square displacements were checked against phonopy's own thermal displacements and agree exactly. The force-constant ω_rms matches the vDOS ω_rms to within 0.7% for 98% of polymorphs.

### Frequency scale

| Target | Definition | Why it matters |
|---|---|---|
| `log_omega_log` | ln of ω_log = exp⟨ln ω⟩ over the vDOS | The overall stiffness/mass scale. It controls high-temperature entropy and is the first thing any spectral target depends on. |
| `log_omega_max` | ln of the highest mode frequency | The bandwidth, set by the stiffest bond (B–X stretching). |
| `cv_300K`, `s_300K` | Harmonic heat capacity and entropy at 300 K, in k_B per atom | Thermodynamic summaries that weight the soft end of the spectrum. |

### Stiffness (mass-free)

| Target | Definition | Why it matters |
|---|---|---|
| `log_k_A`, `log_k_B`, `log_k_X` | ln of the on-site (Einstein) force constant k_s = ⅓ Tr Φ(j,j), averaged over sublattice s, in eV/Å² | The restoring force on one displaced atom. It contains **no mass**, which separates bonding from mass. It's exact: the BZ average of Σ_ν \|e_jν\|² ω_ν² equals Tr Φ(j,j)/m_j. |

### Per-sublattice spectral position (scale-free)

| Target | Definition | Why it matters |
|---|---|---|
| `rel_centroid_s` | Centroid of s's projected DOS ÷ the vDOS mean frequency | Where each sublattice sits in the spectrum, independent of overall scale. This is what makes it comparable across anion families. |
| `rel_width_s` | Standard deviation of s's projected DOS ÷ the mean frequency | How spread out a sublattice's modes are. A narrow A band is a localized, rattler-like mode. |
| `low_enrich_A` | Fraction of A's spectral weight in the lowest 10% of modes, ÷ 0.1 | A-site rattling. Values > 1 mean A dominates the soft end. Relevant to low thermal conductivity and anharmonicity. |
| `low_enrich_X` | The same, for X | Weight of octahedral tilt and rotation modes at the soft end. |
| `high_enrich_B` | Fraction of B's weight in the highest 10% of modes, ÷ 0.1 | How strongly B participates in the B–X stretching band. |

### Vibrational amplitude

| Target | Definition | Why it matters |
|---|---|---|
| `log_msd_s` | ln of the isotropic mean-square displacement U_iso of sublattice s at 300 K, in Å² | Weighted toward soft modes (~1/(mω²)). Connects to Debye–Waller factors, rattling and dynamic disorder. |

### Spectral shape (functional target)

| Target | Definition | Why it matters |
|---|---|---|
| `shape_pc1` … `shape_pc4` | PCA scores of the scale-normalized vDOS quantile function Q(p)/ω̄ (Section 2a) | A compact, scale-free representation of the whole spectrum. Together PC1–4 explain 92% of the shape variance (68%, 11%, 9%, 3%). |
| `rel_gap` | Largest gap in the vDOS ÷ bandwidth | Phonon band gaps matter for phonon–phonon scattering channels. |

PC1 has a step at p = 0.8, which is the B–X stretching band: 3 of every 15 modes in ABX₃ are stretching, so it occupies the top 20%. PC1 therefore measures how far the stretching band separates from the rest, and it tracks `rel_gap` and `rel_width_X`.

### Lattice instability

| Target | Definition | Why it matters |
|---|---|---|
| `log_distortion_energy` | ln(1 + E_symmetric − E_polymorph), in meV/atom: how far the structure relaxed below its symmetric MP parent | The depth of the double well the polymorph sits in. A proxy for anharmonicity and for how far the structure is from a phase transition. |
| `soft_mode_depth` | −min(ω) of the symmetric parent, in cm⁻¹, with 0 if the parent is stable | How unstable the parent is: the strength of the soft mode the polymorph condensed. |

### 2a. Spectral distance (Wasserstein-2)

Two spectra are compared through their quantile functions Q(p), the frequency below which a fraction p of the modes lie. The Euclidean distance between quantile functions is the Wasserstein-2 distance (up to a constant factor). It treats a shifted band as moving by its frequency difference, which a binned-DOS distance does not.

Writing Q = ω̄ · Q̃ splits W2² into a pure scale term and a shape term. On random polymorph pairs, **63%** of W2² is pure frequency scale. That's why the shape is modeled separately through `shape_pc*`.

---

## 3. Metrics: inputs, in tiers by what they cost to compute

The tiers matter for inverse design. Anything learnable from **C** can be screened over any enumerated composition for free. Anything that needs **G** requires one relaxation per candidate.

| Tier | Needs | Descriptors (count) |
|---|---|---|
| **M** | Formula | Site masses; reduced masses μ_BX, μ_AX; ln(m_A/m_X) (6) |
| **C** | + site assignment | From `composition_features.py`: Shannon radii r_A, r_B, r_X; Goldschmidt tolerance factor and octahedral factor; Bartel τ; Pauling electronegativities; oxidation states; site heterogeneity (number of species, mass coefficient of variation, oxidation-state spread on A and B); halide fractions `x_frac_{F,Cl,Br,I}` (29) |
| **C+G** | + one MLIP relaxation | BX₆ geometry: mean B–X length and its spread across octahedra, bond-length distortion, quadratic elongation, bond-angle variance, B off-centering. Tilt: mean and minimum B–X–B angle, tilt proxy (180° − ⟨θ_BXB⟩)/2. A site: ECoN, mean and minimum A–X distance, A–X spread. Realized tolerance factor t_geom = ⟨d_AX⟩/(√2⟨d_BX⟩). Bonding: bond-valence-sum deviation per sublattice, global instability index (GII). Volume per formula unit. (48) |
| **C+G+I** | + phonons of the parent structure | Distortion energy and soft-mode frequency of the symmetric parent and, where one exists (13% of rows), the cubic Pm-3m parent (52) |

**Composition features.**
- Oxidation states come from charge-balance guesses, then bond valence, then all oxidation states, with a plausibility check. Radii are Shannon with a small fallback table.
- Each feature is NaN only when that feature itself can't be resolved. Missing fractions: r_A 10%, tolerance factor 11%; 261 MP rows have no resolvable oxidation states.
- Moving to these features from the older per-site averages raised the composition-only within-family R² a lot, e.g. rel_centroid_B 0.39 → 0.95.

**X-site disorder features** (`X_DISORDER_FEATURES`, not yet in the notebook's C tier):
- x_mix = 1 − Σx², which is 2x(1−x) for a binary mix;
- mass_cv_X, the coefficient of variation of the halide masses;
- r_spread_X and en_spread_X, the spread of the halide radii and electronegativities.

All four are zero for a single halide and mean the same thing for every halide pair. They are under test in Section 8.

Notes:

- **M is not a "trivial mass" baseline.** Under formula-grouped CV, the masses act as element identifiers. That's how **M** reaches R² ≈ 0.94 even on the mass-free `log_k_A`. Read **M** as "element identity, no chemistry descriptors."
- **Bond-valence descriptors are missing for 17% of rows**, for elements with no Brese–O'Keeffe parameters (e.g. Ac). The gradient-boosted models handle these missing values natively.
- **The I tier is never used to predict the instability targets**, since they are built from the same calculations.

---

## 4. The tests

Each candidate target is run through the tests below. They answer different questions, and a target is only worth modeling if it passes all of them. Values are summarized in Section 5.

### Test 1: Noise floor (SNR)

**What.** SNR = σ_dataset / σ_models. The numerator is the target's spread across the 2,657 polymorphs. The denominator is its spread across eight MLIPs (MACE and ORB variants) on the same structure, taken as the median over structures.

**Why it matters.** A model trained on MLIP data can't resolve differences smaller than the MLIP's own uncertainty. If a target varies across the dataset by about as much as MLIPs disagree on a single structure, its "signal" is mostly model error, and a surrogate fitted to it would learn that error. SNR ≲ 3 is the rejection threshold.

**Two noise sets:**
- **Gauntlet** (primary): Pnma CsPbBr₃, with each model relaxing its own soft-mode structure. It's stable and tilted, and it includes each model's geometry error, so it represents the polymorphs actually being modeled.
- **Benchmark**: five cubic cells at the DFT geometry, all dynamically unstable. It's pessimistic for anything weighted toward the soft end, because there that end includes imaginary branches.

Where DFPT references exist, the MLIP-vs-DFPT RMS error on Cv and S is also reported: 0.14 and 0.19 of their dataset spread.

**Result.** Every target clears SNR ≥ 3 on the gauntlet. The weakest are `shape_pc2` (3.6) and `shape_pc4` (4.4). On the cubic benchmark, the soft-end descriptors drop below 3: `log_msd_A` 1.8, `low_enrich_A` 2.6. That reflects the unstable reference cells, not the stable polymorphs, but it's a reason to verify soft-end targets against DFPT on tilted cells.

For mixed halides there is a second noise source, the choice of SQS cell. Section 9 measures it.

### Test 2: Anion share (η²)

**What.** η² = 1 − SS_within / SS_total, the fraction of a target's variance explained by anion family alone (and separately by the composition clusters from `cluster_compositions.py`).

**Why it matters.** A target that's mostly "which anion" has only a handful of attainable values. A model will learn the anion and look accurate while saying nothing about design within a family. η² also determines which form of R² (Test 4) to trust.

**Result.** The absolute targets are **78–87% anion**: ω_log, ω_max, Cv, S, all `log_k_*` and all `log_msd_*`. The scale-free targets are mostly *not* anion: `rel_centroid_A` 0.28, `rel_width_A` 0.09, `shape_pc1` 0.19, `log_distortion_energy` 0.16, `soft_mode_depth` 0.03. That's the design-relevant variation.

The composition clusters are `cluster_compositions.py --features-csv` (2026-11-19), KMeans k=10 on the C descriptors. They label all but 323 of the 2,762 polymorphs; the unlabelled ones have no oxidation state. Cluster η² is 0.86–0.94 for the absolute targets and 0.12–0.57 for the scale-free ones. Regenerating the clusters from the 2026-09 set, which left 788 unlabelled, moved no value by more than 0.05.

### Test 3: Redundancy (Spearman ρ, participation ratio)

**What.**
- The Spearman rank correlation between every pair of targets, shown as a hierarchically clustered map.
- The participation ratio PR = (Σλ)² / Σλ² of the target correlation matrix's eigenvalues, an estimate of how many independent directions the targets span.

**Why it matters.** Targets in a block with |ρ| > 0.85 measure the same physics. Modeling all of them adds no design freedom and inflates any score summed over targets. The participation ratio sets an upper bound on how many phonon properties can be designed independently. Spearman rather than Pearson is used because the relationships are monotonic but not linear.

**Result.** PR = **4.6 effective dimensions out of 26 targets**. The clustered map shows the blocks:
- frequency scale (ω_log, S, Cv, all `log_msd_*`, `log_k_A`);
- stiffness (`log_k_B`, `log_k_X`, ω_max);
- A-site position and width;
- stretching-band separation (`shape_pc1`, `rel_width_X`, `rel_gap`);
- B-sublattice weight (`rel_centroid_B`, `rel_width_B`, `high_enrich_B`, anticorrelated with `rel_centroid_X`);
- instability (`log_distortion_energy`, `soft_mode_depth`).

### Test 4: Learnability (three forms of R²)

**What.** Gradient-boosted trees (HistGradientBoostingRegressor) predict each target from each input tier. Three scores come out:

| Score | How | Question it answers |
|---|---|---|
| **R²** | 5-fold `GroupKFold`, grouped by formula | Can the model interpolate to new formulas? Grouping keeps polymorphs of one formula from sitting on both sides of a split. |
| **R²w** (within-family) | The same CV predictions, scored against each anion family's own mean | Can it rank compounds *within* an anion family? This excludes the between-family variance that η² showed dominates absolute targets. |
| **R²(held-out anion)** | Train on three anion families, predict the fourth (oxide, fluoride, chloride, bromide/iodide), scored against that family's own variance, median over families | Can it extrapolate to chemistry outside its training distribution? This is what inverse design actually asks of a model. |

**Why it matters.** The forward model is the surrogate the inverse search runs on, so a target is only designable if it's predictable. The three scores test increasingly hard settings, and they disagree in informative ways:
- Plain R² is near 1 for almost everything, because it mostly rewards knowing the anion.
- R²w is the honest number for design within a family.
- Held-out-anion R² says whether the learned relationship is physics or a lookup table.
- The gap R²w(C+G) − R²w(C) (`geometry_gain_within`) is what one relaxation per candidate buys.

**Result.**
- **Within a family**, most targets are learnable: R²w(C+G) ≥ 0.8 for 17 of 26. Composition alone gets most of the way there, and geometry adds ≤ 0.11. The exceptions are `log_distortion_energy` (+0.16) and `soft_mode_depth` (+0.14), which depend on the realized polymorph.
- **Across anions**, only the A-site and instability descriptors transfer well:

  | Target | Held-out-anion R² |
  |---|---|
  | `log_distortion_energy` | 0.84 |
  | `rel_width_A` | 0.81 |
  | `rel_centroid_A` | 0.75 |
  | `low_enrich_A` | 0.43 |
  | `log_msd_X` | 0.42 |
  | `rel_width_B` | 0.40 |
  | `rel_centroid_B` | 0.37 |
  | `shape_pc1` | 0.36 |
  | `rel_width_X` | 0.33 |

  The absolute frequency targets fail outright: ω_log −1.08, S −0.96, Cv −2.97, ω_max −1.61. The model can't predict the scale of an unseen anion family. `log_k_A` (0.51 in the earlier 1,984-polymorph run) is now −0.42. The held-out oxide fold, at −2.76, drives that drop.
- **Weak:** `shape_pc2`–`pc4`, `rel_gap` and `soft_mode_depth` (R²w 0.59–0.72). Spectral detail beyond PC1 isn't well captured by these descriptors.

### Test 5: Invertibility (ρ_inv)

**What.** For each target, take each polymorph's 10 nearest neighbors in that target's value. Then compute ρ_inv as the median distance between those neighbors in the standardized **C**-descriptor space, divided by the median distance between random pairs.

**Why it matters.** Inverse design maps a target value back to chemistry, so it matters whether that map is narrow or broad.
- **ρ_inv ≪ 1:** a target value pins down the chemistry, giving a near-unique answer.
- **ρ_inv ≈ 1:** many distinct chemistries reach the same value. That's the "multiple synthetic routes" property this project looks for, but it means the inverse problem returns a broad family rather than a single answer.

Read it together with learnability. Learnable *and* ρ_inv near 1 means the forward model has found structure-property rules that cut across composition-space distances.

**Result.**
- No target localizes the chemistry strongly (ρ_inv 0.69–0.93).
- The anion-dominated absolute targets localize most: `log_k_A`, ω_log, S and Cv are all 0.69.
- The scale-free targets are broad: A-site descriptors 0.88–0.90, `shape_pc*` 0.88–0.93. A given spectral shape is reachable from many chemistries.
- Section 10 asks the multi-target version of this question.

### Test 6: Feature attribution (permutation importance)

**What.** For targets that pass Tests 1 and 4 (SNR ≥ 3, R²w ≥ 0.5), compute the permutation importance of each C+G descriptor on a held-out 20% split grouped by formula.

**Why it matters.** This identifies the structural side of the eventual model: which descriptors carry the phonon information, and so which ones a generative or search step must control. It's also a physics sanity check. Importances that don't make physical sense would indicate leakage or an artifact.

**Result** (from the earlier 1,984-polymorph run; the current notebook draws the figure but this table hasn't been re-read from it). The attributions are physically sensible:

| Target | Main descriptors | Interpretation |
|---|---|---|
| `rel_centroid_A`, `low_enrich_A` | ln(m_A/m_X) | The A/X mass ratio sets where the A band sits |
| `rel_width_A` | A-site ECoN | Coordination sets how localized the A mode is |
| ω_log, S | ⟨d_AX⟩ and μ_BX | Bond length and reduced mass set the scale |
| `log_k_A` | X oxidation state | Anion charge sets the A-site restoring force |
| `shape_pc1`, `rel_width_X` | A electronegativity, octahedral factor, A-site bond-valence deviation | These set how far the stretching band separates |

### Shortlist

The notebook builds a shortlist greedily by R²w(C+G). It skips any target with SNR < 3 or R²w < 0.5, and any target with |ρ| > 0.85 to one already picked. At that threshold it keeps 20 targets, which is too many given PR ≈ 4.6. The recommended core set (Section 6) takes one target per correlation block instead.

---

## 5. Scorecard (key targets)

From `data/mlip_stable_polymorphs/descriptor_scorecard.csv`. Columns:
- **SNR**: gauntlet / benchmark noise sets.
- **η²**: anion share.
- **R²w C / C+G**: within-family R² from composition, and with geometry added.
- **R² held-out**: held-out-anion R², C+G tier.
- **ρ_inv**: invertibility.

| Target | SNR | η² | R²w C | R²w C+G | R² held-out | ρ_inv |
|---|---|---|---|---|---|---|
| `log_omega_log` | 11.8 / 7.6 | 0.87 | 0.86 | 0.93 | −1.08 | 0.69 |
| `s_300K` | 11.1 / 8.6 | 0.87 | 0.86 | 0.94 | −0.96 | 0.69 |
| `log_k_A` | 8.9 / 4.9 | 0.83 | 0.80 | 0.89 | −0.42 | 0.69 |
| `rel_centroid_A` | 10.4 / 3.7 | 0.28 | 0.91 | 0.93 | 0.75 | 0.89 |
| `rel_width_A` | 6.0 / 3.2 | 0.09 | 0.83 | 0.92 | 0.81 | 0.90 |
| `low_enrich_A` | 16.9 / 2.6 | 0.33 | 0.87 | 0.88 | 0.43 | 0.88 |
| `shape_pc1` | 7.2 / 4.2 | 0.19 | 0.85 | 0.88 | 0.36 | 0.88 |
| `rel_width_X` | 5.1 / 5.7 | 0.37 | 0.81 | 0.87 | 0.33 | 0.85 |
| `rel_centroid_B` | 19.8 / 9.3 | 0.65 | 0.95 | 0.96 | 0.37 | 0.77 |
| `high_enrich_B` | 15.5 / 14.3 | 0.71 | 0.89 | 0.91 | −0.30 | 0.77 |
| `log_omega_max` | 7.7 / 13.5 | 0.83 | 0.82 | 0.85 | −1.61 | 0.73 |
| `log_k_B` | 5.9 / 6.0 | 0.81 | 0.75 | 0.81 | −0.90 | 0.75 |
| `log_distortion_energy` | — | 0.16 | 0.59 | 0.75 | 0.84 | 0.88 |
| `soft_mode_depth` | — | 0.03 | 0.45 | 0.59 | 0.13 | 0.92 |
| `rel_gap` | — | 0.21 | 0.62 | 0.62 | 0.13 | 0.90 |
| `shape_pc2` | 3.6 / 3.7 | 0.27 | 0.65 | 0.72 | −0.06 | 0.90 |

A dash means no SNR: the instability and gap targets can't be computed for the noise-set structures, so they pass Test 1 by default.

---

## 6. Recommended core targets

One target per independent direction, matching the participation ratio:

| Direction | Target | Role in design |
|---|---|---|
| Frequency scale | `log_omega_log` | Set mostly by anion choice. Enumerate the anion, then the model works within that family. Near-linear across halide mixtures (Section 7). |
| A-site rattling | `rel_centroid_A` and/or `rel_width_A` | The most learnable and best-transferring axis, controlled by the A/X mass ratio and A-site coordination. In mixed halides, `rel_width_A` is disorder-limited at one SQS per composition (Section 9); prefer `rel_centroid_A` there. |
| B–X stretching separation | `shape_pc1` | The dominant spectral-shape mode, related to phonon gaps. |
| B-sublattice weight | `rel_centroid_B` | Where B sits in the spectrum. |
| Lattice instability | `log_distortion_energy` | Needs a relaxation per candidate, but transfers across anions. |

This implies a **two-stage forward model**: composition → relaxed geometry (tilt, ECoN, bond-valence deviation) → phonon targets. Composition alone suffices for most targets. The geometry stage pays off for the instability targets and the B-site band width.

---

## 7. Mixed halides: does the model fill in Cl–Br–I mixtures?

**Data.** `scripts/dataset/generate_mixed_halides.py --source mp-clusters` takes every MP cation framework (fixed A and B species) that has at least two halide end members. It mixes each pair at a grid of fractions x as SQS supercells (≥ 24 halide sites), starting from the majority end member's MP structure at the Vegard volume. The 993 kept compositions form 231 series: one framework and one halide pair, varied in x.

**Score.** R²(series) counts only how a target changes *along* a series: residuals are taken against each series' own mean. It's the strict number for a model meant to tune x. The linear baseline interpolates linearly in x between the two end members' MP values (672 compositions have both).

**Tests and results** (trees unless noted; `descriptor_halide_interpolation.csv`):

| Test | Result |
|---|---|
| MP only → mixed | Plain R² is fine (0.4–0.9 for most targets), but R²(series) is negative for every target except `low_enrich_X` (0.31). Trained on MP alone, the model gets each framework's offset right and the x-trend wrong. |
| Linear end-member baseline | Strong for the frequency scale: s_300K and ω_log 0.93, Cv 0.90, rel_centroid_X 0.81. Poor or negative for the widths, shapes and `log_msd_*`, which bow. |
| Held-out pair, + the other pairs' mixtures | Training on other pairs' mixtures helps most targets (e.g. s_300K −0.62 → 0.62), but `shape_pc1`, `rel_width_X`, `log_omega_max` and `high_enrich_B` stay far below zero. |
| Held-out pure family, + mixed | Adding mixtures bridges the families. Held-out Br/I improves on most targets, e.g. s_300K −1.78 → 0.92 and rel_centroid_B −0.16 → 0.90. |

**Why the series fail** (diagnostic, 2026-11-18):
- Within-series variation is a small part of the total (std 0.2–0.47 of the dataset's), so any error in the framework offset swamps it.
- The poor targets bow. A straight line in x explains 42–63% of their within-series variance, a quadratic 85–91%.
- Holding out a whole halide pair is much harder than holding out a series with the pair seen. Trees on rel_centroid_B score 0.51 on held-out series, −0.60 on a held-out pair. On ω_max the drop is −0.55 → −6.73, on rel_width_X −0.48 → −7.96. A held-out pair puts `x_frac_*` outside the training range.
- The C tier has no feature that is the same for every pair and peaks at x = ½. The X-site disorder features (Section 3) are meant to fill that gap; Section 8 tests them.

---

## 8. Gaussian process vs trees

**Model** (`gp_vs_tree/gp_try.py`, results in `gp_all.csv`): ConstantKernel × ARD-RBF + WhiteKernel on the standardized C+G inputs. Hyperparameters are fitted on a 1,000-row subsample of each training set, then the GP is conditioned on the whole set.

Medians over 23 targets:

| Test | Tree | GP | GP better on |
|---|---|---|---|
| MP within-family R²w | 0.87 | 0.86 | 11/23 |
| MP only → mixed, R²(series) | −1.22 | −1.52 | 12/23 |
| Held-out pair + other mixtures, R²(series) | −0.30 | 0.02 | 20/23 |
| Held-out chloride + mixed | 0.49 | 0.80 | 22/23 |
| Held-out Br/I + mixed | 0.68 | 0.79 | 17/23 |

- On the MP scorecard the two models are equivalent. Neither learns the x-dependence from MP alone.
- With mixtures in training, the GP's smooth interpolation pays off. On the held-out pair, s_300K and ω_log reach 0.95, matching the linear baseline. The GP beats the baseline where the mixtures bow: log_k_A 0.79 vs −1.74, rel_centroid_B 0.64 vs −1.09.
- Still negative for every model on a held-out pair: `rel_width_X` (GP −3.1), `log_omega_max` (−2.9), `rel_width_B`, `high_enrich_B`, `shape_pc1`–`pc3`.

### 8a. X-site disorder features and a bowing-only target don't help (2026-11-20)

`gp_vs_tree/gp_disorder.py` (Sherlock array 52985751) fits three variants, each with a GP and with trees, for 9 targets:

- **base:** the C+G inputs;
- **+ disorder:** the X-site disorder features added (x_mix, mass_cv_X, r_spread_X, en_spread_X);
- **bowing-only:** the same features, with the target taken relative to the end-member line, predicting only the deviation from it.

They are scored two ways: on a held-out halide pair, and on held-out series where every pair is seen in training (GroupKFold(5)). All variants are scored on the same 672 compositions that have both end members. Output: `gp_vs_tree/disorder/gp_disorder.csv`. Figure: `gp_vs_tree/fig_gp_disorder.png` (`plot_gp_disorder.py`).

R²(series), GP:

| target | pair: line | pair: base | pair: + disorder | pair: bowing-only | series: base | series: + disorder | series: bowing-only |
|---|---|---|---|---|---|---|---|
| log_omega_log | 0.93 | 0.95 | 0.91 | 0.92 | 0.96 | 0.96 | 0.59 |
| log_k_A | −1.74 | 0.79 | 0.79 | −1.63 | 0.82 | 0.82 | −15.4 |
| rel_centroid_A | 0.19 | 0.71 | 0.70 | 0.18 | 0.87 | 0.87 | −1.75 |
| rel_centroid_B | −1.09 | 0.71 | 0.30 | −0.91 | 0.87 | 0.86 | −2.51 |
| rel_width_A | −3.44 | 0.07 | −0.01 | −2.96 | 0.31 | 0.30 | −9.05 |
| log_msd_A | −2.14 | −0.07 | 0.03 | −1.86 | 0.34 | 0.35 | −18.5 |
| shape_pc1 | −2.77 | −0.08 | −0.04 | −3.54 | 0.36 | 0.35 | −8.75 |
| log_omega_max | −7.20 | −2.61 | −5.66 | −5.93 | −0.16 | −0.10 | −2.02 |
| rel_width_X | −8.78 | −3.19 | −13.9 | −7.75 | 0.33 | 0.35 | −14.8 |

How to read R²(series): it compares the error with the variation *within* a series (along x at fixed framework), not with the spread of the whole dataset. That variation is small. For rel_width_X its std is 0.020, against a base-GP error of 0.025–0.054 on a held-out pair, so R² = −3 means an error about twice the within-series spread. It is not a catastrophic error on the scale of the dataset (std 0.12).

**+ disorder features: no gain, and losses on a held-out pair.** On held-out series the R² change is ≤ 0.02, so the features carry nothing the base inputs lack. On a held-out pair they hurt rel_width_X (−3.2 → −13.9), rel_centroid_B (0.71 → 0.30) and log_omega_max (−2.6 → −5.7). Reason, panel (c) of the figure: all 2,657 MP rows have disorder features of exactly 0, and each pair occupies its own interval. mass_cv_X, for example, is 0.15–0.23 for Br/I, 0.23–0.42 for Cl/Br and 0.31–0.67 for Cl/I. So the features act as a *pair label*. A held-out pair always lies outside the training range, and the GP extrapolates a pair-specific offset. For rel_width_X with Cl/Br held out, that offset is a mean bias of −0.064, three times the within-series std. They would only work with a mechanism that is the same function of the features across pairs, and with three pairs the GP can't learn one.

**Bowing-only target: worse everywhere.** Two causes, both checked:
1. *The mixtures don't sit on the end-member line, even at x = 1/6.* The mean offset from the line is larger than the variation along the series: for rel_width_X, median |offset| 0.035 against a within-series std of 0.020. It grows with halide mismatch: the mean offset is 0.007 for Br/I, 0.038 for Cl/Br and 0.097 for Cl/I, and the same ordering holds for ω_max and shape_pc1. Part of the offset is structural: the end member is a relaxed MP polymorph, while the mixtures are rattled SQS supercells. Predicting "line + bowing" therefore means predicting this offset for an unseen pair, the same extrapolation as above.
2. *On held-out series the predicted bowing is noise.* Its spread is 2–2.5× the true bowing (log_k_A: 0.30 vs 0.12; rel_width_X: 0.084 vs 0.044), with correlation 0.01 (log_k_A), 0.1 (shape_pc1) and 0.4 (rel_width_X). The bowing is a small difference of two noisy values. A likely contributor, not tested: the 2,657 MP rows enter with bowing exactly 0, so the GP's fitted noise level reflects them rather than the mixtures. Predicting the full value, as base does, is the better parameterization.

**Upshot.** On held-out series the base GP is already good, with R² 0.8–0.96 for frequency scale, A-site and B-site targets. The remaining failures are transfer to an unseen pair, and within-series targets (width, MSD, shape PC1) stuck at 0.3–0.35. The first is an extrapolation problem that composition features can't fix with three pairs. The second could be the single-SQS label noise (Section 9), which the 5-cell ensemble tests (Section 9b).

---

## 9. Disorder noise: is one SQS cell per composition enough?

Each production mixture is one SQS realization. A different cell of the same composition gives slightly different phonons, a noise source the MLIP gauntlet doesn't see.

**Set-up.** `select_noise_check_series.py` picked 23 series, stratified by halide pair and single vs double perovskite. There are 115 compositions with 5 SQS seeds each, 528 finished runs and 509 stable realizations. Seed 0 is the production cell: it reproduces the production values exactly (`sqs0_repro` = 0). `sqs_ensemble_average.py` averages the realizations per composition. Results: `descriptor_sqs_noise.csv`.

**Measures.**
- **seed_std:** the spread across seeds.
- **SNR_series:** the along-series spread ÷ seed_std.
- **Ceiling:** the best R²(series) any model could reach against a single noisy SQS value.
- Each model is scored against the single cell and against the ensemble mean.

**Results.**

| Class | Targets (ceiling) |
|---|---|
| Not disorder-limited (≥ 0.96) | rel_centroid_X, Cv, ω_log, S, log_k_X |
| Mild (0.85–0.93) | low_enrich_X/A, shape_pc1–4, log_k_B, high_enrich_B, rel_centroid_A/B |
| Disorder-limited (SNR_series ≤ 1.9) | ω_max 0.73, rel_width_X 0.77, rel_width_A 0.62, rel_width_B 0.67, log_msd_B 0.62, log_msd_X 0.43, log_msd_A 0.35, log_k_A 0.28 |

- On these 115 compositions, scoring against the ensemble mean barely changes any model's R². The full ensemble (Section 9b) shows otherwise. Where models fall short, the x-dependence is unlearned, not drowned in noise. Most targets have models well below their ceiling.
- The GP beats trees and the linear baseline on almost every target here too.
- For the disorder-limited targets, production would need ensemble averages over several SQS cells before a model could be scored meaningfully along x.
- **Seed noise vs MLIP noise** on the core targets: ω_log 0.29×, rel_centroid_A 1.02×, rel_width_A 0.78×, shape_pc1 0.59×, rel_centroid_B 1.47×. Disorder noise is comparable to the MLIP noise, and larger for rel_centroid_B.

**How many SQS cells make it negligible?** `scripts/analysis/sqs_convergence.py` → `sqs_convergence.csv`, `sqs_convergence_by_pair.csv` (in the noise check's `analysis/`).
- **Model:** the error of an N-cell average is σ_seed/√N, with σ_seed from the 5 seeds. "Negligible" means both criteria hold:
  - σ_seed/√N ≤ ½ × the gauntlet MLIP noise, so disorder adds ≤ 12% to the model noise;
  - the R²(series) ceiling is ≥ 0.95, against the within-series variance of all production series. That variance is larger than for the 23 noise-check series alone, so these N = 1 ceilings sit above the table above.
- **N needed**, pooled over compositions, max of the two criteria:

  | N | Targets |
  |---|---|
  | 1 | ω_log, S, rel_centroid_X |
  | 2–4 | Cv, log_k_X, log_k_B, shape_pc2–4 |
  | 5 | rel_centroid_A, shape_pc1, ω_max, low_enrich_X |
  | 7 | rel_width_A/B/X |
  | 9 | rel_centroid_B, high_enrich_B, log_msd_B |
  | 11–14 | low_enrich_A, log_k_A, log_msd_A/X |

- **Core targets:** ω_log 1, rel_centroid_A 5, shape_pc1 5, rel_width_A 7, rel_centroid_B 9. At N = 5 their ceilings are ≥ 0.94, and only rel_centroid_B's disorder stays above half the MLIP noise (0.65×).
- **Per pair:** Cl/Br converges fastest, at ≤ 4 cells for most targets. Br/I needs 8–15 for the shape and width targets. Cl/I needs about 20 for the B-site targets.
- **The 90th-percentile composition** needs 2–3× more. That figure is from 5-seed variances, so it's a conservative bound.
- **Caveats:**
  - The 40-atom cells have few distinct arrangements: only 62 of 115 compositions gave 5 distinct cells from 5 seeds. Averaging more seeds than there are distinct cells converges to that cell size's ensemble, not to the random alloy.
  - Every seed uses the same supercell, so the finite-cell bias is not measured. It needs a larger supercell.

### 9b. Full SQS ensemble: re-scoring the series models (2026-11-20)

**Set-up.**
- **Ensemble:** 5 SQS cells for every Cl/Br and Br/I composition, from array 52996336 plus 1,200 reused runs (`data/mixed_halides_ensemble/reuse.csv`). Cl/I was skipped.
- **Pipeline:** `scripts/analysis/postprocess_sqs_ensemble.sh` links the runs, selects and describes them, averages per composition and re-scores. Output: `data/mixed_halides_ensemble/analysis/`.
- **Coverage:** 3,709 stable cells over 885 compositions. 614 compositions have all 5 cells stable, and 782 of the 993 production compositions matched.
- **Scored rows:** compositions with ≥ 3 stable cells, in series with ≥ 2 such rows. That leaves 530 compositions in 118 series (375 Cl/Br, 155 Br/I).
- **Scoring:** `sqs_ensemble_rescore.py` scores each set of held-out-series predictions on the same rows two ways: against the production cell and against the 5-cell mean. The bracket is a series-bootstrap 95% CI on the change.
- **Noise test (`tracks_cell`):** the slope of (prediction − mean) on (production cell − mean). A slope of 0 means the model predicts the mean. A slope of 1 means it predicts that cell's own disorder noise.

**Results.**

1. **The existing models predict that cell's noise, not the composition.** The base series GP has `tracks_cell` 0.6–1.0 on most targets. So its R² drops when it is scored against the mean (`figures/rescore_r2.png`):

   | target | GP vs production cell | GP vs 5-cell mean | ΔR² 95% CI | ceiling |
   |---|---|---|---|---|
   | ω_log | 0.95 | 0.94 | [−0.01, 0.00] | 1.00 |
   | rel_centroid_A | 0.85 | 0.62 | [−0.38, −0.09] | 0.95 |
   | rel_centroid_B | 0.84 | 0.68 | [−0.25, −0.08] | 0.97 |
   | shape_pc1 | 0.25 | −0.11 | [−0.53, −0.21] | 0.92 |
   | rel_width_A | 0.17 | −0.15 | [−0.57, −0.15] | 0.89 |
   | rel_width_X | 0.20 | −0.12 | [−0.54, −0.18] | 0.93 |
   | log_k_A | 0.78 | −0.18 | [−1.56, −0.62] | 0.79 |
   | log_msd_A | 0.31 | −0.31 | [−1.06, −0.28] | 0.74 |

   - **Ceiling:** the expected R² of a perfect predictor against the 5-cell mean, given the averaging error that remains.
   - **How the GP sees the noise:** the GP never sees the held-out cell's own label. It reads that cell's noise from the cell's own geometry, the G-tier features.
   - **Why the model has learned this:** trained on single-cell labels, it learns how geometry noise maps to label noise.
   - **Check:** a tree refit on ensemble labels but the same single-cell geometry has `tracks_cell` ≈ 0 (0.01 for log_k_A, against 0.78 for the base tree). It ignores geometry noise that no longer predicts the label.
   - **No leakage:** no formula appears in more than one series.
   - **This corrects §9.** That section said the ensemble re-scoring barely changes the R², but that came from 115 compositions; the full set shows the drop clearly.

2. **Refitting on averaged labels and averaged geometry helps the trees, but not where it matters.** These are tree-only refits, scored against the 5-cell mean (`gp_vs_tree/disorder_*`):

   | target | base tree | averaged labels + averaged geometry | C tier, averaged labels | bowing-only tree (single) |
   |---|---|---|---|---|
   | rel_centroid_A | 0.51 | 0.61 | 0.37 | 0.60 |
   | rel_centroid_B | 0.37 | 0.56 | 0.34 | 0.65 |
   | log_k_A | −0.21 | 0.45 | −1.47 | −1.12 |
   | log_msd_A | −0.34 | −0.01 | −2.08 | −0.87 |
   | rel_width_A | −0.61 | −0.38 | −1.91 | −0.64 |
   | rel_width_X | −1.31 | −1.03 | −2.31 | −0.60 |
   | shape_pc1 | −2.16 | −2.18 | −3.27 | −0.55 |

   - **C tier:** composition-only features are far worse. Geometry carries real signal, not just noise.
   - **Widths and shape:** they stay below zero under every tree variant, against ceilings of 0.89–0.93. These targets are unlearned along x, not noise-limited.
   - **Bowing-only tree:** it is the best tree on shape_pc1 and rel_width_X (it was the worst GP variant in §8a). Its ensemble refit hasn't been run.

3. **Br/I is the hard pair.** It does worst on both B-site targets: for the base series GP, rel_centroid_B goes from 0.55 to −0.67 against the mean, and log_k_A goes from 0.51 to −0.76 (ceiling 0.61). Cl/Br is better on both.

**Upshot.**
- **Production R²:** the single-cell R² values in §8–8a for A-site, B-site and stiffness targets are partly disorder noise.
- **The decisive test is the GP refit on averaged labels and geometry.** `gp_disorder.py --labels ensemble --geometry ensemble` is ready and needs Sherlock.
- **What the trees suggest for widths and shape:**
  - averaging won't lift them above zero;
  - a GP trained to predict the mean may hold onto ω_log and the centroids and recover log_k_A.

---

## 10. Target-space redundancy: which compositions give the same targets?

`scripts/analysis/cluster_targets.py` clusters compositions on the six core targets of Section 6 (output in `data/target_clusters/`).

**Data.** The MP polymorphs whose X site is O, S, Se, Te, N, F, Cl, Br or I (intermetallic antiperovskites dropped), plus the mixed halides: 3,438 polymorphs, 3,394 formulas.

**Units.** Each target is divided by its gauntlet MLIP noise. `log_distortion_energy` has no gauntlet value, so its noise is set by assumption, to std ÷ the other targets' median SNR (9.3).

**Method.** Complete linkage with the Chebyshev metric, cut at tol = 1, 2 and 3 noise units, so every pair in a cluster is within tol on every target. **Chemical spread** is the median pairwise distance in standardized C descriptors ÷ the random-pair median.

| tol | Formulas with a partner | Redundant clusters | Max formulas per cluster | Clusters with spread > 0.5 |
|---|---|---|---|---|
| 1 | 19% | 281 | 5 | 1 |
| 2 | 51% | 659 | 11 | 28 |
| 3 | 75% | 854 | 16 | 63 |

- Most redundancy is chemically trivial (median spread 0.14): rare-earth swaps in oxides, and different orderings or supercells of one parent.
- The non-trivial cases are mostly mixed-halide double perovskites, e.g. Cs2AgSb(I2Cl)2 ≈ CsEuBr3. At tol 2, 11 of the 28 are oxides, e.g. TiCdO3 ≈ La10MgTi7O30.
- `log_omega_log` keeps clusters mostly inside one anion family. Drop it to look for scale-free redundancy across anions.
- The noise unit is measured on CsPbBr₃ only. For mixed halides, SQS seed noise (Section 9) is the same order as the MLIP noise unit, so their clusters at tol 1 aren't reliable.

---

## 11. Mixing energies: which mixtures would actually form?

`scripts/analysis/mixing_energies.py` writes `<set>/analysis/mixing_energies.csv`, one row per SQS run. With x the X2 share of the X site, taken from the cell's actual formula:

- ΔE_mix = E(mix) − (1−x)·E(X1 end) − x·E(X2 end), the MLIP static energy in meV/atom. Mixtures and end members have the same atoms per formula unit, so per-atom energies compare directly.
- Each end member's reference is its lowest-energy dynamically stable state over all MP entries of its formula. A second column uses the specific MP entry the mixture was built from; the two agree to within 0.1 meV/atom for every stable row.
- ΔF_mix(T) = ΔE_mix + ΔF_vib − T·ΔS_conf.
  - ΔS_conf = −f_X[x ln x + (1−x) ln(1−x)] is the ideal random-alloy entropy, with f_X = 3/5 X sites per atom.
  - ΔF_vib is the harmonic vibrational free energy (including zero point), integrated over the saved vDOS quantile functions. The same quadrature reproduces the exact ΔS_vib to a median of 0.0002 k_B/atom.
- Along each series, the height above the lower convex hull (end members at 0) says whether a mixture would decompose into its neighbours.

Production set, 689 dynamically stable mixtures with both references (meV/atom, 300 K):

| Pair | n | ΔE_mix median | ΔE_mix > 0 | T·ΔS_conf median | ΔF_vib median | ΔF_mix < 0 | On the ΔF hull |
|---|---|---|---|---|---|---|---|
| Cl/Br | 390 | 5.2 | 96% | 9.9 | −0.6 | 99% | 71% |
| Br/I | 155 | 6.7 | 92% | 9.9 | −0.1 | 85% | 57% |
| Cl/I | 144 | 22.7 | 100% | 9.9 | −1.2 | 6% | 5% |

- Nearly every mixture costs energy at 0 K, scaling with the halide size mismatch. At 300 K, configurational entropy stabilizes most Cl/Br and Br/I mixtures, but almost no Cl/I ones: those would phase-separate. That's consistent with the size-mismatch screen in `generate_mixed_halides.py` (median radius mismatch Cl/I 0.22, Br/I 0.12, Cl/Br 0.08).
- Vibrational entropy helps slightly (mixtures are softer), but it is about 9× smaller than T·ΔS_conf (median ratio).
- **SQS seed spread** (noise-check set, 115 compositions): the std of ΔE_mix is 0.7 meV/atom median, 3.6 at p90. That's small against the Cl/I values, comparable to the Cl/Br margin.
- **ΔE_mix < 0 mostly flags a bad reference**, not favourable mixing. In 19 of 689 rows it is below −2 meV/atom, mostly at x = 0.83 with a single minority halide. There, the rattled 40-atom SQS cell found a distortion the 10-atom end-member cell can't hold. For example, Rb₂AgAuBr₆ is counted as stable in Fm-3m at −0.98 cm⁻¹, inside the 1 cm⁻¹ tolerance. The CSV carries the cell sizes and states, so these rows can be filtered.
- For design, ΔF_mix(300 K) and its hull height act as a filter: a proposed mixed composition is only useful if it doesn't phase-separate.

---

## 12. Prospective validation: proposal set and retrospective check (2026-11-19/20)

**Setup** (`scripts/analysis/propose_candidates.py`). A design target is a vector of values on the five core axes (ω_log, rel_centroid_A, rel_width_A, shape_pc1, rel_centroid_B). A composition **hits** the target when *every* axis is within ±tol × that axis's MLIP noise (Chebyshev distance ≤ tol). The forward model is one GP per axis on C-tier inputs (composition and site assignment only), because a never-computed candidate has no relaxed geometry. For each candidate the GP gives a predictive N(μ, σ²) per axis, and candidates are ranked by Σ log P(|y − t| ≤ 3 × noise), which rewards both closeness and confidence.

**Retrospective check: the same procedure on known answers.** Each of the 10 chemical clusters is held out in turn:
1. Train on the MP polymorphs outside it.
2. Draw 6 members of the held-out cluster as targets.
3. Rank the rest of the cluster and take the top 8 (distinct formulas, the target's own formula excluded).
4. Compare the top 8's hit rate with that of a random cluster member (the base rate).

That makes 60 target draws. Output: `data/proposals/retrospective/`; figure `fig_retrospective.png` (`scripts/analysis/plot_retrospective.py`).

| tol (noise units) | random member hits | top-8 hits | enrichment |
|---|---|---|---|
| ±2 | 1.3% | 6.2% | 4.7× |
| ±3 | 3.6% | 18.3% | 5.1× |
| ±5 | 12.0% | 46.5% | 3.9× |

- **The model concentrates the search about 4–5× over random.** In 97% of the draws the top 8 have a smaller median miss than the cluster as a whole: median 5.3 vs 11.7 noise units.
- **Most picks still miss at ±3.** The median top-8 miss of 5.3 noise units means a typical pick is off by more than 3 noise units on at least one of the five axes. Five axes at once is strict.
- **It works for oxides and fails for halides, and that is mostly an artefact of how the clusters fall.** The chemical clusters split largely by anion: 0, 3, 4 and 8 are oxides, 7 chalcogenides, 2 and 6 fluorides, 1 chlorides (96% Cl), 5 bromides/iodides. The top-8 hit rate at ±3 is 0.27–0.38 for the oxide and chalcogenide clusters, 0.15 and 0.06 for the fluorides, and 0.02 for the chlorides and 0.00 for the bromides/iodides. Holding out an oxide cluster leaves three other oxide clusters in training. Holding out cluster 1 or 5 removes *every* chloride or *every* bromide/iodide, so it tests a held-out anion, which the scorecard already shows to be the hard case.
- **For the prospective set this is pessimistic for halides.** There, all MP clusters (halides included) are in training. The halide targets t1–t3 (Rb₂LiTlBr₆, Cs₂LiBiF₆, K₂NaTlCl₆) still have predicted median misses of 4.1–5.5, against 1.4–1.9 for the oxide targets. So expect fewer hits on them.
- **Limits.** Only 6 targets per cluster, so per-cluster rates are uncertain by about ±0.1–0.2. The retrospective trains on MP polymorphs only (no mixed halides). It scores against MLIP labels, so it says nothing about MLIP-vs-DFT fidelity.

**Prospective set** (`data/proposals/mp_pool_v1/`): 48 proposals (6 medoid targets × 8) and 17 random controls from 1,219 never-computed MP perovskites. Nothing is computed yet. It needs a no-D3 MLIP run (`MODEL=...+nod3`), and some cells are 152–156 atoms (Ca–Mn oxides).

## Caveats and next step

- **All data is MLIP.** The noise floor measures disagreement between models, not error against DFT. Only Cv and S have a DFPT comparison, and only on five cubic cells.
- **No dispersion correction.** Every dataset here was computed without DFT-D3. The pipeline now adds D3(BJ) by default (2026-11-18), with a `-D3BJ-<xc>` output label; `+nod3` turns it off. New runs aren't comparable with these unless D3 is switched off. D3 visibly softens tilt instabilities: with D3, cubic CsPbBr₃ is only −3.4 cm⁻¹ unstable, and its soft-mode step reaches Cm with no energy gain rather than a tilted phase.
- **68% of the MP polymorphs come from "theoretical" entries.**
- **Formula-grouped CV may still be optimistic.** Closely related double perovskites (single-site substitutions) can fall on both sides of a split. Grouping by composition cluster would be stricter.
- **Composition clusters** now cover every polymorph with an ionic picture. That makes cluster-grouped CV (above) possible, but it isn't implemented yet.
- **Mixed halides:** single SQS cells limit the width and MSD targets (Section 9). No model predicts the bowing for an unseen halide pair yet (Section 8).
- **Inverse design can target about 4–5 numbers, not a full spectrum.** Spectral shape beyond PC1 is only weakly predictable from these descriptors.

**Decisive next step: a prospective test.** Choose target values on the core axes. Have the model propose compositions, ideally from a held-out composition cluster. Then compute the top candidates with DFPT. If the predicted descriptors hold up, the framework is validated end to end. If they don't, the gap is MLIP fidelity, not the descriptor framework.
