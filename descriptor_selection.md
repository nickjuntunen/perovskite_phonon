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

The composition-cluster η² is computed on a stale clustering: `data/mp_composition_clusters.csv` (2026-09-17) has no label for 788 of the 2,762 polymorphs. See Caveats.

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

**Running (2026-11-19):** `gp_vs_tree/gp_disorder.py` tests whether the X-site disorder features and a bowing-only target fix that. It covers 9 targets and three variants (base C+G; + disorder features; + disorder features with the target taken relative to the linear baseline). Each variant is fitted with GP and trees and scored on both a held-out pair and a held-out series (GroupKFold(5)). Output: `gp_vs_tree/disorder/gp_disorder.csv`. A quick local check reproduces the notebook's linear-baseline R²(series) for rel_centroid_A (0.19).

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

- Scoring against the ensemble mean barely changes any model's R². Where models fall short, the x-dependence is unlearned, not drowned in noise. Most targets have models well below their ceiling.
- The GP beats trees and the linear baseline on almost every target here too.
- For the disorder-limited targets, production would need ensemble averages over several SQS cells before a model could be scored meaningfully along x.
- **Seed noise vs MLIP noise** on the core targets: ω_log 0.29×, rel_centroid_A 1.02×, rel_width_A 0.78×, shape_pc1 0.59×, rel_centroid_B 1.47×. Disorder noise is comparable to the MLIP noise, and larger for rel_centroid_B.

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

## Caveats and next step

- **All data is MLIP.** The noise floor measures disagreement between models, not error against DFT. Only Cv and S have a DFPT comparison, and only on five cubic cells.
- **No dispersion correction.** Every dataset here was computed without DFT-D3. The pipeline now adds D3(BJ) by default (2026-11-18), with a `-D3BJ-<xc>` output label; `+nod3` turns it off. New runs aren't comparable with these unless D3 is switched off. D3 visibly softens tilt instabilities: with D3, cubic CsPbBr₃ is only −3.4 cm⁻¹ unstable, and its soft-mode step reaches Cm with no energy gain rather than a tilted phase.
- **68% of the MP polymorphs come from "theoretical" entries.**
- **Formula-grouped CV may still be optimistic.** Closely related double perovskites (single-site substitutions) can fall on both sides of a split. Grouping by composition cluster would be stricter.
- **The composition clusters are stale.** `cluster_compositions.py --mp-metadata` still builds its own per-site descriptors, and the 2026-09-17 run left 788 polymorphs unlabelled. Regenerating them from `composition_features.csv`, which has a row for every polymorph, would fix the chem-cluster η² and enable cluster-grouped CV.
- **Mixed halides:** single SQS cells limit the width and MSD targets (Section 9). No model predicts the bowing for an unseen halide pair yet (Section 8).
- **Inverse design can target about 4–5 numbers, not a full spectrum.** Spectral shape beyond PC1 is only weakly predictable from these descriptors.

**Decisive next step: a prospective test.** Choose target values on the core axes. Have the model propose compositions, ideally from a held-out composition cluster. Then compute the top candidates with DFPT. If the predicted descriptors hold up, the framework is validated end to end. If they don't, the gap is MLIP fidelity, not the descriptor framework.
