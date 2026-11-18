# Descriptor selection for perovskite phonon inverse design

**Question.** Which phonon descriptors can an inverse-design model target, and which composition and structure descriptors predict them?

**Short answer.** The approach is viable, with a narrower scope than "design an arbitrary vDOS." Across 1,984 MLIP-relaxed, dynamically stable perovskite polymorphs:

- The phonon side has about **4–5 independent, learnable directions**: frequency scale, A-site rattling, B–X stretching-band separation, B-sublattice weight, and lattice instability.
- All five are well above MLIP noise and predictable within an anion family (within-family R² ≈ 0.8–0.95).
- The A-site and instability descriptors also transfer to an anion family the model never saw (R² 0.5–0.84).
- The absolute frequency scale does not transfer, but the anion is a discrete choice made by enumeration anyway.
- Finer spectral shape is only weakly predictable.
- Every result rests on MLIP data. A prospective DFPT check is the decisive next step (see [Caveats](#caveats-and-next-step)).

Reproduce with `descriptor_selection.ipynb` (about 4 min). Its inputs are built by `scripts/analysis/extract_design_descriptors.py`, and the descriptor definitions are in `src/analysis/design_descriptors.py`.

---

## 1. Data

| | |
|---|---|
| Structures | Materials Project perovskites that passed the corner-sharing-octahedra check (`src/data/perovskite_structure_check.py`) |
| Phonons | MACE-matpes-r2scan-omat-ft, finite displacements (phonopy), from `scripts/mlip/run_mlip_phonons.py` |
| Polymorph per formula | The lowest-energy dynamically stable state among *symmetric*, *rattled* and *soft-mode* relaxations (`select_stable_polymorphs.py`) |
| Kept | 1,984 polymorphs, 1,952 formulas: 1,289 oxides, 284 fluorides, 211 chlorides, 154 bromides/iodides, 46 other |
| Dropped | Formulas with no dynamically stable state, and 2 antiperovskites (K₃IO, Rb₃IO) with no A/B sublattice |

All descriptors come from the saved force constants (`phonopy_params.yaml`), so no new MLIP calls are needed. Those files total 224 GB on Sherlock scratch. They are read with `mlip_phonons.load_compact`, which keeps only the unit-cell force-constant rows and gives bit-identical frequencies to `phonopy.load` at 10–70× the speed.

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
| `shape_pc1` … `shape_pc4` | PCA scores of the scale-normalized vDOS quantile function Q(p)/ω̄ (Section 2a) | A compact, scale-free representation of the whole spectrum. Together PC1–4 explain 90% of the shape variance (69%, 11%, 7%, 4%). |
| `rel_gap` | Largest gap in the vDOS ÷ bandwidth | Phonon band gaps matter for phonon–phonon scattering channels. |

PC1 has a step at p = 0.8, which is the B–X stretching band: 3 of every 15 modes in ABX₃ are stretching, so it occupies the top 20%. PC1 therefore measures how far the stretching band separates from the rest, and it tracks `rel_gap` and `rel_width_X`.

### Lattice instability

| Target | Definition | Why it matters |
|---|---|---|
| `log_distortion_energy` | ln(1 + E_symmetric − E_polymorph), in meV/atom: how far the structure relaxed below its symmetric MP parent | The depth of the double well the polymorph sits in. A proxy for anharmonicity and for how far the structure is from a phase transition. |
| `soft_mode_depth` | −min(ω) of the symmetric parent, in cm⁻¹, with 0 if the parent is stable | How unstable the parent is: the strength of the soft mode the polymorph condensed. |

### 2a. Spectral distance (Wasserstein-2)

Two spectra are compared through their quantile functions Q(p), the frequency below which a fraction p of the modes lie. The Euclidean distance between quantile functions is the Wasserstein-2 distance (up to a constant factor). It treats a shifted band as moving by its frequency difference, which a binned-DOS distance does not.

Writing Q = ω̄ · Q̃ splits W2² into a pure scale term and a shape term. On random polymorph pairs, **64%** of W2² is pure frequency scale. That's why the shape is modeled separately through `shape_pc*`.

---

## 3. Metrics: inputs, in tiers by what they cost to compute

The tiers matter for inverse design. Anything learnable from **C** can be screened over any enumerated composition for free. Anything that needs **G** requires one relaxation per candidate.

| Tier | Needs | Descriptors |
|---|---|---|
| **M** | Formula | Site masses; reduced masses μ_BX, μ_AX; ln(m_A/m_X) |
| **C** | + site assignment | Shannon radii r_A, r_B, r_X; Goldschmidt tolerance factor and octahedral factor; Bartel τ; Pauling electronegativities; oxidation states; site heterogeneity (number of species, mass coefficient of variation, oxidation-state spread on A and B) |
| **C+G** | + one MLIP relaxation | BX₆ geometry: mean B–X length and its spread across octahedra, bond-length distortion, quadratic elongation, bond-angle variance, B off-centering. Tilt: mean and minimum B–X–B angle, tilt proxy (180° − ⟨θ_BXB⟩)/2. A site: ECoN, mean and minimum A–X distance, A–X spread. Realized tolerance factor t_geom = ⟨d_AX⟩/(√2⟨d_BX⟩). Bonding: bond-valence-sum deviation per sublattice, global instability index (GII). Volume per formula unit. |
| **C+G+I** | + phonons of the parent structure | Distortion energy and soft-mode frequency of the symmetric parent and, where one exists (6% of rows), the cubic Pm-3m parent |

Notes:

- **M is not a "trivial mass" baseline.** Under formula-grouped CV, the masses act as element identifiers. That's how **M** reaches R² ≈ 0.96 even on the mass-free `log_k_*`. Read **M** as "element identity, no chemistry descriptors."
- **Bond-valence descriptors are missing for 4% of rows**, for elements (e.g. Ac) with no Brese–O'Keeffe parameters. The gradient-boosted models handle these missing values natively.
- **The I tier is never used to predict the instability targets**, since they are built from the same calculations.

---

## 4. The tests

Each candidate target is run through the tests below. They answer different questions, and a target is only worth modeling if it passes all of them. Values are summarized in Section 5.

### Test 1: Noise floor (SNR)

**What.** SNR = σ_dataset / σ_models. The numerator is the target's spread across the 1,984 polymorphs. The denominator is its spread across eight MLIPs (MACE and ORB variants) on the same structure, taken as the median over structures.

**Why it matters.** A model trained on MLIP data can't resolve differences smaller than the MLIP's own uncertainty. If a target varies across the dataset by about as much as MLIPs disagree on a single structure, its "signal" is mostly model error, and a surrogate fitted to it would learn that error. SNR ≲ 3 is the rejection threshold.

**Two noise sets:**
- **Gauntlet** (primary): Pnma CsPbBr₃, with each model relaxing its own soft-mode structure. It's stable and tilted, and it includes each model's geometry error, so it represents the polymorphs actually being modeled.
- **Benchmark**: five cubic cells at the DFT geometry, all dynamically unstable. It's pessimistic for anything weighted toward the soft end, because there that end includes imaginary branches.

Where DFPT references exist, the MLIP-vs-DFPT RMS error on Cv and S is also reported: 0.14 and 0.19 of their dataset spread.

**Result.** Every target clears SNR ≥ 3 on the gauntlet. The weakest are `shape_pc3` (3.4) and `shape_pc2` (3.8). On the cubic benchmark, the soft-end descriptors drop below 3: `log_msd_A` 1.8, `low_enrich_A` 2.4. That reflects the unstable reference cells, not the stable polymorphs, but it's a reason to verify soft-end targets against DFPT on tilted cells.

### Test 2: Anion share (η²)

**What.** η² = 1 − SS_within / SS_total, the fraction of a target's variance explained by anion family alone (and separately by the composition clusters from `cluster_compositions.py`).

**Why it matters.** A target that's mostly "which anion" has only a handful of attainable values. A model will learn the anion and look accurate while saying nothing about design within a family. η² also determines which form of R² (Test 4) to trust.

**Result.** The absolute targets are **80–90% anion**: ω_log, ω_max, Cv, S, all `log_k_*` and all `log_msd_*`. The scale-free targets are mostly *not* anion: `rel_centroid_A` 0.23, `rel_width_A` 0.13, `shape_pc1` 0.19, `log_distortion_energy` 0.15, `soft_mode_depth` 0.01. That's the design-relevant variation.

### Test 3: Redundancy (Spearman ρ, participation ratio)

**What.**
- The Spearman rank correlation between every pair of targets, shown as a hierarchically clustered map.
- The participation ratio PR = (Σλ)² / Σλ² of the target correlation matrix's eigenvalues, an estimate of how many independent directions the targets span.

**Why it matters.** Targets in a block with |ρ| > 0.85 measure the same physics. Modeling all of them adds no design freedom and inflates any score summed over targets. The participation ratio sets an upper bound on how many phonon properties can be designed independently. Spearman rather than Pearson is used because the relationships are monotonic but not linear.

**Result.** PR = **4.1 effective dimensions out of 26 targets**. The clustered map shows the blocks:
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
- **Within a family**, most targets are learnable: R²w(C+G) ≥ 0.8 for 15 of 26. Composition alone gets most of the way there, and geometry adds ≤ 0.10, except for `log_distortion_energy` (+0.17), `soft_mode_depth` (+0.18) and `rel_width_B` (+0.13). Those are the ones that depend on the realized polymorph.
- **Across anions**, only the A-site and instability descriptors transfer:

  | Target | Held-out-anion R² |
  |---|---|
  | `rel_width_A` | 0.84 |
  | `log_distortion_energy` | 0.84 |
  | `rel_centroid_A` | 0.63 |
  | `log_msd_X` | 0.59 |
  | `rel_width_X` | 0.53 |
  | `log_k_A` | 0.51 |
  | `shape_pc1` | 0.48 |

  The absolute frequency targets fail outright (R² −1 to −6). The model can't predict the scale of an unseen anion family.
- **Weak:** `shape_pc2`–`pc4` and `rel_gap` (R²w 0.55–0.69). Spectral detail beyond PC1 isn't well captured by these descriptors.

### Test 5: Invertibility (ρ_inv)

**What.** For each target, take each polymorph's 10 nearest neighbors in that target's value. Then compute ρ_inv as the median distance between those neighbors in the standardized **C**-descriptor space, divided by the median distance between random pairs.

**Why it matters.** Inverse design maps a target value back to chemistry, so it matters whether that map is narrow or broad.
- **ρ_inv ≪ 1:** a target value pins down the chemistry, giving a near-unique answer.
- **ρ_inv ≈ 1:** many distinct chemistries reach the same value. That's the "multiple synthetic routes" property this project looks for, but it means the inverse problem returns a broad family rather than a single answer.

Read it together with learnability. Learnable *and* ρ_inv near 1 means the forward model has found structure-property rules that cut across composition-space distances.

**Result.**
- No target localizes the chemistry strongly (ρ_inv 0.67–0.95).
- The anion-dominated absolute targets localize most: `log_k_A` 0.67, ω_log 0.72.
- The scale-free targets are broad: A-site descriptors about 0.89, `shape_pc*` 0.88–0.95. A given spectral shape is reachable from many chemistries.

### Test 6: Feature attribution (permutation importance)

**What.** For targets that pass Tests 1 and 4 (SNR ≥ 3, R²w ≥ 0.5), compute the permutation importance of each C+G descriptor on a held-out 20% split grouped by formula.

**Why it matters.** This identifies the structural side of the eventual model: which descriptors carry the phonon information, and so which ones a generative or search step must control. It's also a physics sanity check. Importances that don't make physical sense would indicate leakage or an artifact.

**Result.** The attributions are physically sensible:

| Target | Main descriptors | Interpretation |
|---|---|---|
| `rel_centroid_A`, `low_enrich_A` | ln(m_A/m_X) | The A/X mass ratio sets where the A band sits |
| `rel_width_A` | A-site ECoN | Coordination sets how localized the A mode is |
| ω_log, S | ⟨d_AX⟩ and μ_BX | Bond length and reduced mass set the scale |
| `log_k_A` | X oxidation state | Anion charge sets the A-site restoring force |
| `shape_pc1`, `rel_width_X` | A electronegativity, octahedral factor, A-site bond-valence deviation | These set how far the stretching band separates |

### Shortlist

The notebook builds a shortlist greedily by R²w(C+G). It skips any target with SNR < 3 or R²w < 0.5, and any target with |ρ| > 0.85 to one already picked. At that threshold it keeps 18 targets, which is too many given PR ≈ 4. The recommended core set (Section 6) takes one target per correlation block instead.

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
| `log_omega_log` | 11.9 / 7.7 | 0.90 | 0.91 | 0.93 | −1.37 | 0.72 |
| `s_300K` | 11.1 / 8.6 | 0.90 | 0.91 | 0.94 | −1.45 | 0.72 |
| `log_k_A` | 9.2 / 5.1 | 0.86 | 0.85 | 0.93 | 0.51 | 0.67 |
| `rel_centroid_A` | 8.7 / 3.1 | 0.23 | 0.92 | 0.95 | 0.63 | 0.89 |
| `rel_width_A` | 5.8 / 3.1 | 0.13 | 0.86 | 0.93 | 0.84 | 0.90 |
| `low_enrich_A` | 15.8 / 2.4 | 0.32 | 0.85 | 0.87 | 0.33 | 0.88 |
| `shape_pc1` | 6.1 / 3.6 | 0.19 | 0.85 | 0.90 | 0.48 | 0.88 |
| `rel_width_X` | 4.5 / 5.0 | 0.19 | 0.84 | 0.89 | 0.53 | 0.89 |
| `rel_centroid_B` | 12.3 / 5.8 | 0.60 | 0.91 | 0.93 | 0.25 | 0.82 |
| `high_enrich_B` | 10.7 / 9.9 | 0.67 | 0.81 | 0.82 | −0.51 | 0.83 |
| `log_omega_max` | 7.7 / 13.5 | 0.88 | 0.79 | 0.82 | −3.82 | 0.74 |
| `log_k_B` | 5.7 / 5.9 | 0.86 | 0.73 | 0.79 | −1.25 | 0.77 |
| `log_distortion_energy` | — | 0.15 | 0.61 | 0.78 | 0.84 | 0.88 |
| `soft_mode_depth` | — | 0.01 | 0.45 | 0.63 | 0.23 | 0.90 |
| `rel_gap` | — | 0.15 | 0.54 | 0.55 | 0.20 | 0.90 |
| `shape_pc2` | 3.8 / 4.4 | 0.03 | 0.51 | 0.57 | 0.06 | 0.95 |

A dash means no SNR: the instability and gap targets can't be computed for the noise-set structures, so they pass Test 1 by default.

---

## 6. Recommended core targets

One target per independent direction, matching the participation ratio:

| Direction | Target | Role in design |
|---|---|---|
| Frequency scale | `log_omega_log` | Set mostly by anion choice. Enumerate the anion, then the model works within that family. |
| A-site rattling | `rel_centroid_A` and/or `rel_width_A` | The most learnable and best-transferring axis, controlled by the A/X mass ratio and A-site coordination. |
| B–X stretching separation | `shape_pc1` | The dominant spectral-shape mode, related to phonon gaps. |
| B-sublattice weight | `rel_centroid_B` | Where B sits in the spectrum. |
| Lattice instability | `log_distortion_energy` | Needs a relaxation per candidate, but transfers across anions. |

This implies a **two-stage forward model**: composition → relaxed geometry (tilt, ECoN, bond-valence deviation) → phonon targets. Composition alone suffices for most targets. The geometry stage pays off for the instability targets and the B-site band width.

---

## Caveats and next step

- **All data is MLIP.** The noise floor measures disagreement between models, not error against DFT. Only Cv and S have a DFPT comparison, and only on five cubic cells.
- **70% of the polymorphs come from MP "theoretical" entries.**
- **Formula-grouped CV may still be optimistic.** Closely related double perovskites (single-site substitutions) can fall on both sides of a split. Grouping by composition cluster would be stricter.
- **Inverse design can target about 4–5 numbers, not a full spectrum.** Spectral shape beyond PC1 is only weakly predictable from these descriptors.

**Decisive next step: a prospective test.** Choose target values on the core axes. Have the model propose compositions, ideally from a held-out composition cluster. Then compute the top candidates with DFPT. If the predicted descriptors hold up, the framework is validated end to end. If they don't, the gap is MLIP fidelity, not the descriptor framework.
