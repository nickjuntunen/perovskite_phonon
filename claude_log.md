# Claude session log

Progress record for Claude Code sessions on this repo (CLAUDE.md points new sessions here).

**Conventions**
- One entry per session, newest first. Each entry has: date range, what was done, where the outputs are, current state, open threads.
- Write facts verified in the session; mark anything not checked as such.
- Keep standing constraints (Sherlock policy, workflow rules) in the section below, not in entries.
- At the end of a session, add an entry or update the current one before stopping.
- The last section is always "## 2026-11-20: job check — SQS ensemble, retrospective, GP disorder, zwitterion D3

- **SQS ensemble (`mlip-mixed-ens`, array 52996336, 50 tasks, no D3).** Submitted by the tmux pipeline 2026-11-19 14:18.
  - Generation: 4,143 cells for the 179 Cl/Br + Br/I series. 1,200 reuse a finished run (878 production seed-0, 322 noise-check), listed in `data/mixed_halides_ensemble/reuse.csv`; 2,943 computed.
  - As of 2026-11-20 10:56: 33 tasks COMPLETED, 7 FAILED, 10 running. Output: `$SCRATCH/perovskite_phonon/mlip_phonons_mixed_ensemble`.
  - Each FAILED task lost exactly one structure; its other runs finished. The 7 lost: 4 × "Niggli reduction failed" (K2Ca2IBr5 sqs0, Rb2Pb2Br5Cl sqs0/sqs4, RbPbBrCl2 sqs2), 1 × "Failed to recognize lattice" (Cs2NaIn(BrCl)3 sqs4), 2 × relaxation not reaching fmax 1e-3 in 1000 steps (Cs2YCu(BrCl)3 sqs2, K2Ag2Br5Cl sqs0). Not resubmitted: these are per-structure failures, dropped like production's.
- **Retrospective check (`propose_candidates.py --retrospective`), done.** Each chemical cluster held out in turn, top 8 per target, Chebyshev miss in MLIP-noise units, n = 60 target draws:

  | tol | base rate | hit rate top 8 | enrichment |
  |---|---|---|---|
  | 2 | 0.013 | 0.062 | 4.7× |
  | 3 | 0.036 | 0.183 | 5.1× |
  | 5 | 0.120 | 0.465 | 3.9× |

  Median miss is 5.3 for the top picks vs 11.7 for the pool. Per cluster (tol 3) the hit rate ranges from 0 (cluster 5) and ~0.02–0.06 (clusters 1, 6, 9) to 0.27–0.38 (clusters 0, 3, 4, 7, 8). Output: `data/proposals/retrospective/`. `propose_candidates.py` committed.
- **GP disorder (52985751 → summary 52985752), done.** `gp_vs_tree/disorder/gp_disorder.csv` (Sherlock; copy in the scratchpad). R², 672 compositions scored:
  - The X-site disorder features don't help. Held-out pair: rel_width_X base GP −3.19 → with disorder features −13.87; rel_centroid_B 0.71 → 0.30; ω_max −2.61 → −5.66; the rest change by ≤ 0.1. Within series: unchanged (±0.02).
  - Predicting the bowing only (`dis_delta`) is far worse: within series, R² from −1.8 to −18 for all but ω_log (0.59).
  - So the unseen-pair width/shape gap stays open, and these features don't close it.
- **Zwitterion D3 (52858514), done.** 10/15 COMPLETED. The 5 FAILED tasks (0, 6, 7, 10, 13) each name one structure whose relaxation missed fmax 1e-3 in 1000 steps (zw-29580-PbCl, zw-88879801-PbCl, zw-12634510-PbBr, zw-155140147-PbBr, zw-54315514-PbBr).

## Goals and key evidence (keep last; rewrite on every update)

*Rewritten 2026-11-20 11:05.*

**Goals.**
- **Inverse design of perovskite phonons:** choose a design target on the phonon side and determine which compositions yield it.
  - The targets are about five learnable descriptors (`descriptor_selection.md` §6): frequency scale, A-site rattling, B–X stretching separation, B-sublattice weight and lattice instability.
  - The candidates are ABX₃ and double perovskites, including mixed halides and the widened chemistry (zwitterions).
- **Redundancy:** find the distinct chemistries that reach the same target (`cluster_targets.py`).
- **Feasibility:** the proposed compositions must actually form. That means dynamically stable and, for mixtures, not phase-separating (`mixing_energies.py`).
- Later: electron–phonon coupling.

**The two pieces of information most likely to settle it:**

1. **Whether the forward model is accurate where the inverse search will query it**, i.e. for compositions outside the training set.
   - Within anion families this is already shown (R²w 0.75–0.96).
   - It is not shown along mixed-halide series for an unseen halide pair, where the width and shape targets bow. The X-site disorder features did not close that gap (GP disorder, 2026-11-20 entry); the bowing-only target is worse.
   - The remaining lever is the noise floor: single-SQS disorder caps how well any model can do along a series, and `sqs_convergence.py` puts the fix at about 5–9 cells per composition for the core targets.
   - **Running:** SQS ensemble `mlip-mixed-ens` array **52996336** (5 seeds, Cl/Br + Br/I; 33/50 tasks done, 10 running as of 2026-11-20 10:56). Post-processing (ensemble averaging with `reuse.csv`, then re-scoring the series models against the averaged targets) is not yet written.
2. **Prospective (inverse) validation:** pick target values, let the model propose unseen compositions, compute them, and compare. The first step is MLIP phonons on proposals from a held-out composition cluster; the second is DFPT on the best few, checking MLIP fidelity on tilted cells. This is the only end-to-end evidence; everything so far is cross-validation on MLIP data.
   - The retrospective check gives about 5× enrichment at tol 3, but it is near zero for clusters 1, 5, 6 and 9.
   - The proposal set is ready: 48 proposals plus 17 controls in `data/proposals/mp_pool_v1/`.
   - **None running.** The proposals need a no-D3 (`+nod3`) MLIP array; it has not been submitted. Some cells are 152–156 atoms (Ca–Mn oxides).
