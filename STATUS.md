# NEXUS-0 Project Status

**Last Updated**: 2026-10-24
**Current Stage**: PHASE 1.5 (Adversarial Validation & Failure Point Discovery — EXPLORATORY)
**Current Branch**: `feature/phase-0-toy-world-eval`
**PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`

---

## Completed Stages
- **Phase 0 & 0.5**: Toy world, splits, metrics, Memorizer, Oracle ceiling, SentenceOnly, SceneOnly, frozen `PREREGISTRATION.md` (`prereg-v1`).
- **STAGE A (`gate-A`)**: 50-seed pooled shortcut audit, 63-feature permutation balance audit, novel-word candidate enumeration, `StubbornOracle` vs `CorrectedOracle`, second model checker `naive_evaluate`, identifiability report, and `ExhaustiveIdealObserver` (`AUDIT_0_6.md`).
- **STAGE B (`gate-B`)**: Categorial grammar (`nexus/learner/categorial.py`), MDL beam learner (`nexus/learner/nexus_learner.py`), explainability & faithfulness (`nexus/learner/explain.py`), Baselines B & C, 20-seed permutation invariance (`nexus/tests/test_stage_b_learner.py`).
- **STAGE C (`gate-C`)**: One-shot 20-seed pre-registered evaluation (`PHASE1_REPORT.md`, `RERUNS.md`).
- **STAGE D (`LIMITATIONS.md`)**: Exploratory ablations D1–D8 and SQLite persistence D9 (`nexus/storage/`).

---

## Active Stage: PHASE 1.5 (Adversarial Validation — `EXPLORATORY`)
- [ ] **Part 1 (Leakage & Protocol Audit)**:
  - Spy/wrapper test asserting zero held-out items appear in `observe()` or experience log before scoring, and `predict()` never mutates state.
  - Exhaustive AST + runtime list of every field read from `Observation`, `Correction`, `Scene`, `SceneObject`, `SceneRelation`.
  - Plain statement of how contrastive negatives in training interact with `_propose_word_candidates`.
- [ ] **Part 2 (Negative Controls — 20 seeds each)**:
  - 2a: Shuffled training truth labels ($\approx 0.50$).
  - 2b: Shuffled training scenes ($\approx 0.50$).
  - 2c: Cross-seed lexicon mismatch (train on seed $s$, test on seed $(s+1)\bmod 20$, $\approx 0.50$).
- [ ] **Part 3 (Difficulty Ladder — 20 seeds each, no tuning)**:
  - 3.1: Training size $N \in \{10, 20, 50, 100, 200, 400\}$ (both curriculum-prefix and stratified).
  - 3.2: Remove paired-contrast examples from training (random negatives only).
  - 3.3: Positive-only training (no false examples).
  - 3.4: Uncorrected label noise $\eta \in \{0\%, 2\%, 10\%, 20\%, 30\%\}$.
  - 3.5: Lexicon scaling: $6, 12, 24, 48$ kinds/words with proportionally scaled attributes.
  - 3.6: Ambiguous words (homonymy/polysemy: 1 form, 2 meanings) and synonyms (2 forms, 1 meaning).
  - 3.7: Longer sentences and nested modification beyond training depth (2-adj, 3-adj same-family, 2-clause chains).
  - 3.8: Beam width $K \in \{1, 2, 5, 10, 50, 200\}$ vs. Baseline C.
- [ ] **Part 4 (Stronger Baselines — 20 seeds each)**:
  - Diagnose and fix Baseline C (`NoRevisionLearner`) `0.505` train accuracy bug (committing to `"N"` fallback on `False` observations).
  - Baseline D: Position-aware cross-situational learner (`nexus/baselines/position_aware_cs.py`).
  - Baseline E: From-scratch NumPy neural model (`nexus/baselines/neural_ref.py`).
  - Baseline F: `NEXUS` with `given_syntax` vs. `given_lexicon`.
- [ ] **Part 5 (Documentation & `PHASE1_5_REPORT.md`)**:
  - Document $\alpha = 0.05$ vs. exact $\log_2$ bit-cost MDL prior (run and report both).
  - Explain `correct()` replay vs. true belief revision.
  - Deliver `PHASE1_5_REPORT.md` and update `RERUNS.md`.
