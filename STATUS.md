# NEXUS-0 Project Status

**Last Updated**: 2026-10-25
**Current Stage**: PHASE 1.5 COMPLETE (Adversarial Validation & Failure-Point Audit — `EXPLORATORY`, Stopped for Review)
**Current Branch**: `feature/phase-0-toy-world-eval`
**PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`

---

## Completed Stages
- **Phase 0 & 0.5**: Toy world, splits, metrics, Memorizer, Oracle ceiling, SentenceOnly, SceneOnly, frozen `PREREGISTRATION.md` (`prereg-v1`).
- **STAGE A (`gate-A`)**: 50-seed pooled shortcut audit, 63-feature permutation balance audit, novel-word candidate enumeration, `StubbornOracle` vs `CorrectedOracle`, second model checker `naive_evaluate`, identifiability report, and `ExhaustiveIdealObserver` (`AUDIT_0_6.md`).
- **STAGE B (`gate-B`)**: Categorial grammar (`nexus/learner/categorial.py`), MDL beam learner (`nexus/learner/nexus_learner.py`), explainability & faithfulness (`nexus/learner/explain.py`), Baselines B & C, 20-seed permutation invariance (`nexus/tests/test_stage_b_learner.py`).
- **STAGE C (`gate-C`)**: One-shot 20-seed pre-registered evaluation (`PHASE1_REPORT.md`, `RERUNS.md`).
- **STAGE D (`LIMITATIONS.md`)**: Exploratory ablations D1–D8 and SQLite persistence D9 (`nexus/storage/`).
- **PHASE 1.5 (`PHASE1_5_REPORT.md` — `EXPLORATORY`)**:
  - [x] **Part 1 (Leakage & Protocol Audit)**: `SpyLearnerWrapper` (`0` violations across `32,680` `observe()` and `30,000` `predict()` calls), static AST field-access audit (`0` generator metadata hits), shuffled training order (`1.000 ± 0.000`), and plain statement of near-miss distributional reliance.
  - [x] **Part 2 (Negative Controls — 20 seeds)**:
    - `2b` (Shuffled Scenes): `0.499 ± 0.012` on `test_novel_combination` (**PASS**).
    - `2c` (Cross-Seed Lexicon): `0.500 ± 0.004` on `test_novel_combination` (**PASS**).
    - `2a` (Shuffled Truth Labels): **`0.657 ± 0.109` on `test_novel_combination`, `0.771 ± 0.081` on `dev`** (**LEAK DETECTED & DIAGNOSED**: near-miss negative scenes preserve unlabeled `(scene, tokens)` co-occurrence, and `_propose_word_candidates` applies asymmetric positive-only weighting).
  - [x] **Part 3 (Difficulty Ladder — 20 seeds, no tuning)**: Full `< 0.90` and `< 0.70` breaking points documented across sample size $N$, random negatives, positive-only training, label noise $\eta$, lexicon scaling (`6..48` kinds), homonymy/polysemy, synonymy, beyond-depth sentences, and beam width $K$.
  - [x] **Part 4 (Stronger Baselines — 20 seeds)**: Diagnosed & fixed Baseline C (`NoRevisionLearner`, `0.505 -> 0.808` train), implemented Baseline D (`PositionAwareCrossSituationalLearner`, `0.918` on `test_novel_combination`), Baseline E (`FromScratchNeuralBaseline`, `0.485` on `test_novel_combination`), and Baseline F (`given_syntax` vs `given_lexicon`).
  - [x] **Part 5 (MDL Bit-Cost Prior & Belief Revision Audit)**: Compared $\alpha = 0.05$ vs. parameter-free $\log_2$ bit-cost prior (`compute_bit_cost_prior`), documented why `correct()` is full episodic replay rather than incremental belief revision, and updated `RERUNS.md` (`6,380` Phase 1.5 test-split evaluations).

---

## Exact Commands to Verify or Resume
```bash
# Run full test suite (65 tests)
.venv/bin/pytest -v

# Re-run Phase 1.5 adversarial validation suite (20 seeds)
.venv/bin/python -m nexus.experiments.phase1_5_adversarial
```
