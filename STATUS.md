# NEXUS-0 Project Status

**Last Updated**: 2026-10-24
**Current Stage**: ALL STAGES COMPLETE (Stages A, B, C, D finished and verified)
**Current Branch**: `feature/phase-0-toy-world-eval`
**PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`

---

## Completed Stages
- **Phase 0**: Toy world, logic, splits, evaluation runner, Baseline A (Memorizer).
- **Phase 0.5**: Initial Oracle ceiling, SentenceOnly & SceneOnly baselines, per-seed randomization, frozen `PREREGISTRATION.md` (`prereg-v1`).
- **STAGE A (Phase 0.6 — Audit Patch, `gate-A`)**:
  - [x] **A1–A8**: Completed 50-seed pooled shortcut audit (`0.500 ± 0.000`), 63-feature permutation balance audit (`p_holm = 1.000`), novel-word candidate enumeration (`candidate_count == 1`), `StubbornOracle` (`0.000`) vs `CorrectedOracle` (`1.000`), independent second model checker (`naive_evaluate`), construction/order identifiability report, locked `PREREGISTRATION.md` SHA-256 in CI, and `ExhaustiveIdealObserver` on reduced world (`1.000`). Tagged `gate-A`.
- **STAGE B (Phase 1 Learner — `gate-B`)**:
  - [x] **B1–B5**: Implemented categorial grammar (`nexus/learner/categorial.py`), MDL beam search learner (`nexus/learner/nexus_learner.py`), explainability & faithfulness interventions (`nexus/learner/explain.py`), Baselines B (`CrossSituationalLearner`) & C (`NoRevisionLearner`), and 20-seed lexicon-permutation invariance + source scan tests (`nexus/tests/test_stage_b_learner.py`). Tagged `gate-B`.
- **STAGE C (Pre-Registered Evaluation — `gate-C`)**:
  - [x] **C1–C5**: Tagged `pre-run`, ran one-shot 20-seed evaluation (`0` reruns) across Baseline A, Baseline B, Baseline C, `NEXUSLearner`, `SentenceOnly`, and `SceneOnly`. Computed 10,000-resample paired bootstrap 95% CIs (`+0.500, CI [+0.500, +0.500]`), compared 3-seed induced lexicons against ground truth (`15/15` exact matches), categorized worst held-out items via `explain()`, wrote `PHASE1_REPORT.md` and `RERUNS.md`, and tagged `gate-C`.
- **STAGE D (Exploratory Analyses & SQLite Persistence — `LIMITATIONS.md`)**:
  - [x] **D1**: Beam width ablation $K \in \{1, 5, 25, 100, 200, 500\}$.
  - [x] **D2**: Noise parameter ablation $\varepsilon \in \{0.001, 0.01, 0.02, 0.05, 0.1\}$.
  - [x] **D3**: Learning curves at $N \in \{50, 100, 200, 400, 800, 1600\}$ vs baselines + gap to A8 `ExhaustiveIdealObserver` on reduced world.
  - [x] **D4**: Hypothesis-space ablations (`given_lexicon`, `given_syntax`, `max_slash_depth=1`).
  - [x] **D5**: Label noise $\{0\%, 2\%, 5\%, 10\%, 20\%\}$ before/after corrections.
  - [x] **D6**: New held-out probes (role reversal on symmetric vs asymmetric relations, stacked 2-adj and 4-adj clauses, 3-adj same-family conflict boundary, unseen `adj + novel_noun`).
  - [x] **D7**: Calibration reliability diagrams + Murphy Brier score decomposition ($\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$).
  - [x] **D8**: Failure taxonomy compiled in `LIMITATIONS.md` (labeled `EXPLORATORY`).
  - [x] **D9**: SQLite persistence round-trip (`nexus/storage/repository.py` + `nexus/storage/schema.sql`) verified in a fresh subprocess (`nexus/tests/test_stage_d_persistence_and_ablations.py`).

---

## Open Issues
- None. All stages (A -> B -> C -> D) and all tests pass.

---

## Commands to Resume / Verify
```bash
# Run full test suite (59 tests)
.venv/bin/pytest -v

# Re-run Stage A, Stage C, or Stage D experiments
.venv/bin/python -m nexus.experiments.stage_a_audit
.venv/bin/python -m nexus.experiments.phase1_eval
.venv/bin/python -m nexus.experiments.stage_d_exploratory
```
