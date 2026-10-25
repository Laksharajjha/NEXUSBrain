# NEXUS-0 Project Status

**Last Updated**: 2026-10-25
**Current Stage**: PHASE 1.6 IN PROGRESS (Make the Mechanisms Earn Their Place — `EXPLORATORY`)
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
  - [x] **Part 1 (Leakage & Protocol Audit)**: `SpyLearnerWrapper`, static AST field-access audit, shuffled training order, and near-miss distributional reliance diagnosis.
  - [x] **Part 2 (Negative Controls — 20 seeds)**: `2b` (`0.499`), `2c` (`0.500`), `2a` (`0.657 ± 0.109` on `test_novel_combination`, `0.771 ± 0.081` on `dev`).
  - [x] **Part 3 (Difficulty Ladder — 20 seeds, no tuning)**: Full `< 0.90` and `< 0.70` breaking points documented.
  - [x] **Part 4 (Stronger Baselines — 20 seeds)**: Baselines C, D, E, F evaluated.
  - [x] **Part 5 (MDL Bit-Cost Prior & Belief Revision Audit)**: Compared $\alpha = 0.05$ vs. $\log_2$ bit-cost prior and episodic replay vs. incremental revision.

---

## Phase 1.6 Active Checklist (`EXPLORATORY`)
- [ ] **Task 0 (Fresh Evaluation Data)**: Seed range `1000..1049` (50 seeds). Old test splits (`0..19`) burned for development; develop/tune on `dev` only; evaluate fresh test splits (`1000..1049`) once at the end and log in `RERUNS.md`.
- [ ] **Task 1 (Separate the Evidence Channels)**: Implement `NEXUS-Truth`, `NEXUS-Cooc`, and `NEXUS-Both` with identical hypothesis space and beam; run Negative Controls `2a/2b/2c`; verify `NEXUS-Truth` goes to chance (`~0.50`) under shuffled labels; report accuracy and examples-to-criterion.
- [ ] **Task 2 (Matched-$N$ Comparison)**: Compare `NEXUS-Both`, Baselines D and F, and Baseline C at $N = 20, 50, 100, 200, 400, 1600$ over 50 seeds (`1000..1049`) with paired bootstrap 95% CIs; break down NEXUS results by NP order, clause order, and particle position.
- [ ] **Task 3 (Does the Beam Matter?)**: Design early-ambiguity / garden-path stimulus order; compare $K = 1, 5, 50, 200$ over 50 seeds with paired CIs; document and test `commit_after_n` (`4, 8, 12, 24, 48`) for Baseline C on `dev`.
- [ ] **Task 4 (Does MDL Matter?)**: Build worlds where competing hypotheses differ in description length (synonyms, homonyms/polysemy, variable-size lexicons with null-meaning filler words); compare $\log_2$ bit-cost prior, flat prior ($\alpha = 0$), and $\alpha = 0.05$ on `dev`.
- [ ] **Task 5 (Report)**: Deliver `PHASE1_6_REPORT.md`, update `RERUNS.md` and `STATUS.md`, run full pytest suite, and stop.

---

## Exact Commands to Verify or Resume
```bash
# Run full test suite
.venv/bin/pytest -v

# Run Phase 1.6 mechanisms evaluation suite (50 seeds: 1000..1049)
.venv/bin/python -m nexus.experiments.phase1_6_mechanisms
```
