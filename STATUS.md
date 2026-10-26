# NEXUS-0 Project Status

**Last Updated**: 2026-10-27  
**Current Stage**: PHASE 3 COMPLETED (Restructuring Evaluation & Exploratory Analyses Complete)  
**Current Branch**: `feature/phase-0-toy-world-eval`  
**Phase 1 PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b` (Frozen & Intact)  
**Phase 3 PREREGISTRATION_P3.md SHA-256**: `bd08fe66a8160483e741e45928aea22ee0f6699d655da9fca43bd98b33b9140d` (Frozen & Hashed)  

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
- **PHASE 1.6 (`PHASE1_6_REPORT.md` — `EXPLORATORY`)**:
  - [x] **Task 0 (Fresh Evaluation Data)**: Seed range `1000..1049` ($N=50$ fresh seeds). Test-split evaluations tracked in `RERUNS.md`.
  - [x] **Task 1 (Separate Evidence Channels)**: Implemented `NEXUS-Truth`, `NEXUS-Cooc`, and `NEXUS-Both`. Verified `NEXUS-Truth` drops to exactly chance ($0.497 \pm 0.021$) under shuffled labels (Control 2a), confirming zero label leak. Showed `NEXUS-Cooc` achieves $0.966$ without labels but leaves an unbridgeable $\approx 5.6\%$ gap on novel combinations.
  - [x] **Task 2 (Matched-$N$ Comparison)**: Evaluated $N \in \{20, 50, 100, 200, 400, 1600\}$ across 50 seeds with paired bootstrap CIs. Showed Baseline D outperforms NEXUS at $N \le 400$ by bypassing syntax, but hits a ceiling ($0.774$ on longer sentences, $0.853$ on contradictions). At $N=1600$, `NEXUS-Both` reaches $1.000$ across all splits ($+0.226$ over Baseline D on longer sentences).
  - [x] **Task 3 (Does the Beam Matter?)**: Tested standard vs. garden-path stimulus order ($K \in \{1, 5, 50, 200\}$). Showed beam width is inert under i.i.d. random order, but earns its place decisively under garden paths ($K \ge 50$ achieves $0.592$ vs. $0.501$ for greedy $K=1$, $+0.0915$ $[+0.0753, +0.1080]$, $p < 10^{-15}$). Evaluated Baseline C `commit_after_n` sensitivity.
  - [x] **Task 4 (Does MDL Matter?)**: Tested variable-length worlds (filler/null words, homonyms/polysemy, synonyms) comparing `log2_bits` vs. `linear` vs. `flat` ($\alpha=0$). Proved `log2_bits` earns its place under filler words ($+0.0497$ $[+0.0325, +0.0674]$ dev gain with smaller model) and polysemy (compressing from $17.20$ to $12.96$ entries).
  - [x] **Task 5 (Report & Documentation)**: Delivered `PHASE1_6_REPORT.md` with comprehensive tables, bootstrap CIs, mechanism verdicts, and architectural recommendations for Phase 3. Updated `RERUNS.md` and test suite (71 passing tests).
- **PHASE 3 (Restructuring: Split / Merge / Exception — Pre-registered)**:
  - [x] **Stage P3-A (`gate-P3-A`)**: Created `nexus/world/restructure.py` with worlds W1–W5. Audited across 50 fresh seeds (`2000..2049`): Oracle = 100%, SentenceOnly and SceneOnly at chance (0.500) on all test splits, 0 naive_evaluate mismatches, identifiability $\ge 5$, split sample sizes reported, and exhaustive ideal observer bounds established for W1 ($P(\text{split}) = 0.989$) and W2 ($P(\text{exception}) = 0.9997$). Delivered `AUDIT_P3.md`.
  - [x] **Stage P3-B (`gate-P3-B`)**: Authored `PREREGISTRATION_P3.md` (SHA-256: `bd08fe66a8160483e741e45928aea22ee0f6699d655da9fca43bd98b33b9140d`). Established hypotheses H1–H4, comparison set, metrics, and numeric pass criteria C1–C6. Added regression test `test_p3_b_preregistration_hash`. Tagged `prereg-p3`.
  - [x] **Stage P3-C (`gate-p3-c`)**: Extended NEXUS with multi-sense lexicon, 3 restructuring operators (Split, Merge, Exception), real incremental belief revision with append-only revision log, retention, and explainability citing restructuring events (`nexus/learner/restructure_learner.py`, 16 unit tests in `nexus/tests/test_stage_p3_learner.py`). Tagged `gate-p3-c`.
  - [x] **Stage P3-D (`pre-run-p3`)**: Pre-registered one-shot evaluation on W1–W5 across 50 fresh seeds (`2000..2049`). 9 models evaluated. Proved Restructuring decisively outperforms NoRestructure everywhere ($\Delta = +0.045$ to $+0.185$, $p < 0.005$) and Baseline D ($\Delta = +0.108$ to $+0.274$). Evaluated criteria C1–C6 honestly. Delivered `PHASE3_REPORT.md`.
  - [x] **Stage P3-E (`LIMITATIONS_P3.md`)**: Exploratory analyses covering label noise robustness (0/2/10/20%), sample efficiency curves across training stream, and SQLite persistence roundtrip (`nexus/storage/restructure_repository.py`, `nexus/tests/test_persistence_p3.py`). Delivered `LIMITATIONS_P3.md` and updated `RERUNS.md`.

---

## Exact Commands to Verify
```bash
# Run complete test suite (97 passing tests)
.venv/bin/pytest -v

# Run Stage P3-A audit suite across 50 seeds (2000..2049)
.venv/bin/python -m nexus.eval.audit_p3

# Run Phase 3 exploratory analyses
PYTHONPATH=. .venv/bin/python nexus/experiments/phase3_exploratory.py

# Inspect Phase 3 evaluation results
cat nexus/experiments/results/phase3_eval.json
```
