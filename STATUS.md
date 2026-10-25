# NEXUS-0 Project Status

**Last Updated**: 2026-10-25
**Current Stage**: PHASE 1.6 COMPLETED (`gate-1.6`, `EXPLORATORY`)
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
- **PHASE 1.6 (`PHASE1_6_REPORT.md` — `EXPLORATORY`)**:
  - [x] **Task 0 (Fresh Evaluation Data)**: Seed range `1000..1049` ($N=50$ fresh seeds). Test-split evaluations tracked in `RERUNS.md`.
  - [x] **Task 1 (Separate Evidence Channels)**: Implemented `NEXUS-Truth`, `NEXUS-Cooc`, and `NEXUS-Both`. Verified `NEXUS-Truth` drops to exactly chance ($0.497 \pm 0.021$) under shuffled labels (Control 2a), confirming zero label leak. Showed `NEXUS-Cooc` achieves $0.966$ without labels but leaves an unbridgeable $\approx 5.6\%$ gap on novel combinations.
  - [x] **Task 2 (Matched-$N$ Comparison)**: Evaluated $N \in \{20, 50, 100, 200, 400, 1600\}$ across 50 seeds with paired bootstrap CIs. Showed Baseline D outperforms NEXUS at $N \le 400$ by bypassing syntax, but hits a ceiling ($0.774$ on longer sentences, $0.853$ on contradictions). At $N=1600$, `NEXUS-Both` reaches $1.000$ across all splits ($+0.226$ over Baseline D on longer sentences).
  - [x] **Task 3 (Does the Beam Matter?)**: Tested standard vs. garden-path stimulus order ($K \in \{1, 5, 50, 200\}$). Showed beam width is inert under i.i.d. random order, but earns its place decisively under garden paths ($K \ge 50$ achieves $0.592$ vs. $0.501$ for greedy $K=1$, $+0.0915$ $[+0.0753, +0.1080]$, $p < 10^{-15}$). Evaluated Baseline C `commit_after_n` sensitivity.
  - [x] **Task 4 (Does MDL Matter?)**: Tested variable-length worlds (filler/null words, homonyms/polysemy, synonyms) comparing `log2_bits` vs. `linear` vs. `flat` ($\alpha=0$). Proved `log2_bits` earns its place under filler words ($+0.0497$ $[+0.0325, +0.0674]$ dev gain with smaller model) and polysemy (compressing from $17.20$ to $12.96$ entries).
  - [x] **Task 5 (Report & Documentation)**: Delivered `PHASE1_6_REPORT.md` with comprehensive tables, bootstrap CIs, mechanism verdicts, and architectural recommendations for Phase 3. Updated `RERUNS.md` and test suite (71 passing tests).

---

## Exact Commands to Verify
```bash
# Run full test suite (71 passing tests)
.venv/bin/pytest -v

# Run Phase 1.6 mechanisms evaluation suite (uses cached checkpoint if already computed)
.venv/bin/python -m nexus.experiments.phase1_6_mechanisms
```
