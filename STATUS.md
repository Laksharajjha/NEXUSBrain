# NEXUS-0 Project Status

**Last Updated**: 2026-10-24
**Current Stage**: STAGE C (Complete — tagging `gate-C`) -> Ready for STAGE D (Exploratory Analyses & Persistence)
**Current Branch**: `feature/phase-0-toy-world-eval`
**PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`

---

## Completed Stages
- **Phase 0**: Toy world, logic, splits, evaluation runner, Baseline A (Memorizer).
- **Phase 0.5**: Initial Oracle ceiling, SentenceOnly & SceneOnly baselines, per-seed randomization, frozen `PREREGISTRATION.md` (`prereg-v1`).
- **STAGE A (Phase 0.6 — Audit Patch, `gate-A`)**:
  - [x] **A1**: Reported `n` per split in `AUDIT_0_6.md`, `README.md`, and `nexus/eval/report.py`; 50-seed pooled Student-$t$ interval test for `SentenceOnly` and `SceneOnly` (`0.500 ± 0.000`, 0 failures).
  - [x] **A2**: Feature-level label-balance audit across all 6 splits and 63 features (`P(feature|true)` vs `P(feature|false)`) with permutation tests + Holm-Bonferroni & Benjamini-Hochberg corrections; fixed generator binding/value-swap symmetry; reran 20-seed benchmark table.
  - [x] **A3**: Verified novel-kind presence is balanced (`1.000` vs `1.000`) on `test_novel_word_in_known_frame`; enumerated consistent candidates from the 2-example support stream in the 15-predicate Phase 1 hypothesis space (`candidate_count == 1`, `ideal_observer_ceiling == 1.000`).
  - [x] **A4**: Implemented `StubbornOracle` (`0.000` on diagnostic probes) and `CorrectedOracle` (`0.000` before corrections -> `1.000` after corrections).
  - [x] **A5**: Implemented independent second model checker `naive_evaluate` (`itertools.permutations`) in `nexus/world/logic.py`; verified on 20 hand-crafted golden cases + 1,000 random `(scene, LF)` pairs.
  - [x] **A6**: Identifiability report across 20 seeds verifying $\ge 183$ role-swap `chases` contrasts, $\ge 253$ NP modifier-attachment contrasts, and $\ge 294$ examples per construction (`flagged_under_5 == []`).
  - [x] **A7**: Enforced even `n_dev_per_construction`, verified `n_train_stacked_nps == 0`, updated `README.md`, locked `PREREGISTRATION.md` SHA-256 in CI (`test_a7_preregistration_hash_and_small_fixes`), tagged `prereg-v1`.
  - [x] **A8**: Implemented `ExhaustiveIdealObserver` on reduced world ($6! = 720$ bijections), achieving `1.000` ceiling across all splits.
  - [x] **GATE A**: All 50 Stage A tests pass, `AUDIT_0_6.md` written, tagged `gate-A`.
- **STAGE B (Phase 1 Learner — `gate-B`)**:
  - [x] **B1–B5**: Implemented categorial grammar (`nexus/learner/categorial.py`), MDL beam search learner (`nexus/learner/nexus_learner.py`), explainability & faithfulness interventions (`nexus/learner/explain.py`), Baselines B (`CrossSituationalLearner`) & C (`NoRevisionLearner`), and 20-seed lexicon-permutation invariance + source scan tests (`nexus/tests/test_stage_b_learner.py`).
  - [x] **GATE B**: All 56 tests pass, tagged `gate-B`.
- **STAGE C (Pre-Registered Evaluation — `gate-C`)**:
  - [x] **C1**: Re-read `STATUS.md` and `PREREGISTRATION.md`, tagged `pre-run` (`8f7102b`).
  - [x] **C2**: Ran Baseline A (`Memorizer`), Baseline B (`CrossSituationalLearner`), Baseline C (`NoRevisionLearner`), `NEXUSLearner`, `SentenceOnly`, `SceneOnly` on 20 seeds (`seeds 0..19`, `world_mode=randomized`) in one shot (`0` reruns).
  - [x] **C3**: Computed paired bootstrap 95% CIs (10,000 resamples) vs `max(A, B)` on `test_novel_combination` (`+0.500, CI [+0.500, +0.500]`) and `test_novel_word_in_known_frame` (`+0.500, CI [+0.500, +0.500]`).
  - [x] **C4**: Extracted top-hypothesis lexicons for seeds 0, 1, 2 alongside ground-truth lexicons (`15/15` exact matches); categorized held-out items per split via `explain()`.
  - [x] **C5**: Wrote `PHASE1_REPORT.md` and `RERUNS.md`, ready for tag `gate-C`.

---

## Active Stage: STAGE D (Exploratory Analyses & SQLite Persistence)
- [ ] **D1**: Beam width $K \in \{1, 5, 25, 100, 200, 500\}$.
- [ ] **D2**: Noise parameter $\varepsilon \in \{0.001, 0.01, 0.02, 0.05, 0.1\}$.
- [ ] **D3**: Learning curves at $N \in \{50, 100, 200, 400, 800, 1600\}$ vs baselines + gap to A8 ideal observer on reduced world.
- [ ] **D4**: Hypothesis-space ablations (`given_lexicon`, `given_syntax`, `max_slash_depth=1`).
- [ ] **D5**: Label noise $\{0\%, 2\%, 5\%, 10\%, 20\%\}$ before/after corrections.
- [ ] **D6**: New held-out probes (role reversal on symmetric vs asymmetric relations, 3-adjective NPs, unseen `adj + novel_noun`).
- [ ] **D7**: Calibration reliability diagrams + Brier decomposition.
- [ ] **D8**: Failure taxonomy (`LIMITATIONS.md`).
- [ ] **D9**: SQLite persistence round-trip (`nexus/storage/repository.py` + `schema.sql`): train, save to SQLite, reload in fresh process, verify identical predictions and provenance.

---

## Open Issues
- None.

---

## Commands to Resume
```bash
# Run full test suite
.venv/bin/pytest -v

# Run Stage D exploratory suite
.venv/bin/python -m nexus.experiments.stage_d_exploratory
```
