# NEXUS-0 Project Status

**Last Updated**: 2026-10-24
**Current Stage**: STAGE B (Phase 1 Learner — Grammar + Meaning Induction)
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
  - [x] **GATE A**: All 50 tests pass, `AUDIT_0_6.md` written, ready for tag `gate-A`.

---

## Active Stage: STAGE B (Phase 1 Learner)
- [ ] **B1**: Categorial grammar hypothesis space (`N, NP, S`, slash depth $\le 2$, forward/backward application only), lexical semantic hypothesis space, and MDL + likelihood scorer ($\varepsilon=0.02$).
- [ ] **B2**: Online beam search ($K=200$) with true-vs-false contrastive proposal generation, Bayesian model averaging over beam, and JSONL experience log for `correct()` replay.
- [ ] **B3**: `explain(sentence, scene)` and `what_changed(example_id)` with provenance + explanation faithfulness intervention test.
- [ ] **B4**: Baseline B (Cross-Situational Only: IBM Model 1 / co-occurrence bag-of-meanings) and Baseline C (Greedy Top-1 No-Revision Learner).
- [ ] **B5**: Lexicon-permutation invariance test over 20 seeds, source scan test (no invented words or oracle imports in `nexus/learner/`), scoring/pruning/explain/faithfulness unit tests.
- [ ] **GATE B**: All Stage B tests pass, tuned on `dev` only, tag `gate-B`.

---

## Open Issues
- None in Stage A (all 6 review findings resolved and verified).

---

## Commands to Resume
```bash
# Run full test suite
.venv/bin/pytest -v

# Re-run Stage A audit
.venv/bin/python -m nexus.experiments.stage_a_audit
```
