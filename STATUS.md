# NEXUS-0 Project Status

**Last Updated**: 2026-10-20
**Current Stage**: STAGE A (Audit Patch — Phase 0.6)
**Current Branch**: `feature/phase-0-toy-world-eval`
**PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`

---

## Completed Stages
- **Phase 0**: Toy world, logic, splits, evaluation runner, Baseline A (Memorizer).
- **Phase 0.5**: Initial Oracle ceiling, SentenceOnly & SceneOnly baselines, per-seed randomization, frozen `PREREGISTRATION.md`.

---

## Active Stage: STAGE A (Audit Patch)
Addressing the 6 independent review findings via tasks A1–A8 before Gate A:
- [ ] **A1**: Report `n` per split; pooled 50-seed t-interval test for `SentenceOnly` and `SceneOnly` (95% CI must contain 0.50 on every test split).
- [ ] **A2**: Feature-level label-balance audit on every split (`P(feature|true)` vs `P(feature|false)`) with permutation tests + Holm/BH correction; diagnose and fix generator artifacts (specifically relation swap/endpoint pattern in `test_novel_combination` and `test_longer`); rerun 20-seed table.
- [ ] **A3**: `test_novel_word_in_known_frame` audit: verify novel-kind presence is balanced in true/false test scenes; enumerate consistent meanings from the 2-example support stream within Phase 1 hypothesis space; compute candidate count and ideal-observer accuracy ceiling; add tests.
- [ ] **A4**: Implement `StubbornOracle` (1 corrupted lexical entry, ignores corrections -> must score < 0.5 on diagnostic probes) and `CorrectedOracle` (applies corrections -> must score 1.0).
- [ ] **A5**: Second naive model checker (`itertools.permutations` / `itertools.product` over object assignments); differential property test on 1,000 random scenes/LFs + 20 hand-verified golden cases.
- [ ] **A6**: Identifiability report across constructions and randomized NP/clause/particle orders: count training examples distinguishing correct analysis from main alternatives; flag any < 5.
- [ ] **A7**: Small fixes: even per-construction dev count; check if any training NP has both size and color modifiers; README wording ("modifier stacking", feedback wording); git tag `prereg-v1` + test verifying `PREREGISTRATION.md` SHA-256; check if any preregistered threshold is infeasible given A3 ceiling.
- [ ] **A8**: Exhaustive Bayesian ideal observer on reduced world (2 kinds, 2 colors, 1 relation, fixed order) small enough to enumerate; compute best achievable accuracy per split.
- [ ] **GATE A**: All tests pass, A1–A8 documented in `AUDIT_0_6.md`, no unresolved artifact, tag `gate-A`.

---

## Open Issues / Review Findings Being Investigated
1. `SceneOnly` scored `0.458 +/- 0.034` on `test_novel_combination` (~5 SE below 0.50) and `0.539 +/- 0.021` on `test_longer` (~8 SE above 0.50). Need to diagnose the exact generator artifact causing this in `generator.py` / `splits.py`.
2. `n` per split not explicitly reported in summary tables.
3. Correction metric validation requires `StubbornOracle` vs `CorrectedOracle`.
4. Single model checker shared between generator and oracle needs independent naive checker validation.
5. Word-order identifiability from training data needs quantitative verification.
6. Novel-word 2-example support stream uniqueness and ideal-observer ceiling need exact enumeration.

---

## Commands to Resume
```bash
# Run full test suite
.venv/bin/pytest -v

# Check git status and tags
git status
git tag -l
```
