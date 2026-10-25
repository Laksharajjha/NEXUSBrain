# NEXUS-0: Evaluation & Test-Split Rerun Log (`RERUNS.md`)

**Stage C Pre-Run Tag**: `pre-run` (`8f7102b`)  
**Stage C Pre-Registered Reruns**: `0`  
**Cumulative Test-Split Evaluations Across All Stages**: `19,380` (`320` Stage C + `80` Stage D + `6,380` Phase 1.5 + `2,400` Phase 1.6 aborted 4a + `10,200` Phase 1.6 final 4b)

---

## Execution Audit Trail

| Run # | Stage / Classification | Git Tag / Commit | Seeds | Test-Split Evals | Command | Status & Notes |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **1 (Primary)** | Stage C (`PREREGISTERED`) | `pre-run` (`8f7102b`) | `0..19` ($N=20$) | `320` (`4` test splits $\times 20$ seeds $\times 4$ models) | `.venv/bin/python -m nexus.experiments.phase1_eval` | **COMPLETED (0 Pre-Registered Reruns)**. Frozen pass criteria in `PREREGISTRATION.md` evaluated once. |
| **2** | Stage D (`EXPLORATORY`) | `99210ed` | `0..4` ($N=5$) | `80` | `.venv/bin/python -m nexus.experiments.stage_d_exploratory` | Exploratory post-gate-C ablations D1–D8 and SQLite round-trip D9 (`LIMITATIONS.md`). |
| **3** | Phase 1.5 (`EXPLORATORY`) | Post-`99210ed` | `0..19` ($N=20$) | **`6,380`** (`80` conditions $\times 20$ seeds $\times 4$ test splits, minus `20` in 3.7) | `.venv/bin/python -m nexus.experiments.phase1_5_adversarial` | **Phase 1.5 Adversarial Validation (`PHASE1_5_REPORT.md`)**: Part 1 Spy Audit (`160`), Part 2 Negative Controls (`240`), Part 3 Difficulty Ladder (`4,140`), Part 4 Stronger Baselines (`1,440`), Part 5 MDL Bit-Cost Prior Audit (`400`). |
| **4a** | Phase 1.6 (`EXPLORATORY`), **ABORTED** | uncommitted (post-`11ef074`) | **fresh `1000..1049`** ($N=50$) | `2,400` (Task 1: 3 channels × 4 conditions × 50 seeds × 4 test splits) | `.venv/bin/python -m nexus.experiments.phase1_6_mechanisms` | Task 1 finished, then the runner crashed in Task 2 (`AttributeError: GrammarConfig has no clause_order`). **Nobody printed, saved, or looked at the Task 1 numbers.** Between 4a and 4b the learner code did not change. The only edits were the runner's grammar-attribute lookup fix and new per-task checkpointing. It is still counted: the fresh Task 1 test splits have been evaluated **twice** (4a + 4b). |
| **4s** | Phase 1.6 smoke test | uncommitted | burned `0..1` | n/a (burned seeds) | `... phase1_6_mechanisms 0 2` | Crash check of the runner on burned seeds, done before touching the fresh seeds again. |
| **4b** | Phase 1.6 (`EXPLORATORY`), final pass | uncommitted (post-`11ef074`) | **fresh `1000..1049`** ($N=50$) | **`10,200`** (`2,400` Task 1 + `7,800` Task 2; Tasks 3 and 4 evaluated on `dev` only) | `.venv/bin/python -m nexus.experiments.phase1_6_mechanisms` | **Completed full 50-seed execution of Phase 1.6 (`PHASE1_6_REPORT.md`)**. Evaluated evidence channels (Task 1), matched-$N$ comparison (Task 2), beam search in garden-path ordering (Task 3), and MDL priors in variable-length worlds (Task 4). Verified zero label leak in `NEXUS-Truth`, confirmed beam necessity under garden paths, confirmed MDL necessity under filler/null words, and proved structural advantage of grammar induction over Baseline D at scale. |
