# NEXUS-0: Evaluation & Test-Split Rerun Log (`RERUNS.md`)

**Stage C Pre-Run Tag**: `pre-run` (`8f7102b`)  
**Stage C Pre-Registered Reruns**: `0`  
**Cumulative Test-Split Evaluations Across All Stages**: `6,780` (`320` Stage C + `80` Stage D + `6,380` Phase 1.5 Exploratory Adversarial Validation)

---

## Execution Audit Trail

| Run # | Stage / Classification | Git Tag / Commit | Seeds | Test-Split Evals | Command | Status & Notes |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **1 (Primary)** | Stage C (`PREREGISTERED`) | `pre-run` (`8f7102b`) | `0..19` ($N=20$) | `320` (`4` test splits $\times 20$ seeds $\times 4$ models) | `.venv/bin/python -m nexus.experiments.phase1_eval` | **COMPLETED (0 Pre-Registered Reruns)**. Frozen pass criteria in `PREREGISTRATION.md` evaluated once. |
| **2** | Stage D (`EXPLORATORY`) | `99210ed` | `0..4` ($N=5$) | `80` | `.venv/bin/python -m nexus.experiments.stage_d_exploratory` | Exploratory post-gate-C ablations D1–D8 and SQLite round-trip D9 (`LIMITATIONS.md`). |
| **3** | Phase 1.5 (`EXPLORATORY`) | Post-`99210ed` | `0..19` ($N=20$) | **`6,380`** (`80` conditions $\times 20$ seeds $\times 4$ test splits, minus `20` in 3.7) | `.venv/bin/python -m nexus.experiments.phase1_5_adversarial` | **Phase 1.5 Adversarial Validation (`PHASE1_5_REPORT.md`)**: Part 1 Spy Audit (`160`), Part 2 Negative Controls (`240`), Part 3 Difficulty Ladder (`4,140`), Part 4 Stronger Baselines (`1,440`), Part 5 MDL Bit-Cost Prior Audit (`400`). |
