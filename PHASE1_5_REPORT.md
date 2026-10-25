# NEXUS-0 — PHASE 1.5 REPORT: Adversarial Validation & Failure-Point Audit (`EXPLORATORY`)

> **Classification:** `EXPLORATORY` (Post-Gate-C Adversarial Audit)
> **Pre-Registration Freeze:** [`PREREGISTRATION.md`](./PREREGISTRATION.md) was **not** edited (`SHA-256 = ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`).
> **No-Tuning Rule:** All experiments in this report evaluate the exact frozen Phase 1 [`NEXUSLearner`](./nexus/learner/nexus_learner.py) across 20 randomized seeds (`0..19`, `world_mode=randomized`) with zero hyperparameter or heuristic tuning.
> **Raw Data Artifact:** [`nexus/experiments/results/phase1_5_adversarial.json`](./nexus/experiments/results/phase1_5_adversarial.json)

---

## Executive Summary: Critical Findings

1. **Part 1 (Protocol & Spy Audit — PASS with Distributional Qualification):**
   - A [`SpyLearnerWrapper`](./nexus/experiments/phase1_5_adversarial.py) wrapping [`NEXUSLearner`](./nexus/learner/nexus_learner.py) across all 20 seeds (`32,680` `observe()` calls, `30,000` `predict()` calls) confirmed **0 held-out leakage violations** and **0 `predict()` state mutations**.
   - Static AST inspection of `nexus/learner/` confirmed **0 accesses** to generator-only fields (`lf`, `construction`, `contrast_group`, `tags`, `ground_truth`, `mislabelled`, `plan`).
   - Mechanistically, `_propose_word_candidates` does **not** pair examples by `contrast_group` or identical token strings (shuffling training order yields `1.000 ± 0.000` on every split). However, it **does** rely distributionally on the generator's near-miss negative scenes.
2. **Part 2 (Negative Controls — LEAK DETECTED IN CONTROL 2a):**
   - **Control 2b (Shuffled Scenes):** Drops to **`0.499 ± 0.012`** on `test_novel_combination` (`0.497 ± 0.032` on `dev`). **PASS.**
   - **Control 2c (Cross-Seed Lexicon):** Drops to **`0.500 ± 0.004`** on `test_novel_combination` (`0.502 ± 0.020` on `dev`). **PASS.**
   - **Control 2a (Shuffled Truth Labels):** **FAILS TO DROP TO CHANCE** — achieves **`0.657 ± 0.109`** on `test_novel_combination`, **`0.771 ± 0.081`** on `dev`, and **`0.756 ± 0.097`** on `train`.
   - **Root Cause of Control 2a Leak:** Two mechanisms interact:
     1. *Generator near-miss scene correlation:* In [`WorldGenerator`](./nexus/world/generator.py), both the `True` scene and the near-miss `False` scene for a sentence $u$ are built around $u$'s true meaning $M(u)$ by mutating only one attribute or relation role. Consequently, when word $w$ appears in `obs.tokens`, its true predicate $p^*$ is present in `obs.scene` ~80–85% of the time **regardless of whether `obs.label` is `True` or `False`**.
     2. *Asymmetric positive-bias in `_propose_word_candidates`:* For nouns in multi-word sentences (`nexus_learner.py` lines 600–606), the proposer only updates scores when `if obs.label:` and has **no `else` branch for `not obs.label`**. For modifiers (lines 735–747), `holds == True` is rewarded by `+1.0 * log_hit` (`+0.673`) when `obs.label == True` but penalized by only `0.1 * log_miss` (`-0.322`) when `obs.label == False`. Under 50/50 shuffled random labels, the expected score drift per occurrence is $+0.175 > 0$ whenever a predicate co-occurs with $w$ in `obs.scene`, allowing `_propose_word_candidates` to induce most of the lexicon from **unlabeled `(scene, tokens)` co-occurrence**!
3. **Part 3 (Difficulty Ladder — Exact Breaking Points `< 0.90` and `< 0.70`):**
   - `1.000 ± 0.000` is an artifact of the clean, 1-to-1, 17-word, $N=1600$ regime. Across the 8 stress ladders, [`NEXUSLearner`](./nexus/learner/nexus_learner.py) breaks below `0.90` and `0.70` under small sample sizes ($N \le 100$ stratified / $N \le 400$ curriculum prefix), vocabulary scaling (`48 kinds`), homonymy/polysemy (`1` and `3` shared surface forms), synonymy on 2-shot word learning (`0.582`), and multi-clause / 3-adjective same-family sentences (`0.500`, `0%` parse rate).
4. **Part 4 (Stronger Baselines):**
   - **Baseline D (`PositionAwareCrossSituationalLearner`)**—which uses word-position and per-object nearest-noun adjacency **without any categorial grammar search**—achieves **`0.952 ± 0.066` on `dev`, `0.918 ± 0.112` on `test_novel_combination`, and `0.940 ± 0.083` on `test_novel_word_in_known_frame`**. This proves that over 90% of `test_novel_combination` is solvable by local per-object adjacency bundling without phrase-structure grammar search; categorial slash directionality (`N/N` vs `N\N`) only provides the final ~8–19% margin on ambiguous multi-NP attachment (`0.918 -> 1.000`) and stacked modifiers in `test_longer` (`0.808 -> 1.000`).

---

## Part 1: Leakage and Protocol Audit

### 1.1 Spy/Wrapper Verification & Training-Order Permutation (20 Seeds)

| Audit Check | Metric / Split | Result (20 seeds) | Status |
|---|---|---:|---|
| `SpyLearnerWrapper` `observe()` calls audited | Total `observe()` calls | `32,680` | Audited |
| `SpyLearnerWrapper` `predict()` calls audited | Total `predict()` calls | `30,000` | Audited |
| Held-out example ID present in log before `predict()` | `spy_leak_violations` | **`0`** | **PASS** |
| Learner state (`_observations`, `experience_log_jsonl`, `_beam`) mutated during `predict()` | `spy_mutation_violations` | **`0`** | **PASS** |
| Shuffled training order (`part1_shuffled_train_order`, separating true/false twins) | `train` / `dev` / all 4 `test_*` splits | **`1.000 ± 0.000`** | **PASS** |

### 1.2 Exact Learner-Protocol Fields Read by `nexus/learner/`

Static AST inspection of [`categorial.py`](./nexus/learner/categorial.py), [`explain.py`](./nexus/learner/explain.py), and [`nexus_learner.py`](./nexus/learner/nexus_learner.py) confirms the learner reads **only** the following `Learner`-protocol attributes:

| Input Type | Fields / Methods Read in `nexus/learner/` | Forbidden Generator Fields Checked (0 Hits) |
|---|---|---|
| `Observation` | `example_id`, `scene`, `tokens`, `label`, `source` | `lf`, `construction`, `contrast_group`, `tags` |
| `Correction` | `target_example_id`, `corrected_label`, `source`, `reason` | `ground_truth`, `mislabelled`, `plan` |
| `Scene` | `objects`, `relations`, `holds`, `to_json` | — |
| `SceneObject` | `id`, `kind`, `color`, `size`, `value` | — |
| `SceneRelation` | `name`, `subject`, `object` | — |

### 1.3 Plain Statement on Paired-Twin Usage

- **Does `_propose_word_candidates` pair examples by twin ID or identical words?**
  **No mechanistically, Yes distributionally.**
  - *Mechanistically:* `_propose_word_candidates` iterates over `_observations` individually. It never groups observations by `contrast_group`, never searches for pairs of observations with identical `tokens`, and is completely invariant to permuting the training stream (`1.000 ± 0.000`).
  - *Distributionally:* Each `False` scene in the generator's contrastive/near-miss stream (`p_contrast_pair = 0.5`, `p_near_miss_false = 0.5`) is created by mutating **one** attribute or relation endpoint of a `True` scene for that sentence. `_propose_word_candidates` relies on these minimal-edit `False` scenes in two ways:
    1. To distinguish asymmetric `chases(x0,x1)` from symmetric `chases:sym(x0,x1)` (which requires seeing a `False` scene where `chases(B, A)` holds in the reverse direction).
    2. Because non-mutated words in a near-miss `False` scene still have their true referents present in `obs.scene`, `obs.tokens` and `obs.scene` remain strongly correlated even when `obs.label` is ignored or shuffled (causing the Control 2a leak below).

---

## Part 2: Negative Controls (20 Seeds, Mean ± Std)

| Negative Control | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` (post) | Verdict |
|---|---|---|---|---|---|---|---|
| **2a. Shuffled Truth Labels** | **0.756 ± 0.097** | **0.771 ± 0.081** | **0.657 ± 0.109** | **0.741 ± 0.105** | **0.614 ± 0.117** | **0.705 ± 0.183** | **LEAK (Above Chance)** |
| **2b. Shuffled Scenes** | 0.499 ± 0.017 | 0.497 ± 0.032 | 0.499 ± 0.012 | 0.499 ± 0.036 | 0.500 ± 0.001 | 0.501 ± 0.008 | **PASS (~0.50)** |
| **2c. Cross-Seed Lexicon** | 0.498 ± 0.013 | 0.502 ± 0.020 | 0.500 ± 0.004 | 0.501 ± 0.006 | 0.500 ± 0.000 | 0.500 ± 0.002 | **PASS (~0.50)** |

### Diagnosis of the Control 2a Leak (`2a_shuffled_labels`)

Why does `NEXUSLearner` score `0.657 ± 0.109` on `test_novel_combination` and `0.771 ± 0.081` on `dev` when training truth labels are randomly permuted (`50% True / 50% False`), while dropping to `0.499` when scenes are permuted (`2b`)?

1. **Unlabeled Scene-Token Co-occurrence in the Generator:**
   In `WorldGenerator`, every example $(s_i, u_i, y_i)$ is generated by first sampling a target meaning $M(u_i)$ and rendering sentence $u_i$. If $y_i = \text{True}$, $s_i$ satisfies $M(u_i)$. If $y_i = \text{False}$, $s_i$ is a falsification of $M(u_i)$ that leaves all other mentioned objects/attributes intact (e.g., for `red cube chases blue ball`, a `False` scene where `chases` is reversed still contains a `red cube` and a `blue ball` 100% of the time). Therefore, the conditional probability $P(\text{predicate } p \in s_i \mid w \in u_i)$ is `~0.80–0.85` for the true predicate $p^*$ and `~0.35–0.45` for distractor predicates, **independent of $y_i$**.
2. **Asymmetric Positive-Only Weighting in `_propose_word_candidates`:**
   - In Phase 2 (Noun induction, [`nexus_learner.py`](./nexus/learner/nexus_learner.py) lines 600–606), on multi-word sentences (`len(core) >= 2`), the learner executes:
     ```python
     if obs.label:
         if present:
             score += 0.5 * log_hit
         else:
             score += 1.5 * log_miss
     ```
     There is **no `else` branch when `obs.label` is `False`**. When `obs.label` is randomly shuffled (`P(obs.label=True) = 0.5`), the random 50% of examples with `obs.label = True` act as an **unlabeled co-occurrence filter** that rewards predicates present in `obs.scene` whenever $w \in \text{tokens}$.
   - In Phase 3 (Modifier induction, lines 735–747), when `holds` is True on the adjacent noun, `obs.label == True` adds `+1.0 * log_hit` (`+0.673`), whereas `obs.label == False` adds `0.1 * log_miss` (`-0.322`). Under 50/50 random labels, any attribute that co-occurs with $w$ has a positive expected drift of $0.5(0.673 - 0.322) = +0.175$ per observation.
   - Syntactic category assignment (`core_len1_counts`, `pre_count` vs `post_count`, verb position relative to nouns) depends **only on `obs.tokens`**, never on `obs.label`.
   - At test time, `predict(scene, tokens)` model-checks the induced lexicon's logical form against the test scene (`evaluate_parsed_lf`), yielding `0.657–0.771` accuracy without ever needing true training labels.

---

## Part 3: Difficulty Ladder — Finding the Breaking Points (20 Seeds, No Tuning)

### Summary of First Conditions Where NEXUS Falls Below `0.90` and `< 0.70`

| Ladder Dimension | First Condition `< 0.90` | First Condition `< 0.70` | Primary Failure Mechanism |
|---|---|---|---|
| **3.1a Sample Size (Stratified)** | **`N = 100`** (`test_longer = 0.897 ± 0.162`) / **`N = 50`** (`novel_comb = 0.724 ± 0.157`) | **`N = 20`** (`novel_comb = 0.512 ± 0.038`, `dev = 0.540 ± 0.054`) | Sparse cross-situational coverage of 17 words across 4 constructions |
| **3.1b Sample Size (Curriculum Prefix)** | **`N = 400`** (`dev = 0.750`, `novel_comb = 0.574`, `longer = 0.500`) | **`N = 400`** (`novel_comb = 0.574 ± 0.015`, `longer = 0.500 ± 0.000`) | First 800 curriculum items contain only `np_assert`/`np_ask` (`0` transitive verbs) |
| **3.2 Random Negatives Only** | Stays `> 0.90` (`novel_comb = 0.955 ± 0.025` at `N = 400`) | Does not drop `< 0.70` | Drops `4.5%` from `1.000` due to missing role-reversed `chases(B,A)` negatives |
| **3.3 Positive-Only Training** | Stays `> 0.90` (`novel_comb = 0.943 ± 0.013`, `longer = 0.960 ± 0.015`) | Does not drop `< 0.70` | Drops `5.7%` because symmetric vs. asymmetric relations cannot be separated without negative examples |
| **3.4 Uncorrected Label Noise** | **`20% noise` at `N = 200`** (`test_longer = 0.878 ± 0.164`, `contradiction_pre = 0.895`) | Does not drop `< 0.70` at `30%` mean (`test_longer = 0.777 ± 0.121`, min seed `0.580`) | High flip rates corrupt low-frequency modifier and verb directional counts |
| **3.5 Lexicon Scaling (`N = 400`)** | **`48 kinds`** (`dev = 0.825`, `novel_comb = 0.734`, `longer = 0.697`) | **`48 kinds`** (`test_longer = 0.697 ± 0.062`) | `106` surface words at `N = 400` gives `< 4` exposures per word |
| **3.6a Homonymy / Polysemy** | **`1 homonym pair`** (`test_longer = 0.790 ± 0.034`, `contradiction = 0.870`) | **`3 homonym pairs`** (`test_longer = 0.551 ± 0.016`, `contradiction = 0.660 ± 0.203`) | `Hypothesis.entries` is a `dict[str, LexicalEntry]` (1 meaning per surface form) |
| **3.6b Synonymy (`2x` forms)** | **`2x noun synonyms`** on `test_novel_word` (`0.582 ± 0.116`) | **`2x noun synonyms`** on `test_novel_word` (`0.582 ± 0.116`) | 2-example support stream only exposes 1 of the 2 synonym forms (or 1 example each) |
| **3.7 Beyond Training Depth** | **`3-adj same-family`** (`0.500`) & **`2-clause chain`** (`0.500`) | **`3-adj same-family`** (`0.500 ± 0.000`) & **`2-clause chain`** (`0.500 ± 0.000`) | `0.0%` parse rate: sortal clash check & no clause-conjunction / relative-clause categories |
| **3.8 Beam Width `K`** | **`K = 10` at `N = 100`** (`novel_comb = 0.863`, `longer = 0.855`; `K = 1` -> `0.835`) | **Baseline C Legacy** (`0.499 ± 0.004`) | Greedy top-1 pruning discards true entries when early small-sample evidence is tied |

---

### Table 3.1: Training Set Size Ladder $N \in \{10, 20, 50, 100, 200, 400, 1600\}$ (20 Seeds)

| Sampling Regime | `N` | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---:|---|---|---|---|---|---|
| **Stratified** | `10` | 0.505 ± 0.007 | 0.503 ± 0.010 | **0.500 ± 0.001** (`<0.70`) | 0.500 ± 0.011 | 0.500 ± 0.000 | 0.520 ± 0.032 |
| **Stratified** | `20` | 0.537 ± 0.054 | 0.540 ± 0.054 | **0.512 ± 0.038** (`<0.70`) | 0.535 ± 0.052 | 0.502 ± 0.004 | 0.560 ± 0.133 |
| **Stratified** | `50` | 0.790 ± 0.141 | 0.784 ± 0.139 | **0.724 ± 0.157** (`<0.90`) | 0.777 ± 0.149 | 0.713 ± 0.178 | 0.766 ± 0.173 |
| **Stratified** | `100` | 0.928 ± 0.119 | 0.917 ± 0.128 | 0.904 ± 0.134 | 0.910 ± 0.151 | **0.897 ± 0.162** (`<0.90`) | 0.983 ± 0.060 |
| **Stratified** | `200` | 0.965 ± 0.085 | 0.964 ± 0.080 | 0.959 ± 0.086 | 0.949 ± 0.121 | 0.950 ± 0.111 | 0.964 ± 0.088 |
| **Stratified** | `400` | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **Stratified** | `1600` | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **Curriculum Prefix** | `10` | 0.510 ± 0.011 | 0.510 ± 0.013 | **0.500 ± 0.002** (`<0.70`) | 0.507 ± 0.013 | 0.500 ± 0.000 | 0.524 ± 0.045 |
| **Curriculum Prefix** | `20` | 0.543 ± 0.046 | 0.546 ± 0.044 | **0.505 ± 0.010** (`<0.70`) | 0.541 ± 0.047 | 0.500 ± 0.000 | 0.565 ± 0.074 |
| **Curriculum Prefix** | `50` | 0.672 ± 0.066 | 0.664 ± 0.065 | **0.536 ± 0.028** (`<0.70`) | 0.607 ± 0.060 | 0.500 ± 0.000 | 0.742 ± 0.153 |
| **Curriculum Prefix** | `100` | 0.732 ± 0.050 | 0.721 ± 0.049 | **0.565 ± 0.025** (`<0.70`) | 0.643 ± 0.061 | 0.500 ± 0.000 | 0.846 ± 0.154 |
| **Curriculum Prefix** | `200` | 0.759 ± 0.012 | 0.750 ± 0.000 | **0.574 ± 0.015** (`<0.70`) | 0.673 ± 0.023 | 0.500 ± 0.000 | 0.879 ± 0.149 |
| **Curriculum Prefix** | `400` | 0.759 ± 0.012 | 0.750 ± 0.000 | **0.574 ± 0.015** (`<0.70`) | 0.673 ± 0.023 | 0.500 ± 0.000 | 0.871 ± 0.146 |

---

### Table 3.2 & 3.3: Negative-Example Ablations (Random Negatives Only vs. Positive-Only, 20 Seeds)

| Training Condition | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---|---|---|---|---|---|
| **Standard (Near-Miss + Contrast Negatives, `N=400`)** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **3.2 Random Negatives Only (`N=200`)** | 0.993 ± 0.006 | 0.979 ± 0.018 | 0.964 ± 0.029 | 1.000 ± 0.000 | 0.973 ± 0.023 | 1.000 ± 0.000 |
| **3.2 Random Negatives Only (`N=400`)** | 0.990 ± 0.006 | 0.974 ± 0.015 | 0.955 ± 0.025 | 1.000 ± 0.000 | 0.967 ± 0.022 | 1.000 ± 0.000 |
| **3.2 Random Negatives Only (`N=1600`)** | 0.991 ± 0.006 | 0.975 ± 0.017 | 0.957 ± 0.028 | 1.000 ± 0.000 | 0.970 ± 0.022 | 1.000 ± 0.000 |
| **3.3 Positive-Only Training (No `False` Examples)** | 0.988 ± 0.003 | 0.967 ± 0.008 | **0.943 ± 0.013** | 1.000 ± 0.000 | 0.960 ± 0.015 | 1.000 ± 0.000 |

---

### Table 3.4: Uncorrected Training Label Noise $\eta \in \{0\%, 2\%, 10\%, 20\%, 30\%\}$ (20 Seeds)

| Budget | Noise $\eta$ | `train` (Clean Eval) | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---:|---|---|---|---|---|---|
| **`N = 1600`** | `0%` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **`N = 1600`** | `2%` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **`N = 1600`** | `10%` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **`N = 1600`** | `20%` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **`N = 1600`** | `30%` | 0.981 ± 0.044 | 0.979 ± 0.046 | 0.959 ± 0.088 | 0.971 ± 0.064 | 0.955 ± 0.095 | 0.972 ± 0.085 |
| **`N = 200`** | `0%` | 0.987 ± 0.052 | 0.987 ± 0.045 | 0.983 ± 0.060 | 0.986 ± 0.065 | 0.977 ± 0.090 | 1.000 ± 0.000 |
| **`N = 200`** | `2%` | 0.999 ± 0.003 | 0.996 ± 0.013 | 0.995 ± 0.016 | 1.000 ± 0.000 | 0.997 ± 0.008 | 0.985 ± 0.069 |
| **`N = 200`** | `10%` | 0.992 ± 0.016 | 0.985 ± 0.023 | 0.972 ± 0.048 | 0.989 ± 0.024 | 0.986 ± 0.025 | 0.984 ± 0.073 |
| **`N = 200`** | `20%` | 0.939 ± 0.094 | 0.924 ± 0.100 | 0.903 ± 0.119 | 0.900 ± 0.134 | **0.878 ± 0.164** (`<0.90`) | 0.955 ± 0.097 |
| **`N = 200`** | `30%` | **0.884 ± 0.065** | **0.873 ± 0.054** | **0.823 ± 0.069** (`<0.90`) | **0.837 ± 0.099** | **0.777 ± 0.121** (`<0.90`) | 0.932 ± 0.093 |

*(Note: Why is `NEXUSLearner` abnormally tolerant to 20–30% label noise at `N = 1600`? Because of the exact same mechanism identified in Part 2a: even when labels are flipped, unlabeled `(scene, tokens)` co-occurrence in near-miss negative scenes still points toward the true predicate!)*

---

### Table 3.5: Lexicon Scaling ($6, 12, 24, 48$ Kinds with Proportionally Scaled Attributes at Fixed $N = 400$, 20 Seeds)

| Ontology Scale | Total Vocab $|V|$ | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---:|---|---|---|---|---|---|
| **`6 kinds` (4c, 3s, 2r)** | `16` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **`12 kinds` (8c, 6s, 2r)** | `29` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **`24 kinds` (16c, 12s, 2r)** | `55` | 0.995 ± 0.011 | 0.992 ± 0.018 | 0.978 ± 0.067 | 0.994 ± 0.018 | 0.979 ± 0.040 | 0.993 ± 0.016 |
| **`48 kinds` (32c, 24s, 2r)** | `107` | 0.916 ± 0.029 | **0.825 ± 0.066** (`<0.90`) | **0.734 ± 0.106** (`<0.90`) | **0.843 ± 0.065** (`<0.90`) | **0.697 ± 0.062** (`<0.70`) | **0.865 ± 0.056** (`<0.90`) |

---

### Table 3.6: Ambiguous Words (Homonymy/Polysemy) and Synonyms (20 Seeds)

| Lexical Stress Condition | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---|---|---|---|---|---|
| **Homonymy (1 pair: `size:small` = `color:red`)** | 0.941 ± 0.008 | 0.943 ± 0.014 | 0.920 ± 0.029 | 0.934 ± 0.027 | **0.790 ± 0.034** (`<0.90`) | **0.870 ± 0.191** (`<0.90`) |
| **Homonymy (3 pairs: all 3 sizes = 3 colors)** | **0.835 ± 0.019** (`<0.90`) | **0.832 ± 0.022** (`<0.90`) | **0.787 ± 0.033** (`<0.90`) | **0.796 ± 0.030** (`<0.90`) | **0.551 ± 0.016** (`<0.70`) | **0.660 ± 0.203** (`<0.70`) |
| **Synonyms (`2x` surface forms per noun, `N=1600`)** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **0.582 ± 0.116** (`<0.70`) | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **Synonyms (`2x` forms for all concepts, `N=1600`)** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **0.578 ± 0.108** (`<0.70`) | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **Synonyms (`2x` forms for all concepts, `N=200`)** | 0.980 ± 0.031 | 0.973 ± 0.030 | 0.962 ± 0.041 | **0.601 ± 0.108** (`<0.70`) | 0.953 ± 0.051 | 0.984 ± 0.040 |

- **Why Homonymy Breaks `NEXUSLearner`:** `Hypothesis.entries` is typed as `Mapping[str, LexicalEntry]`, storing **at most one `LexicalEntry` per surface word $w$**. When `w` means both `color:red` and `size:small`, `NEXUSLearner` commits `w` to whichever sense has higher frequency (`color:red`) and fails on every sentence using `w` in its `size:small` sense (and on stacked `small red` NPs in `test_longer`, which become `w w noun` with two `color:red` modifiers that fail sortal exclusivity!).
- **Why Synonymy Breaks `test_novel_word_in_known_frame` (`0.582 ± 0.116`):** The one-shot support stream has only 2 examples (1 `True`, 1 `False`). When the novel kind is randomly realized as either `w` or `w_syn2`, half the time the 2 support examples only contain `w` (leaving `w_syn2` completely unknown on 50% of test items), or `w` gets 1 `True` example and `w_syn2` gets 1 `False` example!

---

### Table 3.7: Longer Sentences & Nested Modification Beyond Training Depth (20 Seeds)

| Structural Depth Probe | Parse Rate (Mean ± Std) | Accuracy (Mean ± Std) | Status / Failure Mechanism |
|---|---:|---:|---|
| **`depth_2adj_diff_family` (`size + color + kind`, length 5–8)** | `1.000 ± 0.000` | `1.000 ± 0.000` | **Supported:** recursive `N/N` or `N\N` application on distinct attribute families |
| **`depth_3adj_same_family` (`size + color1 + color2 + kind`)** | **`0.000 ± 0.000`** | **`0.500 ± 0.000`** (`<0.70`) | **Breaks:** sortal exclusivity in [`categorial.py`](./nexus/learner/categorial.py) blocks two modifiers of the same attribute family on one NP |
| **`depth_2clause_chain` (`NP1 rel1 NP2 rel2 NP3`, length 5–9)** | **`0.000 ± 0.000`** | **`0.500 ± 0.000`** (`<0.70`) | **Breaks:** depth-2 verbs `(S\NP)\NP` saturate to `S` after 2 NPs; grammar has no conjunction `(S\S)/S` or relative-clause `(N\N)/(S\NP)` categories |

---

### Table 3.8: Beam Width $K \in \{1, 2, 5, 10, 50, 200\}$ vs. Baseline C (`NoRevisionLearner`, 20 Seeds)

| Model / Beam Width | Regime | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---|---|---|---|---|---|---|
| **NEXUS (`K = 200`)** | `N = 100` | 0.942 ± 0.093 | 0.939 ± 0.092 | 0.936 ± 0.089 | 0.919 ± 0.158 | 0.913 ± 0.142 | 0.940 ± 0.108 |
| **NEXUS (`K = 50`)** | `N = 100` | 0.945 ± 0.104 | 0.935 ± 0.112 | 0.927 ± 0.114 | 0.926 ± 0.134 | 0.918 ± 0.156 | 0.964 ± 0.089 |
| **NEXUS (`K = 10`)** | `N = 100` | 0.895 ± 0.125 | **0.890 ± 0.121** (`<0.90`) | **0.863 ± 0.138** (`<0.90`) | 0.902 ± 0.141 | **0.855 ± 0.171** (`<0.90`) | 0.942 ± 0.109 |
| **NEXUS (`K = 5`)** | `N = 100` | 0.920 ± 0.117 | 0.910 ± 0.121 | **0.896 ± 0.131** (`<0.90`) | 0.916 ± 0.138 | **0.876 ± 0.175** (`<0.90`) | 0.918 ± 0.133 |
| **NEXUS (`K = 2`)** | `N = 100` | 0.934 ± 0.120 | 0.927 ± 0.127 | 0.918 ± 0.136 | 0.906 ± 0.182 | 0.902 ± 0.169 | 0.929 ± 0.143 |
| **NEXUS (`K = 1`)** | `N = 100` | **0.873 ± 0.134** (`<0.90`) | **0.866 ± 0.139** (`<0.90`) | **0.835 ± 0.168** (`<0.90`) | **0.898 ± 0.123** (`<0.90`) | **0.818 ± 0.185** (`<0.90`) | **0.885 ± 0.150** (`<0.90`) |
| **NEXUS (`K = 1..200`)** | `N = 1600` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **Baseline C (`NoRevision`, Fixed)** | `N = 1600` | **0.808 ± 0.111** (`<0.90`) | **0.807 ± 0.104** (`<0.90`) | **0.765 ± 0.116** (`<0.90`) | **0.787 ± 0.113** (`<0.90`) | **0.704 ± 0.141** (`<0.90`) | **0.745 ± 0.133** (`<0.90`) |
| **Baseline C (`NoRevision`, Legacy Buggy)** | `N = 1600` | **0.505 ± 0.009** (`<0.70`) | **0.507 ± 0.010** (`<0.70`) | **0.499 ± 0.004** (`<0.70`) | **0.501 ± 0.007** (`<0.70`) | **0.500 ± 0.000** (`<0.70`) | **0.500 ± 0.002** (`<0.70`) |

---

## Part 4: Stronger Baselines (20 Seeds, Same Harness)

### Table 4.1: Full Comparison Across Baselines A, B, C (Legacy & Fixed), D, E, F, and NEXUS (`N = 1600`, 20 Seeds)

| Model | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` (pre $\to$ post) |
|---|---|---|---|---|---|---|
| **Baseline A: `Memorizer`** | 1.000 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 $\to$ 0.500 |
| **Baseline B: `CrossSituationalLearner`** | 0.724 ± 0.012 | 0.579 ± 0.019 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 $\to$ 0.500 |
| **Baseline C: `NoRevisionLearner` (Legacy Buggy)** | 0.505 ± 0.009 | 0.507 ± 0.010 | 0.499 ± 0.004 | 0.501 ± 0.007 | 0.500 ± 0.000 | 0.500 $\to$ 0.500 |
| **Baseline C: `NoRevisionLearner` (Fixed, `commit=12`)** | 0.808 ± 0.111 | 0.807 ± 0.104 | 0.765 ± 0.116 | 0.787 ± 0.113 | 0.704 ± 0.141 | **0.745 $\to$ 0.745 (No Revision)** |
| **Baseline D: `PositionAwareCrossSituational`** | **0.958 ± 0.059** | **0.952 ± 0.066** | **0.918 ± 0.112** | **0.940 ± 0.083** | **0.808 ± 0.168** | 0.868 $\to$ 0.868 |
| **Baseline E: `FromScratchNeuralBaseline` (NumPy MLP)** | 0.637 ± 0.167 | 0.539 ± 0.040 | 0.485 ± 0.024 | 0.497 ± 0.020 | 0.513 ± 0.024 | 0.504 $\to$ 0.504 |
| **Baseline F1: `NEXUS + given_syntax`** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 $\to$ 1.000 |
| **Baseline F2: `NEXUS + given_lexicon`** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 $\to$ 1.000 |
| **NEXUS Phase 1 (`NEXUSLearner`)** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 $\to$ 1.000** |

### Why Baseline C (`NoRevisionLearner`) Originally Scored `0.505` on `train` (and What Fixing It Shows)

1. **Three Compounding Bugs in Original [`no_revision.py`](./nexus/baselines/no_revision.py):**
   - **Bug 1 (Committing on `False` Initial Observations):** When an uncommitted word first appeared in a `False` observation (`observation.label == False`), any absent predicate trivially evaluated to `False == observation.label`, immediately committing the word to the alphabetically first absent predicate (`kind:ball` or `color:blue`).
   - **Bug 2 (Single-Scene Multi-Object Ambiguity at `n_obs = 1`):** Even on a `True` 1-token observation, each scene contains 2–4 objects; committing on the very first observation picks a distractor object ~65% of the time.
   - **Bug 3 (Blind Category `"N"` Fallback):** Whenever backtracking failed or hit 250 steps on a multi-word sentence, lines 131–134 executed `self._committed[w] = w_cands[0]`, which unconditionally assigned syntactic category `"N"` (`kind:*`) to modifiers, transitive verbs, and the question particle. Once modifiers and verbs were assigned `"N"`, `parse_tokens` returned `None` (`p_true = 0.5`) on every multi-token sentence, leaving only 1-token sentences parseable (`0.505` train accuracy).
2. **After Fix (`C_no_revision_fixed`):**
   - Committing each word greedily after `commit_after_n = 12` exposures using the same local candidate scorer as `NEXUSLearner` (and locking committed entries permanently) raises `train` accuracy from `0.505` to `0.808 ± 0.111` and `test_novel_combination` from `0.499` to `0.765 ± 0.116`.
   - However, because `NoRevisionLearner` freezes words before transitive frames are fully disambiguated and ignores `correct()`, it remains `19–29%` below `NEXUSLearner` across splits and shows **zero recovery (`0.745 -> 0.745`)** on `test_contradiction`.

### Table 4.2: Baseline F Small-Sample Separation of Lexicon Search vs. Syntax Search (`N = 10, 20, 50, 100`, 20 Seeds)

| Sample Size `N` | Variant | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---:|---|---|---|---|---|---|
| **`N = 10`** | `NEXUS (Full Search)` | 0.503 ± 0.010 | 0.500 ± 0.001 | 0.500 ± 0.011 | 0.500 ± 0.000 | 0.520 ± 0.032 |
| **`N = 10`** | `F1: given_syntax (Induce Lexicon)` | 0.579 ± 0.043 | 0.517 ± 0.031 | 0.539 ± 0.029 | 0.503 ± 0.010 | 0.855 ± 0.142 |
| **`N = 10`** | `F2: given_lexicon (Induce Syntax)` | **0.702 ± 0.057** | **0.599 ± 0.069** | **0.648 ± 0.061** | **0.522 ± 0.028** | **0.903 ± 0.087** |
| **`N = 20`** | `NEXUS (Full Search)` | 0.540 ± 0.054 | 0.512 ± 0.038 | 0.535 ± 0.052 | 0.502 ± 0.004 | 0.560 ± 0.133 |
| **`N = 20`** | `F1: given_syntax (Induce Lexicon)` | 0.704 ± 0.099 | 0.623 ± 0.107 | 0.668 ± 0.097 | 0.554 ± 0.072 | 0.903 ± 0.124 |
| **`N = 20`** | `F2: given_lexicon (Induce Syntax)` | **0.809 ± 0.075** | **0.729 ± 0.085** | **0.753 ± 0.091** | **0.598 ± 0.082** | **0.925 ± 0.139** |
| **`N = 50`** | `NEXUS (Full Search)` | 0.784 ± 0.139 | 0.724 ± 0.157 | 0.777 ± 0.149 | 0.713 ± 0.178 | 0.766 ± 0.173 |
| **`N = 50`** | `F1: given_syntax (Induce Lexicon)` | 0.915 ± 0.057 | 0.876 ± 0.078 | 0.889 ± 0.074 | 0.854 ± 0.129 | 1.000 ± 0.001 |
| **`N = 50`** | `F2: given_lexicon (Induce Syntax)` | **0.947 ± 0.046** | **0.928 ± 0.049** | **0.938 ± 0.048** | **0.878 ± 0.119** | **1.000 ± 0.000** |
| **`N = 100`** | `NEXUS (Full Search)` | 0.917 ± 0.128 | 0.904 ± 0.134 | 0.910 ± 0.151 | 0.897 ± 0.162 | 0.983 ± 0.060 |
| **`N = 100`** | `F1: given_syntax (Induce Lexicon)` | 0.992 ± 0.013 | 0.987 ± 0.020 | 0.990 ± 0.027 | 0.987 ± 0.023 | 1.000 ± 0.000 |
| **`N = 100`** | `F2: given_lexicon (Induce Syntax)` | **0.994 ± 0.013** | **0.991 ± 0.019** | **0.992 ± 0.017** | **0.993 ± 0.015** | **1.000 ± 0.000** |

---

## Part 5: MDL Prior Audit, Episodic Replay vs. Belief Revision, and Validity Summary

### 5.1 How $\alpha = 0.05$ Was Chosen & Explicit $\log_2$ Bit-Cost Prior Comparison (20 Seeds)

- **Origin of $\alpha = 0.05$:** In [`PREREGISTRATION.md`](./PREREGISTRATION.md), $\alpha = 0.05$ was chosen heuristically as a small complexity tie-breaker rather than derived from an information-theoretic code-length calculation.
- **Explicit Information-Theoretic Bit-Cost Prior (`compute_bit_cost_prior`):**
  Encoding each lexical entry $e \in L_h$ from $|\mathcal{C}| = 8$ syntactic categories and $|\mathcal{P}|$ semantic predicates requires:
  $$\text{Bits}(L_h) = \sum_{e \in L_h} \left( \log_2 |\mathcal{C}| + 2 \cdot \text{depth}(e) + \text{atoms}(e) \cdot \log_2 |\mathcal{P}| \right)$$
  Converting to natural log units ($\times \ln 2$) yields a penalty of `~6.8–9.6 nats` per lexical entry (roughly **`130x` larger** than $\alpha = 0.05$).
- **Empirical Comparison Across 20 Seeds:**

| Prior Formulation | Regime | `train` | `dev` | `test_novel_combination` | `test_novel_word_in_known_frame` | `test_longer` | `test_contradiction` |
|---|---|---|---|---|---|---|---|
| **Linear $\alpha = 0.05$ (`prior_mode="linear"`)** | `N = 1600` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **$\log_2$ Bit-Cost in Nats (`prior_mode="log2_bits"`)** | `N = 1600` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **$\log_2$ Bit-Cost in Raw Bits (`prior_mode="log2_raw_bits"`)** | `N = 1600` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **Linear $\alpha = 0.05$ (`prior_mode="linear"`)** | `N = 50` | 0.790 ± 0.141 | 0.784 ± 0.139 | 0.724 ± 0.157 | 0.777 ± 0.149 | 0.713 ± 0.178 | 0.766 ± 0.173 |
| **$\log_2$ Bit-Cost in Nats (`prior_mode="log2_bits"`)** | `N = 50` | 0.789 ± 0.140 | 0.783 ± 0.138 | 0.722 ± 0.156 | 0.775 ± 0.149 | 0.712 ± 0.177 | 0.764 ± 0.170 |

- **Why the Results Are Nearly Identical (`< 0.002` difference):** In Phase 1, every full-vocabulary hypothesis $h$ in the beam assigns exactly 1 entry to each seen surface word ($|L_h| = |V|$) and 1 semantic atom per entry ($\text{atoms}(e) = 1$). Because all competing permutations of noun meanings, modifier meanings, and verb meanings have the **exact same description length**, the MDL prior cancels out across full-lexicon bijections! The MDL prior will only do non-trivial work in **Phase 3**, where hypotheses have **variable lexicon sizes** (e.g., splitting a polysemous word into 2 entries vs. keeping 1 entry, or inducing multi-atom composite concepts).

---

### 5.2 Plain Statement on `correct()`: Full Episodic Replay Is Not Incremental Belief Revision

In [`NEXUSLearner.correct()`](./nexus/learner/nexus_learner.py):
1. Every `observe()` call appends the full scene and token tuple to an in-memory list `self._observations` and a JSONL log `self.experience_log_jsonl`.
2. When `correct(correction)` is called for `target_example_id`, the learner looks up `idx = self._obs_by_id[target_example_id]`, overwrites `self._observations[idx].label = correction.corrected_label`, and marks `self._beam_dirty = True`.
3. On the next query, `_build_beam_for_observations(self._observations)` **recomputes all word candidate scores and rebuilds the beam from scratch over the entire history of $N$ observations** (`O(N)` time and `O(N)` memory).
4. **Why This Is NOT True Incremental Belief Revision:**
   - A genuine incremental belief-revision system maintains compact sufficient statistics or belief states with provenance dependencies and updates only the affected beliefs in `O(1)` or `O(|affected|)` time without re-reading every historical scene from $t = 1 \dots N$.
   - Notice also that in `test_contradiction`, `NEXUSLearner` already achieved `1.000` *before* `correct()` (`test_contradiction_after_misleading = 1.000` at $N=1600$) because 1,600 clean training observations outweighed the 8 misleading observations (`8 vs ~300`), meaning `test_contradiction` at $N=1600$ did not even force a top-hypothesis flip until $N$ was reduced (`N <= 100`) or tested on a novel/low-count word!

---

### 5.3 Which Pre-Registered Phase 1 Conclusions Remain Valid After Parts 2 and 3?

| Pre-Registered Claim / Conclusion | Post-Audit Status | Qualification / Evidence |
|---|---|---|
| **1. Zero test-item leakage & strict Learner-protocol compliance** | **VALID** | Confirmed by `SpyLearnerWrapper` (`0` violations across `30,000` predictions) and static AST inspection. |
| **2. Compositional generalization to held-out `adj + noun` and `noun @ role` (`test_novel_combination`)** | **VALID (with Baseline D qualification)** | `NEXUS` reaches `1.000` at $N \ge 400$ (`0.904` at $N=100$), beating `Memorizer` (`0.500`) and `FromScratchNeural` (`0.485`). However, `PositionAwareCrossSituational` (Baseline D) already reaches `0.918` without grammar search; categorial grammar search specifically accounts for the final `+0.082` (`0.918 -> 1.000`) where linear nearest-noun distance is ambiguous. |
| **3. Length extrapolation to stacked `size + color + noun` NPs (`test_longer`)** | **VALID (within depth $\le 2$, distinct families)** | `NEXUS` reaches `1.000` vs Baseline D (`0.808`) and Baseline C (`0.704`). However, extrapolation fails (`0.0%` parse rate, `0.500` accuracy) on 3-adj same-family NPs and 2-clause relational chains. |
| **4. 2-shot novel noun acquisition (`test_novel_word_in_known_frame`)** | **VALID for 1-to-1 nouns; FAILS under synonymy** | Reaches `1.000` when each kind has 1 surface noun, but drops to `0.582 ± 0.116` when kinds have 2 synonymous surface forms. |
| **5. Supervision comes strictly from truth labels (`y_i`)** | **INVALIDATED BY CONTROL 2a** | Shuffling training truth labels (`2a_shuffled_labels`) leaves `NEXUS` at `0.657` (`test_novel_combination`) and `0.771` (`dev`) because near-miss negative scenes preserve scene-token co-occurrence and `_propose_word_candidates` weights positive co-occurrence asymmetrically. |
| **6. Clean retraction on `correct()` (`test_contradiction`)** | **PARTIALLY VALID (Batch Replay Only)** | Works via full `O(N)` episodic log replay rather than incremental belief revision, and at $N=1600$ the 8 misleading examples are already outnumbered by clean training examples before `correct()` is called. |

---

## Implications for Phase 3 (Concept Restructuring & Belief Revision)

1. **Fix the Unlabeled Co-occurrence Leak Before Phase 3:**
   - Both the world generator (which must include scenes where mentioned predicates are absent on `False` items or balanced scene-token pairs) and `_propose_word_candidates` (which must weight `obs.label == True` and `obs.label == False` symmetrically via exact log-likelihood ratios rather than positive-only rules) must be fixed so that **Control 2a (`shuffled_labels`) drops to `0.500 ± 0.02`** before any Phase 3 claims are made.
2. **Allow Multi-Sense Lexical Entries (`dict[str, list[LexicalEntry]]`) Regulated by the $\log_2$ Bit-Cost MDL Prior:**
   - Phase 1 breaks completely on homonymy/polysemy (`0.551` on `test_longer` under 3 homonym pairs) because `Hypothesis.entries` maps each word string to a single `LexicalEntry`. Phase 3 concept restructuring must allow a word $w$ to map to multiple senses `(w, sense_id) -> LexicalEntry`, where adding a second sense pays the explicit $\log_2$ bit-cost prior (`compute_bit_cost_prior`) unless justified by joint log-likelihood gain.
3. **Replace Full Episodic Replay with Sufficient-Statistic / Provenance-Indexed Belief Revision:**
   - Instead of re-running `_build_beam_for_observations` over all $N$ raw scenes on `correct()`, Phase 3 must maintain per-example provenance deltas in SQLite so retracting or correcting `example_id` updates the affected sufficient statistics incrementally in `O(1)` time.
4. **Extend Categorial Inventory Beyond Depth-2 Single Clauses:**
   - Supporting multi-clause relational chains (`NP1 rel1 NP2 rel2 NP3`, currently `0%` parse rate) requires adding conjunction/coordination (`(S\S)/S`) or relative-clause (`(N\N)/(S\NP)`) combinators.
