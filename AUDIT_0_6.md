# NEXUS-0 Phase 0.6: Harness & Statistical Audit Report (`AUDIT_0_6.md`)

**Stage**: STAGE A (Audit Patch — Gate A)  
**World Mode**: `randomized` (per-seed lexicon permutation, 6 NP orders, 3 clause orders, 2 Q-particle positions)  
**Pre-Registration Commit Tag**: `prereg-v1`  
**`PREREGISTRATION.md` SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`

---

## Executive Summary

An independent review of Phase 0.5 identified six methodological and statistical gaps in the evaluation harness before Phase 1 learner implementation could begin:
1. `SceneOnly` scored `0.458 ± 0.034` on `test_novel_combination` (~5 SE below chance) and `0.539 ± 0.021` on `test_longer` (~8 SE above chance) across 20 seeds, indicating systematic scene-feature leakage introduced by the Phase 0.5 falsification and hold-out sampling logic.
2. Split sample sizes `n` were omitted from summary tables, leaving standard deviations uninterpretable.
3. `Oracle` scoring `1.000` on `test_contradiction` did not validate the correction metric, because the uncorrupted `Oracle` never needed corrections.
4. `Oracle` and `WorldGenerator` shared a single model checker (`evaluate`), so a bug in `evaluate` would be invisible.
5. Word-order and construction identifiability from training examples was unverified across randomized grammars.
6. The uniqueness of the 2-example support stream in `test_novel_word_in_known_frame` was unverified, and calling the false example's label "zero feedback" was inaccurate because the binary truth label `False` is itself supervisory feedback.

All six findings have been diagnosed, resolved, and locked behind automated regression tests in `nexus/tests/test_stage_a_audit.py`.

---

## A1. Split Sample Sizes (`n`) & 50-Seed Pooled Shortcut Audit

### Split Sample Sizes (`world_default.json`)

| Split | `n` (total items per seed) | True / False Count | Construction Breakdown |
|---|---:|---:|---|
| `train` | **1,600** | 800 / 800 | Curriculum Stage 1 (`np_assert`, `np_ask`), Stage 2 (`trans_assert`, `trans_ask`) |
| `dev` | **400** | 200 / 200 | Strictly stratified: **100** per construction (50 True / 50 False each) |
| `test_novel_combination` | **400** | 200 / 200 | Held-out `(adj, kind)` pairs and `(kind, role)` pairs |
| `test_novel_word_in_known_frame` | **200** (+ 2 support) | 100 / 100 | 1 contrastive bare-NP support pair (2 examples), 200 queries in other frames |
| `test_longer` | **300** | 150 / 150 | Stacked adjectives per NP (`len(tokens) > max_train_length`) |
| `test_contradiction` | **200** (+ 40 support) | 100 / 100 | 8 mislabelled + 24 filler observations + 8 corrections; 200 diagnostic probes |

*(Note: `nexus/eval/report.py` now automatically outputs the `n` column in all generated markdown tables.)*

### 50-Seed Pooled Student-$t$ Audit (`SentenceOnly` & `SceneOnly`)

Both shortcut baselines were trained on `train` ($n=1,600$) and evaluated on all splits across **50 randomized seeds** (`seeds 0..49`, $df = 49$, $t_{0.975, 49} = 2.0096$).

| Split | `n` | `SentenceOnly` Mean ± Std | `SentenceOnly` 95% $t$-CI | Contains `0.50`? | `SceneOnly` Mean ± Std | `SceneOnly` 95% $t$-CI | Contains `0.50`? |
|---|---:|---|---|---|---|---|---|
| `train` | 1,600 | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** |
| `dev` | 400 | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** |
| `test_novel_combination` | 400 | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** |
| `test_novel_word_in_known_frame` | 200 | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** |
| `test_longer` | 300 | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** |
| `test_contradiction` | 200 | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** | `0.500 ± 0.000` | `[0.5000, 0.5000]` | **PASS** |

**Failures after fix**: **0 / 12**.

---

## A2. Root-Cause Diagnosis of Phase 0.5 Artifacts, Generator Changelog & Feature-Balance Audit

### Diagnosis of the Phase 0.5 `SceneOnly` Bias (`0.458` on `test_novel_combination`, `0.539` on `test_longer`)

Decomposing `SceneOnly`'s logistic regression weights and comparing $P(\text{feature}\mid\text{true})$ vs $P(\text{feature}\mid\text{false})$ on the Phase 0.5 generator uncovered three distinct mechanisms:

1. **Held-Out Selection Bias vs Uniform Falsification**:
   - In `train` and `test_longer`, `HeldOutPlan` rejected meanings containing held-out `(adj, kind)` and `(kind, role)` pairs (`_train_ok`), causing non-held-out kinds and adjectives to appear more frequently as witness objects in `True` scenes than held-out ones.
   - However, when `single_edits` or `matched_false_scene` constructed `False` scenes in Phase 0.5, it replaced witness attributes by sampling uniformly over `known_kinds` or `ontology.values(attr)`—injecting held-out kinds/adjectives into `False` scenes at a higher rate than `True` scenes!
   - Consequently, `SceneOnly` learned positive weights ($w > 0$) on non-held-out features and negative weights ($w < 0$) on held-out features. On `test_longer` (which also excludes held-out pairs), this gave `SceneOnly` a systematic advantage (`0.539 ± 0.021`). On `test_novel_combination` (which selects *only* held-out pairs), the sign flipped, driving `SceneOnly` systematically below chance (`0.458 ± 0.034`).
2. **Relation Graph Topology Distortion from the Phase 0.5 Endpoint Swap**:
   - In Phase 0.5, `single_edits` falsified transitive sentences by swapping endpoints `(name, s, o) -> (name, o, s)` on one relation while leaving a second relation in the scene untouched (creating 2-cycles `topo:2_cycle` in False scenes, $z = -5.87$), whereas `matched_false_scene` sampled 2 distinct endpoint pairs via `rng.sample(pairs, 2)`, which forbade parallel relations (`topo:2_parallel`, $z = +6.87$) and never collided with the witness edge (inflating `n_rels` in False scenes, $z = -3.82$).
3. **Novel-Kind and Contradiction-Color Asymmetry**:
   - In `test_novel_word_in_known_frame`, `novel_kind_present` was `1.000` in True scenes (`true_scene`) vs `0.467` in False scenes (`matched_false_scene`, $z = +47.78$).
   - In `test_contradiction`, `_diagnostic_example` placed the `target` color in True scenes and the `confusion` color in False scenes ($z = +4.46$).

### Generator & Split Builder Changelog (`nexus/world/generator.py`, `nexus/eval/splits.py`)

1. **Feature- and Topology-Preserving Binding Edits (`WorldGenerator.binding_edits`)**:
   - Multi-predicate meanings ($\ge 1$ adjective or transitive relation) are falsified by swapping full object attribute bundles `attrs[i] <-> attrs[j]` or single attributes `attrs[i][a] <-> attrs[j][a]` between objects in `true_scene` while keeping `scene.relations` untouched. This strictly preserves `n_objs`, every `kind`/`color`/`size` count, `n_rels`, and the exact directed relation graph topology (`topo:*`).
2. **Symmetric $2\times 2$ Crossover Quads for Value-Replacement Edits (`WorldGenerator.value_swap_quad_scenes`)**:
   - Whenever falsification requires changing a value in the scene (bare-NP sentences `"a K1 exists"`, or direct lexical value-replacement contrasts in `train`), the generator constructs a symmetric $2\times 2$ crossover quad $((m_1, S_1, \text{True}), (m_1, S_2, \text{False}), (m_2, S_2, \text{True}), (m_2, S_1, \text{False}))$ where $m_2$ substitutes $v_1 \to v_2$ and $S_2$ substitutes $v_1 \to v_2$ on the witness object of $S_1$. Across the quad, the multiset of True scenes $\{S_1, S_2\}$ and False scenes $\{S_2, S_1\}$ is identical, and the multiset of True sentences $\{m_1, m_2\}$ and False sentences $\{m_1, m_2\}$ is identical.
3. **Symmetric Diagnostic Pairs in `test_contradiction` (`SplitBuilder._diagnostic_pair`)**:
   - Both True and False diagnostic scenes contain both the `target` color (on the witness or distractor) and the `confusion` color (on the distractor or witness), swapped between the two objects to flip truth under both the true reading and the confusion reading while keeping scene and sentence features 100% identical.

### Feature-Level Label-Balance Audit Table (20 Seeds Pooled, 1,000 Permutations + Holm & BH)

Across all 6 splits (`train`, `dev`, `test_novel_combination`, `test_novel_word_in_known_frame`, `test_longer`, `test_contradiction`) and all 63 audited features:

| Feature Family | Audited Features | `train` Max $|\Delta|$ | `dev` Max $|\Delta|$ | `test_novel_combination` Max $|\Delta|$ | `test_novel_word` Max $|\Delta|$ | `test_longer` Max $|\Delta|$ | `test_contradiction` Max $|\Delta|$ | Min $p_{\text{Holm}}$ / $q_{\text{BH}}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Object count | `n_objs`, `n_objs==2..4` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Relation count | `n_rels`, `count:rel:*`, `present:rel:*` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Relation topology | `topo:0`, `topo:1`, `topo:2_parallel`, `topo:2_cycle`, `topo:2_chain`, `topo:2_disjoint`, `topo:2_shared_src`, `topo:2_shared_dst` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Kind counts/presence | `count:kind:*`, `present:kind:*` (6 kinds) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Color counts/presence | `count:color:*`, `present:color:*` (4 colors) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Size counts/presence | `count:size:*`, `present:size:*` (3 sizes) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Novel kind presence | `novel_kind_present` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |
| Token presence & length | `sentence_length`, `token_pred:*` (16 tokens) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.000` / `1.000` |

### Rerun 20-Seed Benchmark Table (`seeds 0..19`, `world_mode=randomized`)

| Model | `train` ($n=1600$) | `dev` ($n=400$) | `test_novel_combination` ($n=400$) | `test_novel_word_in_known_frame` ($n=200$) | `test_longer` ($n=300$) | `test_contradiction` ($n=200$) |
|---|---|---|---|---|---|---|
| **Oracle** | `1.000 ± 0.000` | `1.000 ± 0.000` | `1.000 ± 0.000` | `1.000 ± 0.000` | `1.000 ± 0.000` | `1.000 ± 0.000` |
| **StubbornOracle (A4)** | `0.925 ± 0.008` | `0.931 ± 0.010` | `0.907 ± 0.037` | `0.908 ± 0.024` | `0.790 ± 0.019` | **`0.000 ± 0.000`** |
| **CorrectedOracle (A4)** | `0.925 ± 0.008` | `0.931 ± 0.010` | `0.907 ± 0.037` | `0.908 ± 0.024` | `0.790 ± 0.019` | **`1.000 ± 0.000`** |
| **SentenceOnly** | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` |
| **SceneOnly** | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` |
| **Memorizer** | `1.000 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` |

---

## A3. `test_novel_word_in_known_frame` Candidate Enumeration & Ceiling

1. **Novel-Kind Scene Balance**:
   - Across all 20 seeds (`n = 200` test items per seed: 100 True, 100 False), every False scene is constructed via `binding_edits` of a True scene containing the novel kind (where a modifier or relational argument binding is swapped to falsify the sentence while keeping the novel-kind object in the scene).
   - Consequently:
     $$P(\text{novel\_kind\_present} \mid \text{True}) = 1.000, \qquad P(\text{novel\_kind\_present} \mid \text{False}) = 1.000 \qquad (\Delta = 0.000)$$
2. **Exact Enumeration of Support-Consistent Meanings in the Phase 1 Hypothesis Space**:
   - The Phase 1 lexical hypothesis space consists of 15 predicates: 6 kinds (`kind:*`), 4 colors (`color:*`), 3 sizes (`size:*`), and 2 relations (`rel:*`).
   - The support stream provides 1 contrastive bare-NP pair (2 examples: Example 1 with label `True` in scene $S_{\text{true}}$, and Example 2 with the identical token sequence and label `False` in near-miss scene $S_{\text{false}}$).
   - Because $S_{\text{false}}$ is constructed from $S_{\text{true}}$ by replacing only the witness object's `kind` attribute (`novel_kind -> k_alt`, where `k_alt` is already present on a distractor or distinct from `novel_kind`) while leaving all object `color`s, `size`s, and `relations` unchanged, **even without using mutual exclusivity from `train`**, the set of predicates in the 15-predicate hypothesis space that hold in $S_{\text{true}}$ and fail in $S_{\text{false}}$ has size:
     $$|\mathcal{C}_{\text{support}}| = 1 \quad (\text{uniquely } \{\texttt{kind:}\langle\text{novel\_kind}\rangle\}) \text{ on } 20 / 20 \text{ seeds.}$$
   - **Ideal-Observer Accuracy Ceiling**: **`1.000`** across all 20 seeds.

---

## A4. Correction Metric Validation (`StubbornOracle` vs `CorrectedOracle`)

To prove that `test_contradiction` genuinely measures belief revision in response to `Correction` events:
- **`StubbornOracle`**: Initialized with the ground-truth grammar except that the surface word for `contradiction_target` (e.g. `color:red`) is mapped to `contradiction_confusion` (e.g. `color:blue`), and `correct()` is a no-op.
  - Accuracy at `before_corrections`: **`0.000 ± 0.000`**
  - Accuracy at `after_corrections` (final split accuracy): **`0.000 ± 0.000`** (strictly $< 0.50$).
- **`CorrectedOracle`**: Initialized with the exact same corrupted lexical entry, records observations in `observe()`, updates labels upon receiving `Correction(target_example_id, corrected_label)` in `correct()`, and re-solves the lexical mapping for the target word over the ontology family.
  - Accuracy at `before_corrections`: **`0.000 ± 0.000`**
  - Accuracy at `after_corrections` (final split accuracy): **`1.000 ± 0.000`**.

---

## A5. Independent Second Model Checker (`naive_evaluate`)

A second, independent model checker `naive_evaluate` was implemented in `nexus/world/logic.py` using `itertools.permutations` over object assignments and inspecting raw `SceneObject.kind`, `SceneObject.color`, `SceneObject.size`, and `Scene.relations` fields directly without sharing `atom_holds`, `scene.holds`, or `obj.value` with `evaluate`.
- **Golden Test Suite**: Verified on **20 hand-crafted golden cases** (`test_a5_naive_model_checker_20_golden_cases`) covering unary kinds, colors, sizes, 2- and 3-atom conjunctions, cross-object binding traps, directed `chases`, symmetric `next_to`, relation-type mismatches, and relation-free scenes.
- **Differential Property Test**: Verified 100% agreement (`evaluate(lf, scene, sym) == naive_evaluate(lf, scene, sym)`) across **1,000 random `(scene, LF)` pairs** (`test_a5_differential_1000_random_scenes_and_lfs`).

---

## A6. Word-Order & Construction Identifiability Report

Across 20 randomized seeds (`n_train = 1,600` per seed), we counted training examples that directly distinguish the true grammatical analysis from alternative analyses:

| Parameter / Construction | Distinguishing Mechanism | Min Count (across 20 seeds) | Mean ± Std (across 20 seeds) | Max Count | Flagged ($< 5$)? |
|---|---|---:|---|---:|---|
| `np_assert` | True/False minimal pairs in Stage 1 | 348 | `417.1 ± 37.9` | 480 | **NO (0/20)** |
| `np_ask` | True/False minimal pairs with `Q` in Stage 1 | 352 | `418.2 ± 30.8` | 478 | **NO (0/20)** |
| `trans_assert` | True/False minimal pairs in Stage 2 | 298 | `380.0 ± 29.8` | 438 | **NO (0/20)** |
| `trans_ask` | True/False minimal pairs with `Q` in Stage 2 | 294 | `384.7 ± 31.6` | 428 | **NO (0/20)** |
| Clause role order (`NP_subj` vs `NP_obj`) | Asymmetric `rel:chases(x0, x1)` where `chases(x1, x0)` has opposite truth value in scene | **183** | `207.8 ± 14.5` | 236 | **NO (0/20)** |
| Within-NP modifier attachment (`np_order`) | Transitive sentences with an adjective where attaching the adjective to the other NP flips truth value in scene | **253** | `293.0 ± 18.7` | 328 | **NO (0/20)** |
| Question particle position (`initial` vs `final`) | `np_ask` + `trans_ask` utterances with boundary `Q` token | **740** | `797.9 ± 39.2` | 874 | **NO (0/20)** |

*Important Note on Within-NP Relative Adjective Order (`size` vs `color`)*:
In `train`, `max_adjectives_train = 1`, so **0 training NPs carry both a `size` and a `color` modifier simultaneously** (`n_train_stacked_nps = 0`). Thus, training examples identify **whether adjectives precede or follow the head noun** (and which side of a noun an adjective attaches to in transitive clauses), which is sufficient to parse stacked adjectives in `test_longer` because conjunctive modification (`size(x) & color(x)`) is semantically commutative once attached to the correct head noun `x`. We explicitly document in `README.md` that relative ordering *between* `size` and `color` inside a single NP is not identifiable from `train` and is never claimed as learned.

---

## A7. Small Fixes & Pre-Registration Integrity

1. **Even Per-Construction Dev Count**: `world_default.json` has `n_dev_per_construction = 100` (`small_config` has `30`), and `nexus/world/config.py` now raises `ConfigError` if `n_dev_per_construction % 2 != 0`. `README.md` Note 8.1(a) was updated to accurately explain the Phase 0 `0.493` dev accuracy (caused by sentence-level Laplace backoff when a training sentence recurred in dev with an opposite-truth scene before paired dev balancing) and why `Memorizer` now scores `0.500 ± 0.000`.
2. **Training Stacked Modifiers Check**: Verified `n_train_stacked_nps == 0` and documented in `README.md`.
3. **README Wording Fixes**: Clarified that in `test_novel_word_in_known_frame`, the second support example provides supervisory truth-value feedback (`label=False`) rather than "zero feedback", and clarified "modifier stacking" in `test_longer`.
4. **Pre-Registration Lock**: Tagged commit `68a452e` as `prereg-v1` and added `test_a7_preregistration_hash_and_small_fixes` verifying `SHA-256(PREREGISTRATION.md) == ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`.
5. **Feasibility Check**: Because the A3 ideal-observer ceiling on `test_novel_word_in_known_frame` is `1.000` (and A8 ideal observer ceiling is `1.000`), all pre-registered pass criteria (`>= 0.80` on `test_novel_combination`, `>= 0.70` on `test_novel_word_in_known_frame`) are feasible. No amendment proposal is needed.

---

## A8. Exhaustive Bayesian Ideal Observer on a Reduced World

In `nexus/eval/audit.py` (`ExhaustiveIdealObserver` / `run_reduced_world_ideal_observer`), we constructed a reduced world small enough for exact, zero-approximation enumeration:
- **Ontology**: 2 known kinds (`cube`, `ball`) + 1 held-out novel kind (`cone`), 2 colors (`red`, `blue`), 1 directed relation (`chases`), fixed word order (`[noun, color]`, `SOV`, `Q`-final).
- **Hypothesis Space**: All $6! = 720$ bijections between the 6 content words (`dax`, `wug`, `zib`, `mip`, `sorn`, `zorp`) and the 6 predicates (`kind:cube`, `kind:ball`, `kind:cone`, `color:red`, `color:blue`, `rel:chases`).
- **Exact Bayesian Posterior**:
  $$\log P(H \mid D) = \log P(H) + \sum_{(s_i, e_i, y_i) \in D} \log \big((1 - \varepsilon)\mathbf{1}[H(s_i, e_i) = y_i] + \varepsilon\mathbf{1}[H(s_i, e_i) \ne y_i]\big), \quad \varepsilon = 0.02$$

### Exhaustive Ideal Observer Results (20 Seeds, Mean ± Std)

| Metric / Split | Result (20 Seeds) |
|---|---|
| Hypotheses Enumerated | `720` (exact enumeration) |
| Top Posterior Hypothesis == Ground Truth Lexicon | **20 / 20 seeds (`100%`)** |
| `train` Accuracy | **`1.000 ± 0.000`** |
| `dev` Accuracy | **`1.000 ± 0.000`** |
| `test_novel_combination` Accuracy | **`1.000 ± 0.000`** |
| `test_novel_word` (`after_support`) Accuracy | **`1.000 ± 0.000`** |
