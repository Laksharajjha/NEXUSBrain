# NEXUS-0: Phase 1 Pre-Registered Evaluation Report (`PHASE1_REPORT.md`)

**Stage**: Stage C (`gate-C`)  
**Pre-Registration SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b`  
**Pre-Run Tag**: `pre-run` (`8f7102b`)  
**Reruns**: `0` (see [RERUNS.md](file:///Users/laksharajjha/Downloads/NEXUS/RERUNS.md))  
**Seeds**: `0..19` ($N = 20$ seeds, `world_mode = randomized`)  
**Summary Artifact**: [phase1_eval_summary.json](file:///Users/laksharajjha/Downloads/NEXUS/nexus/experiments/results/phase1_eval_summary.json)

---

## 1. Executive Summary & Frozen Pass Criteria

All six learners were evaluated across 20 randomized seeds (`seeds 0..19`), where each seed independently permutes the 16-word lexicon, within-NP modifier order (`6` permutations of `(noun, color, size)`), transitive clause order (`SOV`, `SVO`, `VSO`), and question particle placement (`initial` vs. `final`).

| # | Frozen Criterion (`PREREGISTRATION.md` §5) | Target Threshold | Measured Value (20 Seeds) | Status |
| :-: | :--- | :---: | :---: | :---: |
| **1** | **Novel Combination Generalization** (`test_novel_combination`) | $\ge 0.800$ | **$1.000 \pm 0.000$** | **PASS** |
| **2** | **Few-Shot Novel Word Acquisition** (`test_novel_word_in_known_frame`) | $\ge 0.700$ | **$1.000 \pm 0.000$** | **PASS** |
| **3** | **Superiority vs. $\max(A, B)$** (mean diff $\ge 0.15$, 95% bootstrap CI excludes $0$) | $\Delta \ge +0.150$ | **$+0.500$, 95% CI $[+0.500, +0.500]$** | **PASS** |
| **4** | **Shortcut Immunity** (`SentenceOnly` and `SceneOnly` within chance CI) | $\in [0.370, 0.630]$ | **$0.500 \pm 0.000$** on all splits | **PASS** |
| **5** | **Catastrophic Forgetting Resistance** (`retention.np_assert.final`) | $\ge 0.900$ | **$1.000 \pm 0.000$** ($\Delta = 0.000$) | **PASS** |

---

## 2. Main 20-Seed Benchmark Table (`world_mode=randomized`)

Values report `mean ± std` across `seeds 0..19`. Split sample sizes `n` per seed are explicitly shown.

### 2.1 Truth-Judgment Accuracy (`mean ± std`)

| Split | `n` / seed | Baseline A (`Memorizer`) | Baseline B (`CrossSituational`) | Baseline C (`NoRevision`) | `SentenceOnly` | `SceneOnly` | **NEXUS Phase 1 (`NEXUSLearner`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | `1600` | `1.000 ± 0.000` | `0.724 ± 0.012` | `0.505 ± 0.009` | `0.500 ± 0.000` | `0.500 ± 0.000` | **`1.000 ± 0.000`** |
| `dev` | `400` | `0.500 ± 0.000` | `0.579 ± 0.019` | `0.507 ± 0.010` | `0.500 ± 0.000` | `0.500 ± 0.000` | **`1.000 ± 0.000`** |
| `test_novel_combination` | `400` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.499 ± 0.004` | `0.500 ± 0.000` | `0.500 ± 0.000` | **`1.000 ± 0.000`** |
| `test_novel_word_in_known_frame` (before support) | `200` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` |
| `test_novel_word_in_known_frame` (after 2-ex support) | `200` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.007` | `0.500 ± 0.000` | `0.500 ± 0.000` | **`1.000 ± 0.000`** |
| `test_longer` | `300` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.500 ± 0.000` | **`1.000 ± 0.000`** |
| `test_contradiction` (after corrections) | `200` | `0.500 ± 0.000` | `0.500 ± 0.000` | `0.499 ± 0.002` | `0.500 ± 0.000` | `0.500 ± 0.000` | **`1.000 ± 0.000`** |
| **Model Size (entries)** | — | `2022.2 ± 19.0` | `15.0 ± 0.0` | `15.0 ± 0.0` | `197.0 ± 0.0` | `17.0 ± 0.0` | **`15.0 ± 0.0`** (`16.0` after novel word) |

### 2.2 Calibration (ECE), Brier Score, Exact Logical Form (LF) Match, and Learning Efficiency for `NEXUSLearner`

| Split | `n` / seed | Accuracy | Exact LF Match | ECE (10 bins) | Brier Score | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | `1600` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.020 ± 0.000` | `0.0004 ± 0.0000` | `0.980 ± 0.000` |
| `dev` | `400` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.020 ± 0.000` | `0.0004 ± 0.0000` | `0.980 ± 0.000` |
| `test_novel_combination` | `400` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.020 ± 0.000` | `0.0004 ± 0.0000` | `0.980 ± 0.000` |
| `test_novel_word_in_known_frame` | `200` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.060 ± 0.015` | `0.0038 ± 0.0019` | `0.940 ± 0.015` |
| `test_longer` | `300` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.020 ± 0.000` | `0.0004 ± 0.0000` | `0.980 ± 0.000` |
| `test_contradiction` | `200` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.020 ± 0.000` | `0.0004 ± 0.0000` | `0.980 ± 0.000` |

- **Examples-to-Criterion ($\ge 0.90$ dev accuracy)**:
  - `np_assert`: `56.4 ± 18.5` construction examples (`115.0 ± 36.6` total examples, $20/20$ seeds reached).
  - `np_ask`: `60.9 ± 22.6` construction examples (`120.0 ± 41.0` total examples, $20/20$ seeds reached).
  - `trans_assert`: `38.9 ± 19.2` construction examples (`905.0 ± 60.5` total examples, $20/20$ seeds reached).
  - `trans_ask`: `39.0 ± 19.1` construction examples (`910.0 ± 55.3` total examples, $20/20$ seeds reached).
- **Retention (Stage End vs. End of Training)**:
  - `np_assert`: `1.000 -> 1.000` ($\Delta = 0.000 \pm 0.000$).
  - `np_ask`: `1.000 -> 1.000` ($\Delta = 0.000 \pm 0.000$).
  - `trans_assert`: `1.000 -> 1.000` ($\Delta = 0.000 \pm 0.000$).
  - `trans_ask`: `1.000 -> 1.000` ($\Delta = 0.000 \pm 0.000$).

---

## 3. Paired Bootstrap 95% Confidence Intervals (10,000 Resamples)

For each seed $s \in \{0, \dots, 19\}$, we compute $\Delta_{\text{AB}}(s) = \text{Acc}_s(\text{NEXUS}) - \max(\text{Acc}_s(\text{A}), \text{Acc}_s(\text{B}))$ and $\Delta_{\text{C}}(s) = \text{Acc}_s(\text{NEXUS}) - \text{Acc}_s(\text{C})$ and resample 10,000 times with replacement (`seed = 20261024`).

| Split | Comparison | Mean Difference | Sample Std | 95% Bootstrap CI | Excludes Zero? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dev` | $\text{NEXUS} - \max(A, B)$ | `+0.4214` | `0.0192` | `[+0.4133, +0.4295]` | **YES** |
| `dev` | $\text{NEXUS} - C$ | `+0.4933` | `0.0105` | `[+0.4889, +0.4976]` | **YES** |
| **`test_novel_combination`** | **$\text{NEXUS} - \max(A, B)$** | **`+0.5000`** | **`0.0000`** | **`[+0.5000, +0.5000]`** | **YES** |
| `test_novel_combination` | $\text{NEXUS} - C$ | `+0.5006` | `0.0039` | `[+0.4992, +0.5025]` | **YES** |
| **`test_novel_word_in_known_frame`** | **$\text{NEXUS} - \max(A, B)$** | **`+0.5000`** | **`0.0000`** | **`[+0.5000, +0.5000]`** | **YES** |
| `test_novel_word_in_known_frame` | $\text{NEXUS} - C$ | `+0.4995` | `0.0067` | `[+0.4965, +0.5023]` | **YES** |
| `test_longer` | $\text{NEXUS} - \max(A, B)$ | `+0.5000` | `0.0000` | `[+0.5000, +0.5000]` | **YES** |
| `test_longer` | $\text{NEXUS} - C$ | `+0.5000` | `0.0000` | `[+0.5000, +0.5000]` | **YES** |
| `test_contradiction` | $\text{NEXUS} - \max(A, B)$ | `+0.5000` | `0.0000` | `[+0.5000, +0.5000]` | **YES** |
| `test_contradiction` | $\text{NEXUS} - C$ | `+0.5005` | `0.0022` | `[+0.5000, +0.5015]` | **YES** |

**Why Baseline B (`CrossSituationalLearner`) scores `0.579` on `dev` and `0.500` on held-out binding splits**:
- Baseline B correctly induces the 15 word-to-predicate associations via IBM-1 / cross-situational alignment and thus succeeds on **bare single-NP** items in `train` (`0.724`) and `dev` (`np_assert: 0.698`, `np_ask: 0.617`), where the False scene replaces the witness kind with another kind.
- However, on modified NPs (`adj + noun`) and transitive clauses (`subj verb obj`), False scenes are constructed by **binding edits** (swapping which object has color $c$, or swapping the subject and object of `chases`), so the unstructured bag of predicates $\{P(w) \mid w \in \text{tokens}\}$ is present in **both** the True scene and the False scene. Without categorial slash composition and variable binding, Baseline B predicts True on both members of every binding-contrast pair (`0.500`).

---

## 4. Top-Hypothesis Induced Lexicons vs. Ground Truth (Seeds 0, 1, 2)

### Seed 0 (`np_order = ['size', 'color', 'noun']`, `trans = ['VERB', 'NP_subj', 'NP_obj']` (VSO), `ask = ['NP', 'Q']`)
Posterior weight of rank-1 hypothesis: `1.0000` (`prior = -2.15`, `likelihood = -32.32`).

| Surface Word | Induced Category | Induced Semantic Predicate | Ground-Truth Concept | Match? | Sample Provenance (Supporting Example IDs) |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `vek` | `N/N` | `color:yellow` | `color:yellow` | $\checkmark$ | `train-01126`, `train-01375`, `train-01226` |
| `zib` | `N` | `kind:star` | `kind:star` | $\checkmark$ | `train-01126`, `train-00175`, `train-00784` |
| `nud` | `N/N` | `size:medium` | `size:medium` | $\checkmark$ | `train-00221`, `train-01538`, `train-00938` |
| `dax` | `N` | `kind:cone` | `kind:cone` | $\checkmark$ | `train-01058`, `train-00536`, `train-01226` |
| `glorp` | `N/N` | `size:large` | `size:large` | $\checkmark$ | `train-00895`, `train-00847`, `train-01577` |
| `fep` | `N` | `kind:cup` | `kind:cup` | $\checkmark$ | `train-01092`, `train-01047`, `train-00000` |
| `wug` | `N` | `kind:cube` | `kind:cube` | $\checkmark$ | `train-01318`, `train-00735`, `train-00584` |
| `ko` | `S\S` | `Q` | `Q` | $\checkmark$ | `train-01318`, `train-01092`, `train-00847` |
| `blick` | `N` | `kind:ball` | `kind:ball` | $\checkmark$ | `train-01153`, `train-01207`, `train-01290` |
| `mip` | `N/N` | `color:blue` | `color:blue` | $\checkmark$ | `train-01290`, `train-01441`, `train-00918` |
| `sorn` | `N/N` | `color:red` | `color:red` | $\checkmark$ | `train-00584`, `train-01285`, `train-01353` |
| `tal` | `N/N` | `color:green` | `color:green` | $\checkmark$ | `train-00068`, `train-00129`, `train-00553` |
| `pim` | `N/N` | `size:small` | `size:small` | $\checkmark$ | `train-00536`, `train-00996`, `train-00247` |
| `lum` | `(S/NP)/NP` | `rel:next_to:sym(x0,x1)` | `rel:next_to` | $\checkmark$ | `train-00762`, `train-00274`, `train-00183` |
| `zorp` | `(S/NP)/NP` | `rel:chases(x0,x1)` | `rel:chases` | $\checkmark$ | `train-00649`, `train-00565`, `train-01484` |

### Seed 1 (`np_order = ['size', 'color', 'noun']`, `trans = ['VERB', 'NP_subj', 'NP_obj']` (VSO), `ask = ['Q', 'NP']`)
Posterior weight of rank-1 hypothesis: `1.0000` (`prior = -2.15`, `likelihood = -32.32`).

| Surface Word | Induced Category | Induced Semantic Predicate | Ground-Truth Concept | Match? | Sample Provenance (Supporting Example IDs) |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `nud` | `N/N` | `size:small` | `size:small` | $\checkmark$ | `train-00448`, `train-01185`, `train-00331` |
| `dax` | `N` | `kind:cone` | `kind:cone` | $\checkmark$ | `train-00448`, `train-01492`, `train-01504` |
| `wug` | `N` | `kind:cube` | `kind:cube` | $\checkmark$ | `train-00581`, `train-00368`, `train-00052` |
| `ko` | `S/S` | `Q` | `Q` | $\checkmark$ | `train-00815`, `train-00651`, `train-00575` |
| `sorn` | `N/N` | `color:yellow` | `color:yellow` | $\checkmark$ | `train-00815`, `train-00472`, `train-01242` |
| `zib` | `N` | `kind:cup` | `kind:cup` | $\checkmark$ | `train-00472`, `train-00048`, `train-00348` |
| `tal` | `N/N` | `color:blue` | `color:blue` | $\checkmark$ | `train-00651`, `train-01492`, `train-00162` |
| `toma` | `N` | `kind:ball` | `kind:ball` | $\checkmark$ | `train-00575`, `train-00534`, `train-00806` |
| `mip` | `N/N` | `color:green` | `color:green` | $\checkmark$ | `train-00368`, `train-00501`, `train-00176` |
| `glorp` | `N/N` | `size:large` | `size:large` | $\checkmark$ | `train-00052`, `train-01401`, `train-00454` |
| `pim` | `N/N` | `size:medium` | `size:medium` | $\checkmark$ | `train-01513`, `train-00276`, `train-01489` |
| `blick` | `N` | `kind:star` | `kind:star` | $\checkmark$ | `train-01076`, `train-01066`, `train-01531` |
| `vek` | `N/N` | `color:red` | `color:red` | $\checkmark$ | `train-00534`, `train-00992`, `train-01370` |
| `zorp` | `(S/NP)/NP` | `rel:chases(x0,x1)` | `rel:chases` | $\checkmark$ | `train-00524`, `train-00311`, `train-00830` |
| `lum` | `(S/NP)/NP` | `rel:next_to:sym(x0,x1)` | `rel:next_to` | $\checkmark$ | `train-00679`, `train-01577`, `train-01252` |

### Seed 2 (`np_order = ['noun', 'color', 'size']`, `trans = ['NP_subj', 'NP_obj', 'VERB']` (SOV), `ask = ['NP', 'Q']`)
Posterior weight of rank-1 hypothesis: `1.0000` (`prior = -2.15`, `likelihood = -32.32`).

| Surface Word | Induced Category | Induced Semantic Predicate | Ground-Truth Concept | Match? | Sample Provenance (Supporting Example IDs) |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `blick` | `N` | `kind:cup` | `kind:cup` | $\checkmark$ | `train-00662`, `train-00411`, `train-00154` |
| `dax` | `N` | `kind:ball` | `kind:ball` | $\checkmark$ | `train-00172`, `train-00413`, `train-00151` |
| `pim` | `N\N` | `size:small` | `size:small` | $\checkmark$ | `train-00627`, `train-01481`, `train-00626` |
| `ko` | `S\S` | `Q` | `Q` | $\checkmark$ | `train-00627`, `train-00589`, `train-01410` |
| `wug` | `N` | `kind:cube` | `kind:cube` | $\checkmark$ | `train-00120`, `train-01063`, `train-00072` |
| `glorp` | `N\N` | `size:large` | `size:large` | $\checkmark$ | `train-01563`, `train-00859`, `train-00875` |
| `tal` | `N\N` | `color:red` | `color:red` | $\checkmark$ | `train-00589`, `train-00172`, `train-00171` |
| `fep` | `N` | `kind:star` | `kind:star` | $\checkmark$ | `train-01410`, `train-00364`, `train-00254` |
| `mip` | `N\N` | `color:yellow` | `color:yellow` | $\checkmark$ | `train-01410`, `train-01449`, `train-01506` |
| `nud` | `N\N` | `size:medium` | `size:medium` | $\checkmark$ | `train-00925`, `train-00645`, `train-01250` |
| `toma` | `N` | `kind:ring` | `kind:ring` | $\checkmark$ | `train-00499`, `train-00280`, `train-01402` |
| `sorn` | `N\N` | `color:green` | `color:green` | $\checkmark$ | `train-00364`, `train-01402`, `train-01008` |
| `vek` | `N\N` | `color:blue` | `color:blue` | $\checkmark$ | `train-00070`, `train-00558`, `train-00428` |
| `lum` | `(S\NP)\NP` | `rel:chases(x0,x1)` | `rel:chases` | $\checkmark$ | `train-00495`, `train-00896`, `train-01076` |
| `zorp` | `(S\NP)\NP` | `rel:next_to:sym(x0,x1)` | `rel:next_to` | $\checkmark$ | `train-01256`, `train-00391`, `train-01439` |

---

## 5. Categorization of 10 Highest-Loss Held-Out Predictions per Split via `explain()`

Across all 20 seeds at `N = 1600`, `NEXUSLearner` makes **0 binary classification errors** on the held-out splits (`accuracy = 1.000`, `lf_accuracy = 1.000`). Ranking held-out items by Brier loss $(p_{\text{true}} - y)^2$ via `learner.explain(tokens, scene)` surfaces two distinct residual-uncertainty categories:

1. **Bayesian Model Averaging Uncertainty on 2-Shot Novel Words (`test_novel_word_in_known_frame`, Brier loss `0.008548`)**:
   - After only 2 support observations (`1 True`, `1 False`) of the held-out noun (e.g. `toma -> kind:ring` on Seed 0), the true lexical entry `(toma, N, kind:ring)` achieves posterior weight $w_0 = 0.9245$, while competing hypotheses in the beam (which assign `toma` to other kinds present on distractor objects in the True support scene, penalized by the mutual-exclusivity tie-breaker) retain $0.0755$ posterior weight.
   - Consequently, Bayesian model averaging predicts $p_{\text{true}} = 0.9075$ on True queries and $0.0925$ on False queries (10 worst items on `test_novel_word_in_known_frame`: `test_novel_word_in_known_frame-00006` through `00024`, all classified in category `bma_support_uncertainty`).
2. **Observation Noise Floor $\varepsilon = 0.02$ (`dev`, `test_novel_combination`, `test_longer`, `test_contradiction`, Brier loss `0.000400`)**:
   - Because the likelihood model incorporates label noise parameter $\varepsilon = 0.02$, a hypothesis with $1.000$ posterior weight outputs $p_{\text{true}} = 1 - \varepsilon = 0.9800$ when `witness` is found (`true_positive_noise_floor`) and $p_{\text{true}} = \varepsilon = 0.0200$ when `find_witness_or_failure` reports the exact falsifying atom (`true_negative_noise_floor`), e.g.:
     - `test_novel_combination-00001` (`['glorp', 'dax', 'ko']`, `ASK(EXISTS x0 . kind:cone(x0) & size:large(x0))`): `Closest assignment (x0->obj_0) failed on: size:large(obj_0) failed (obj_0.size=medium)`.
     - `test_longer-00001` (`['zorp', 'glorp', 'sorn', 'blick', 'vek', 'zib', 'ko']`): `Closest assignment (x0->obj_2, x1->obj_0) failed on: size:large(obj_2) failed (obj_2.size=small)`.
     - `test_contradiction-00001` (`['zorp', 'sorn', 'zib', 'tal', 'fep']`): `Closest assignment (x0->obj_3, x1->obj_1) failed on: color:red(obj_3) failed (obj_3.color=yellow)`.
     (In Stage D3/D5, we evaluate low-resource regimes $N \in \{50, 100, 200\}$ and high label-noise regimes $\{5\%, 10\%, 20\%\}$ where genuine lexical, syntactic, and role-binding errors occur; see [LIMITATIONS.md](file:///Users/laksharajjha/Downloads/NEXUS/LIMITATIONS.md).)
