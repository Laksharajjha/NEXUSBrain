# NEXUS-0 Phase 1: Exploratory Analyses, Failure Taxonomy & Limitations (`LIMITATIONS.md`)

**Label**: `EXPLORATORY` (All analyses in this document were executed in Stage D after the frozen Stage C evaluation tagged `gate-C`).  
**Artifact**: [stage_d_exploratory.json](file:///Users/laksharajjha/Downloads/NEXUS/nexus/experiments/results/stage_d_exploratory.json)  
**Runner**: [stage_d_exploratory.py](file:///Users/laksharajjha/Downloads/NEXUS/nexus/experiments/stage_d_exploratory.py)

---

## 1. D1: Beam Width Sensitivity ($K \in \{1, 5, 25, 100, 200, 500\}$, `EXPLORATORY`)

We evaluated `NEXUSLearner` across beam widths $K \in \{1, 5, 25, 100, 200, 500\}$ on `seeds 0..4` (`world_mode = randomized`) in both a low-resource regime ($N = 120$ stratified training examples) and the full training regime ($N = 1600$):

| Beam Width $K$ | `test_novel_combination` ($N=120$) | `test_novel_combination` ($N=1600$) | `test_novel_word_in_known_frame` Acc ($N=1600$) | `test_novel_word` Brier Score ($N=1600$) |
| :---: | :---: | :---: | :---: | :---: |
| `1` | `0.893 ± 0.104` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.0004` |
| `5` | `0.888 ± 0.108` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.0043` |
| `25` | `0.905 ± 0.102` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.0043` |
| `100` | `0.868 ± 0.126` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.0043` |
| `200` (Default) | `0.863 ± 0.128` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.0043` |
| `500` | `0.868 ± 0.132` | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.0043` |

- **Finding**: Notice the contrast between `NoRevisionLearner` ($K=1$ greedy commitment on first encounter, which scores `0.500`) vs. `NEXUSLearner(beam_width=1)` (`0.893` at $N=120$, `1.000` at $N=1600$). `NEXUSLearner(K=1)` still recomputes contrastive evidence across the full experience log and revises its top hypothesis, whereas `NoRevisionLearner` freezes each word on its first single example.
- On 2-shot novel word learning (`test_novel_word_in_known_frame`), $K \ge 5$ retains competing hypotheses in the beam (posterior weight $\approx 0.075$ on distractor kinds present in the True support scene), producing honest epistemic uncertainty (`Brier = 0.0043`, $p_{\text{true}} \approx 0.913$), whereas $K = 1$ collapses epistemic uncertainty to the top-1 noise floor (`0.0004`).

---

## 2. D2: Observation Noise Parameter $\varepsilon \in \{0.001, 0.01, 0.02, 0.05, 0.1\}$ (`EXPLORATORY`)

| Noise $\varepsilon$ | Clean `test_novel_combination` Acc | Clean ECE | Clean Brier Score | 10% Uncorrected Label Noise Acc ($N=400$) |
| :---: | :---: | :---: | :---: | :---: |
| `0.001` | `1.000` | `0.0010` | `0.000001` | `1.000` |
| `0.010` | `1.000` | `0.0100` | `0.000100` | `1.000` |
| `0.020` (Default) | `1.000` | `0.0200` | `0.000400` | `1.000` |
| `0.050` | `1.000` | `0.0500` | `0.002500` | `1.000` |
| `0.100` | `1.000` | `0.1000` | `0.010000` | `1.000` |

- **Finding**: Under noise-free test labels, a fully converged posterior ($w_0 = 1.0$) predicts $p_{\text{true}} \in \{\varepsilon, 1 - \varepsilon\}$, so $\text{ECE} = \varepsilon$ and $\text{Brier} = \varepsilon^2$. Setting $\varepsilon = 0.02$ provides a principled trade-off between sharp calibration on clean test splits ($\text{ECE} = 0.020$, $\text{Brier} = 0.0004$) and finite log-likelihood penalties ($\log \varepsilon = -3.912$) when misleading or noisy observations occur before correction.

---

## 3. D3: Sample Complexity Learning Curves & Gap to A8 Ideal Observer (`EXPLORATORY`)

### 3.1 Full World Learning Curve (`test_novel_combination`, `seeds 0..4`)

| Stratified Training Size $N$ | `NEXUSLearner` | Baseline A (`Memorizer`) | Baseline B (`CrossSituational`) | Baseline C (`NoRevision`) |
| :---: | :---: | :---: | :---: | :---: |
| `50` | `0.532 ± 0.033` | `0.500` | `0.500` | `0.500` |
| `100` | `0.896 ± 0.130` | `0.500` | `0.500` | `0.502` |
| `200` | **`1.000 ± 0.000`** | `0.500` | `0.500` | `0.502` |
| `400` | **`1.000 ± 0.000`** | `0.500` | `0.500` | `0.502` |
| `800` | **`1.000 ± 0.000`** | `0.500` | `0.500` | `0.502` |
| `1600` | **`1.000 ± 0.000`** | `0.500` | `0.500` | `0.496` |

### 3.2 Gap to Exhaustive Bayesian Ideal Observer on Reduced World ($6! = 720$ Bijections)

On the reduced world (`cube, ball, cone`, `red, blue`, `chases`), we compared `NEXUSLearner` directly against `ExhaustiveIdealObserver` (which enumerates all $6! = 720$ bijective lexicons with the true grammar given):

| Reduced Training Examples $N_{\text{red}}$ | A8 `ExhaustiveIdealObserver` (True Syntax Given) | `NEXUSLearner` (Jointly Inducing Syntax + Lexicon) | Gap ($\text{Ideal} - \text{NEXUS}$) |
| :---: | :---: | :---: | :---: |
| `4` (bare NPs only) | `0.500` | `0.500` | `0.000` |
| `8` (bare NPs + 1 adj) | `1.000` | `0.500` | `+0.500` (transitive verb unseen) |
| `12` (first 2 transitive pairs) | `1.000` | `0.833` | `+0.167` |
| `16` (all constructions seen) | `1.000` | **`1.000`** | **`0.000`** |

- **Why the gap exists at $N_{\text{red}} = 8$ and $12$**: `ExhaustiveIdealObserver` is given the fixed syntactic structure (`np_order`, `SOV` template, and exact bijection constraint over the 6 words), so by elimination after 5 words are grounded at $N_{\text{red}} = 8$, the 6th word (`zorp -> rel:chases`) is deduced *before* any transitive sentence is even observed. `NEXUSLearner` does not assume a closed-world bijection over unseen syntactic categories and must observe `zorp` in at least a few transitive clauses ($N_{\text{red}} = 16$) to induce its depth-2 category `(S\NP)\NP` and argument order.

---

## 4. D4: Hypothesis-Space Ablations (`EXPLORATORY`)

| Ablation Variant | `test_novel_combination` ($N=100$) | `test_novel_combination` ($N=1600$) | `dev` NP Constructions (`np_assert`, `np_ask`) | `dev` Transitive Constructions (`trans_assert`, `trans_ask`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full `NEXUSLearner`** (induces syntax + lexicon, depth $\le 2$) | `0.896` | `1.000` | `1.000` | `1.000` |
| **Given True Lexicon** (induces slash directions & symmetry only) | `0.974` | `1.000` | `1.000` | `1.000` |
| **Given True Syntax** (induces lexical predicates only) | `0.994` | `1.000` | `1.000` | `1.000` |
| **Slash Depth $\le 1$** (`max_slash_depth = 1`, no `(S\NP)\NP` etc.) | `0.574` | `0.574` | **`1.000`** | **`0.500` (0% parsed)** |

- **Finding**: Removing slash depth 2 (`max_slash_depth = 1`) leaves `np_assert` and `np_ask` at `1.000` (since `N`, `N/N`, `N\N`, `S/S`, `S\S` have depth $\le 1$), but collapses `trans_assert` and `trans_ask` to `0.500` (100% unparseable), proving that depth-2 binary functors (`(S\NP)\NP`, `(S\NP)/NP`, `(S/NP)/NP`) are structurally necessary and non-redundant for transitive argument binding.

---

## 5. D5: Label Noise Robustness Before vs. After Corrections (`EXPLORATORY`)

| Training Label Corruption Rate $\eta$ | $N=120$ Before Corrections | $N=120$ After Corrections | $N=400$ Before Corrections | $N=400$ After Corrections |
| :---: | :---: | :---: | :---: | :---: |
| `0%` | `0.863` | `0.863` | `1.000` | `1.000` |
| `2%` | `0.853` | `0.863` | `1.000` | `1.000` |
| `5%` | `0.920` | `0.863` | `1.000` | `1.000` |
| `10%` | `0.765` | `0.863` | `1.000` | `1.000` |
| `20%` | `0.798` | `0.863` | `0.990` | **`1.000`** |

- **Finding**: At $N = 400$, cross-situational likelihood pooling tolerates up to $10\%$ random label corruption with zero accuracy loss (`1.000`) and degrades slightly to `0.990` at $20\%$ label noise. In the low-sample regime ($N = 120$), $10\%\text{–}20\%$ label noise degrades accuracy to `0.765–0.798` before corrections, and replaying `Correction` events over the JSONL experience log restores the exact clean-data accuracy (`0.863` at $N=120$, `1.000` at $N=400$).

---

## 6. D6: New Held-Out Probes (`EXPLORATORY`)

| Diagnostic Probe Suite | Measured Accuracy / Parse Rate (`seeds 0..4`) | Mechanism Verified / Boundary Exposed |
| :--- | :---: | :--- |
| **Role Reversal: Asymmetric Relation (`chases(A, B)` vs. `chases(B, A)`)** | **`1.000`** | Distinguishes subject vs. object binding (`rel:chases(x0,x1)`). |
| **Role Reversal: Symmetric Relation (`next_to(A, B)` vs. `next_to(B, A)`)** | **`1.000`** | Induces bidirectional symmetry (`rel:next_to:sym(x0,x1)`). |
| **Stacked 2-Adjective Single NP (`color + size + noun`)** | **`1.000`** | Zero-shot recursive modifier application (`N/N` or `N\N`). |
| **Stacked 4-Adjective Transitive Clause (`2 adj on subj + 2 adj on obj`)** | **`1.000`** | Composes 7-token sentences (exceeding all training lengths). |
| **Unseen `adj + novel_noun` After 2-Example Bare-NP Support** | **`1.000`** | Composes newly learned `N` with known `N/N`, `N\N`, and verbs. |
| **Conflicting 3-Adjective NP (`color_1 + color_2 + size + noun`)** | **`0.000` parse rate** (`p_true = 0.50`) | **Boundary condition**: `_apply_forward` / `_apply_backward` enforces sortal exclusivity (at most one modifier per attribute family per noun). |

---

## 7. D7: Calibration Reliability Diagrams & Murphy Brier Decomposition (`EXPLORATORY`)

Murphy's (1973) decomposition partitions Brier score into $\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$ (lower Reliability = better calibration; higher Resolution = sharper discrimination; $\text{Uncertainty} = 0.25$ on balanced splits):

| System & Regime | Pooled Items | Brier Score ($\downarrow$) | Reliability ($\downarrow$) | Resolution ($\uparrow$) | Uncertainty | Reconstructed Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NEXUSLearner` ($N=1600$, `test_novel_combination`)** | `2000` | **`0.000400`** | **`0.000400`** | **`0.250000`** | `0.250000` | `0.000400` |
| **`NEXUSLearner` ($N=1600$, `test_novel_word` after 2 support)** | `1000` | **`0.004360`** | **`0.004318`** | **`0.250000`** | `0.250000` | `0.004318` |
| **`NEXUSLearner` ($N=100$, `test_novel_combination`)** | `2000` | `0.057074` | `0.001506` | `0.194501` | `0.250000` | `0.057004` |
| **Baseline B (`CrossSituational`, $N=1600$, `test_novel_combination`)** | `2000` | `0.267310` | `0.017102` | `0.000000` | `0.250000` | `0.267102` |

### Reliability Diagram Bins (`NEXUSLearner` at $N = 100$ on `test_novel_combination`, 2000 Pooled Items)

| Confidence Bin $[a, b]$ | Item Count | Mean Predicted $p_{\text{true}}$ | Empirical True Rate | Absolute Calibration Gap |
| :---: | :---: | :---: | :---: | :---: |
| `[0.0, 0.1]` | `774` | `0.0212` | `0.0000` | `0.0212` |
| `[0.2, 0.3]` | `3` | `0.2768` | `0.0000` | `0.2768` |
| `[0.4, 0.5]` | `444` | `0.4998` | `0.4977` | **`0.0020`** |
| `[0.6, 0.7]` | `1` | `0.6920` | `1.0000` | `0.3080` |
| `[0.7, 0.8]` | `24` | `0.7202` | `1.0000` | `0.2798` |
| `[0.9, 1.0]` | `754` | `0.9754` | `1.0000` | `0.0246` |

- Notice how at $N = 100$, when `NEXUSLearner` cannot yet parse a sentence (`444` items in bin `[0.4, 0.5]`), it outputs $p_{\text{true}} \approx 0.500$ where the empirical accuracy is `0.498` (calibration gap `0.0020`), rather than making overconfident guesses.

---

## 8. D8: Failure Taxonomy & Honest Architectural Limitations (`EXPLORATORY`)

Across low-sample ($N = 60$), noisy ($\eta = 0.20$), and structural ablation regimes, `explain()` categorizes all observed failures into four explicit mechanisms:

1. **Unparseable Slash Direction / Missing Category at Low $N$ (`111` errors at $N = 60$ across 5 seeds)**:
   - *Mechanism*: When a modifier or transitive verb has been observed only 1–2 times alongside nouns whose own `kind:*` predicates have not yet stabilized, `_propose_word_candidates` cannot yet lock in the modifier's slash direction (`N/N` vs. `N\N`) or verb category (`(S\NP)\NP` vs. `(S/NP)/NP`).
   - *Example*: `test_novel_combination-00001` (`['glorp', 'dax', 'ko']`, Seed 0 at $N=60$): `No valid categorial derivation (> or <) spans the sentence.`
2. **Underdetermined Lexical Mapping Before Contrastive Pair ($19$ errors at $N = 60$)**:
   - *Mechanism*: Before a word has been observed in a minimal contrast pair (or across enough scenes to rule out co-present distractor attributes), an adjective may temporarily map to a co-occurring attribute on a distractor object, or the question marker `ko` may be misanalyzed as a content word before appearing in long transitive questions.
   - *Example*: `test_novel_combination-00002` (`['lum', 'wug', 'mip', 'blick', 'ko']`, Seed 0 at $N=60$): `ko` was temporarily assigned `size:large` instead of `Q`, producing `ASSERT(... & size:large(x1))` which failed because `obj_3.size = small`.
3. **Slash Depth Expressivity Boundary (`50/50` transitive failures when `max_slash_depth = 1`)**:
   - *Mechanism*: Binary relations require two argument slots (`(S\NP)\NP`, `(S\NP)/NP`, `(S/NP)/NP`, depth 2). Restricting the categorial hypothesis space to depth $\le 1$ renders all transitive clauses unparseable. Similarly, ditransitive verbs (`give(x, y, z)`) or clause-embedding complementizers would require slash depth $3$, which is outside the Phase 1 depth-2 hypothesis space.
4. **Sortal Exclusivity & Existential-Conjunction-Only Semantics**:
   - *Mechanism*: Phase 1 logical forms are strictly positive existential conjunctions $\exists \vec{x} \bigwedge_i A_i(\vec{x})$, and each object has one single value per attribute family (`kind`, `color`, `size`). Consequently:
     - Stacked modifiers from the *same* attribute family on a single NP (e.g., `red blue cube`, D6 probe) are rejected by sortal exclusivity (`0.0%` parse rate).
     - Universal quantification (`every cube is red`), negation (`not red`), disjunction (`red or blue`), and relative clauses (`the cube that chases the ball is large`) are not representable in the Phase 1 logical language.
     - Concept restructuring (splitting an over-general predicate or merging synonyms when ontology granularity shifts) is not active in Phase 1 (deferred to Phase 3).

---

## 9. D9: SQLite Persistence Round-Trip Verification (`EXPLORATORY`)

`SQLiteLearnerRepository` ([repository.py](file:///Users/laksharajjha/Downloads/NEXUS/nexus/storage/repository.py), [schema.sql](file:///Users/laksharajjha/Downloads/NEXUS/nexus/storage/schema.sql)) persists the full learner state to a relational SQLite database and reloads it in an isolated Python subprocess:
- `subprocess_roundtrip_success`: **`True`**
- `max_prediction_abs_diff`: **`0.0`** (bit-for-bit identical `p_true` across all test items)
- `explanation_provenance_identical`: **`True`** (identical derivation trees, supporting example IDs, and witnesses)
- `what_changed_identical`: **`True`** (identical promoted/demoted/created hypothesis records)
