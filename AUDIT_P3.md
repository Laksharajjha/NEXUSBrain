# NEXUS-0 Phase 3: Restructuring Worlds & Harness Audit (`AUDIT_P3.md`)

**Stage**: STAGE P3-A (New Worlds and Harness Audit — Gate P3-A)  
**Seed Range**: Fresh Seeds `2000..2049` ($N=50$ seeds)  
**Status**: **ALL AUDITS PASSED — GATE P3-A CLEARED**  
**Phase 1 PREREGISTRATION.md SHA-256**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b` (Strictly Unmodified)

---

## Executive Summary

Phase 3 tests the core capability of the NEXUS learner: dynamically changing its conceptual representations (split, merge, exception list) when evidence demands, and demonstrating when that beats maintaining a rigid structure patched with ad-hoc exceptions.

Before implementing any Phase 3 learner logic, Stage P3-A designs and audits five completely new world variants in [`nexus/world/restructure.py`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/world/restructure.py) without modifying Phase 1 worlds or splits:
- **W1 (Split-required)**: Two kinds share all surface properties and one word form early, then diverge in relational contexts.
- **W2 (Exception-sufficient)**: One concept with exactly 1–2 genuine exceptions. Splitting here is strictly penalized as over-splitting.
- **W3 (Merge-required)**: Two word forms behave identically in all grammatical contexts (synonyms) and should be merged.
- **W4 (Ambiguous-evidence)**: Evidence is insufficient to decide; the correct normative behavior is to maintain competing hypotheses in the beam with calibrated lower confidence.
- **W5 (Exceptions-cannot-match)**: Divergence occurs pervasively across data such that an exception list grows $O(N)$ and is strictly worse under MDL than splitting.

Every world was subjected to rigorous audits across **50 fresh random seeds (`2000..2049`)**:
1. **Oracle Ceiling**: [`RestructureOracle`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/baselines/oracle.py) scores **1.000 (100%)** on train, dev, and all test splits across all 50 seeds.
2. **Shortcut Baselines**: [`SentenceOnly`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/baselines/sentence_only.py) and [`SceneOnly`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/baselines/scene_only.py) stay within a 95% $t$-confidence interval of **0.500** on all test splits across all 50 seeds ($0.500 \pm 0.000$, CI `[0.5000, 0.5000]`).
3. **Differential Model Checker Test**: An independent naive model checker ([`naive_evaluate`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/world/logic.py)) produced **0 mismatches** against the primary evaluator across all generated items.
4. **Identifiability**: Every needed restructuring decision has $\ge 5$ distinguishing training examples across all 50 seeds (min observed count is $22 \ge 5$).
5. **Split Sample Sizes**: Exact sample sizes ($n$) are documented for every split.
6. **Exhaustive Ideal Observer Upper Bounds**: On reduced versions of W1 and W2, marginal likelihood model selection confirms the correct identifiable structure:
   - In reduced W1, `split` wins with $P(\text{split}) = 0.9890$ and reaches a **1.000** test ceiling.
   - In reduced W2, `exception` wins with $P(\text{exception}) = 0.9997$ (over-splitting is decisively rejected) and reaches a **1.000** test ceiling.

---

## 1. Split Sample Sizes (`n`) by World

Every split is constructed using twin contrast pairs (or value-swap quads), guaranteeing exact 50% True / 50% False label balance and identical marginal distributions of objects, attributes, and relations.

| World | Train $n$ | Dev $n$ | Test Split Name | Test $n$ | Label Balance (True / False) | Purpose / Description |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **`w1_split`** | 400 | 100 | `test_heldout` | 100 | 50% / 50% | Stage 1 shared bare/modified NPs (160 items); Stage 2 relational divergence (240 items). Test holds out novel combinations of both senses. |
| **`w2_exception`** | 300 | 100 | `test_heldout` | 100 | 50% / 50% | Regular concept with exactly 2 genuine idiosyncratic exceptions in train. Dev and test strictly evaluate the regular rule. |
| **`w3_merge`** | 300 | 100 | `test_heldout` | 100 | 50% / 50% | Two exact synonyms ($W_1$ and $W_2$) interleaved 50/50 across all constructions. Test evaluates unseen held-out combinations. |
| **`w4_ambiguous`** | 80 | 40 | `test_ambiguous` | 100 | 50% / 50% | Balanced training where $W_{\text{amb}}$ appears only in ambiguous contexts. Test presents diagnostic probes separating $H_1$ and $H_2$. |
| **`w5_exceptions_cannot_match`** | 300 | 100 | `test_heldout` | 100 | 50% / 50% | Pervasive 50/50 divergence across all training pairs. Test evaluates held-out generalization where an exception list fails. |

---

## 2. Oracle Ceiling Audit (50 Seeds: 2000..2049)

[`RestructureOracle`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/baselines/oracle.py) operates behind the strict import guard (`_check_oracle_import_guard`) and is allowed to read the ground-truth grammar and logical form structure. It evaluates semantic truth with 100% accuracy across every split.

| World | Train Accuracy (Mean $\pm$ Std) | Dev Accuracy (Mean $\pm$ Std) | Test Accuracy (Mean $\pm$ Std) | Min Across 50 Seeds | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`w1_split`** | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 1.000 | **PASS (100% Ceiling)** |
| **`w2_exception`** | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 1.000 | **PASS (100% Ceiling)** |
| **`w3_merge`** | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 1.000 | **PASS (100% Ceiling)** |
| **`w4_ambiguous`** | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 1.000 | **PASS (100% Ceiling)** |
| **`w5_exceptions_cannot_match`** | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 1.000 | **PASS (100% Ceiling)** |

---

## 3. Shortcut Baselines Audit (50 Seeds: 2000..2049)

Both [`SentenceOnly`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/baselines/sentence_only.py) and [`SceneOnly`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/baselines/scene_only.py) were trained on `train` and evaluated across all splits for 50 seeds ($df = 49, t_{0.975, 49} = 2.0096$).

| World | Split | `SentenceOnly` Mean $\pm$ Std | `SentenceOnly` 95% $t$-CI | Contains 0.50? | `SceneOnly` Mean $\pm$ Std | `SceneOnly` 95% $t$-CI | Contains 0.50? | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`w1_split`** | `dev` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| `w1_split` | `test_heldout` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| **`w2_exception`** | `dev` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| `w2_exception` | `test_heldout` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| **`w3_merge`** | `dev` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| `w3_merge` | `test_heldout` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| **`w4_ambiguous`** | `dev` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| `w4_ambiguous` | `test_ambiguous` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| **`w5_exceptions_cannot_match`** | `dev` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |
| `w5_exceptions_cannot_match` | `test_heldout` | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | $0.500 \pm 0.000$ | $[0.5000, 0.5000]$ | **Yes** | **PASS** |

*(Note: In `w2_exception`, `train` carries the 2 designated exception items; on all held-out evaluation splits, both shortcut models stay at chance $0.5000 \pm 0.0000$.)*

---

## 4. Independent Naive Model Checker Differential Test

An independent model checker ([`naive_evaluate`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/world/logic.py)), written with a separate AST evaluation recursive interpreter, was evaluated against [`evaluate`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/world/logic.py) on every generated item across all splits:
- **Total Comparisons**: $> 50,000$ items across 50 seeds.
- **Mismatches**: **0 mismatches** (100% exact match).

---

## 5. Identifiability Audit (50 Seeds: 2000..2049)

To ensure that the dataset permits identifiable learning, each world requires $\ge 5$ distinguishing training examples for its core structural decision.

| World | Needed Restructuring Decision | Min Count Across 50 Seeds | Max Count Across 50 Seeds | Meets $\ge 5$ Criterion? |
| :--- | :--- | :---: | :---: | :---: |
| **`w1_split`** | Split polysemous word into Sense $A$ and Sense $B$ | **152** | 230 | **PASS ($\ge 5$)** |
| **`w2_exception`** | Retain concept and record exception items | **298** | 298 | **PASS ($\ge 5$)** |
| **`w3_merge`** | Merge synonym words $W_1$ and $W_2$ | **22** | 52 | **PASS ($\ge 5$)** |
| **`w4_ambiguous`** | Diagnostic probes separating ambiguous hypotheses | **100** | 100 | **PASS ($\ge 5$)** |
| **`w5_exceptions_cannot_match`** | Split pervasively diverging senses | **90** | 150 | **PASS ($\ge 5$)** |

Every world provides abundant evidence ($22\text{--}298$ examples) for the learner to identify the true underlying structure.

---

## 6. Exhaustive Ideal Observer on Reduced W1 and W2

On a reduced ontology (2 kinds, 2 colors, 1 relation), the three candidate structural models were scored by exact marginal likelihood under the Minimum Description Length (MDL) prior:
$$\log P(M \mid D) = \log P(D \mid M) - \text{MDL}(M)$$
where $\text{MDL}(M) = \text{MDL}_{\text{base}} + \text{cost}(M)$.

### Reduced World 1 (`w1_split`)
- **Data**: $N_{\text{train}} = 400$, with $40$ diverging relational examples between Sense $A$ and Sense $B$.
- **Model Scores**:
  - $M_{\text{split}}$ (2 senses, constant entry bit-cost $24$ bits): $\log P(M_{\text{split}} \mid D) = -24.40$. Posterior: **$P(\text{split}) = 0.9890$**.
  - $M_{\text{exception}}$ (single concept $+ 40$ stored exceptions $\times 8$ bits $= 320$ bits): $\log P(M_{\text{exception}} \mid D) = -320.40$. Posterior: $P(\text{exception}) \approx 2.78 \times 10^{-129}$.
  - $M_{\text{multiple}}$ (beam uncertainty overhead): $\log P(M_{\text{multiple}} \mid D) = -28.90$. Posterior: $P(\text{multiple}) = 0.0110$.
- **Winning Structure**: **`split`** ($P = 0.9890$).
- **Generalization Ceiling**: **$1.000$ (100%)** on held-out test data.

### Reduced World 2 (`w2_exception`)
- **Data**: $N_{\text{train}} = 300$, with exactly $2$ genuine idiosyncratic exception items.
- **Model Scores**:
  - $M_{\text{exception}}$ (single concept $+ 2$ stored exceptions $\times 8$ bits $= 16$ bits): $\log P(M_{\text{exception}} \mid D) = -16.30$. Posterior: **$P(\text{exception}) = 0.9997$**.
  - $M_{\text{split}}$ (2 senses, redundant entry cost $24$ bits): $\log P(M_{\text{split}} \mid D) = -24.30$. Posterior: $P(\text{split}) = 0.0003$. Over-splitting is decisively rejected!
  - $M_{\text{multiple}}$ (beam uncertainty overhead): $\log P(M_{\text{multiple}} \mid D) = -28.80$. Posterior: $P(\text{multiple}) \approx 3.7 \times 10^{-6}$.
- **Winning Structure**: **`exception`** ($P = 0.9997$).
- **Generalization Ceiling**: **$1.000$ (100%)** on regular held-out test data.

---

## Gate P3-A Sign-Off

All Stage P3-A audit checks have passed without exception:
- [x] Oracle = 100% across all splits in all 5 worlds.
- [x] Shortcut baselines at chance (0.500) on all test splits across 50 seeds.
- [x] Naive model checker differential test passes with 0 mismatches.
- [x] Identifiability counts $\ge 5$ across all 50 seeds.
- [x] Split sample sizes $n$ reported.
- [x] Exhaustive ideal observer upper bounds established for W1 and W2.

**GATE P3-A IS CLEARED.** Proceed to Stage P3-B (Pre-registration).
