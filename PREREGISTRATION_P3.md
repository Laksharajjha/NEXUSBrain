# NEXUS-0 Phase 3: Restructuring Pre-Registration (`PREREGISTRATION_P3.md`)

**Date**: 2026-10-26  
**Stage**: STAGE P3-B (Frozen Pre-Registration — Gate P3-B)  
**Commit Tag**: `prereg-p3`  
**Fresh Seeds**: `2000..2049` ($N=50$ random seeds)  
**Parent Phase 1 Hash**: `ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b` (`PREREGISTRATION.md` strictly unmodified)

---

## 1. Research Question & Purpose

Phase 1.6 established that truth scoring, co-occurrence, beam search, and MDL bit-cost priors each perform an identifiable function in synthetic conditions. Phase 3 tests the core architectural hypothesis of NEXUS:

> **Can the learner change its own conceptual and lexical structure when evidence demands (via explicit Split, Merge, and Exception operators), and does dynamic restructuring beat maintaining a rigid structure patched with ad-hoc exceptions?**

---

## 2. Hypotheses

- **H1 (Restructuring Beats No-Restructuring in W1 and W5)**:
  In worlds where word forms cover multiple underlying senses that diverge in relational behavior (W1: Split-required, W5: Exceptions-cannot-match), dynamic restructuring (the **Split** operator) will achieve superior generalization on held-out test combinations compared to fixed-structure learners ([`NEXUS-NoRestructure`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/learner/nexus_learner.py)) and exception-memorizing learners ([`NEXUS-ExceptionsOnly`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/learner/nexus_learner.py)).
- **H2 (Over-Splitting Control in W2)**:
  In worlds with genuine isolated exceptions (W2: Exception-sufficient, 1–2 defect examples), the Minimum Description Length (MDL) prior will prefer the **Exception** operator over the Split operator, preventing spurious over-splitting and outperforming eager splitters ([`NEXUS-SplitAlways`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/learner/nexus_learner.py)) in parsimony.
- **H3 (Synonym Unification in W3)**:
  In worlds where multiple word forms share identical distributions and extensions (W3: Merge-required), the learner will invoke the **Merge** operator to unify redundant lexical entries, compressing total description length without loss of predictive accuracy.
- **H4 (Calibrated Beam Uncertainty in W4)**:
  In worlds where early evidence is insufficient to distinguish competing hypotheses (W4: Ambiguous-evidence), the learner will retain multiple hypotheses in the beam margin and report calibrated lower confidence ($p \approx 0.50$), avoiding overconfident errors and achieving superior Brier scores on diagnostic probe tests.

---

## 3. Comparison Set

All learners implement the standard [`Learner`](file:///Users/laksharajjha/Downloads/NEXUS/nexus/eval/learner_api.py) protocol (`observe`, `correct`, `predict`, `model_size`). No learner receives generator metadata, twin pairings, or logical forms.

1. **`NEXUS-Restructure` (Full Primary Learner)**:
   Equipped with the three restructuring operators (**Split**, **Merge**, **Exception**) scored uniformly under the MDL posterior, plus beam search with uncertainty.
2. **`NEXUS-NoRestructure` (Phase 1.6 Baseline)**:
   Standard CCG beam learner without restructuring operators.
3. **`NEXUS-ExceptionsOnly` (Ablation / Patch Baseline)**:
   Restricted to retaining existing concepts and appending misclassifications to an exception list (Operator a only).
4. **`NEXUS-SplitAlways` (Stupid-Split Control)**:
   Splits lexical entries on every encountered contradiction without considering exception cost or MDL parsimony.
5. **`Baseline D` (Position-Aware Cross-Situational Baseline)**:
   Associative positional n-gram baseline without formal grammar or restructuring.
6. **`RestructureOracle` (Ceiling Baseline)**:
   Reads ground-truth grammar and disambiguated logical forms to define the experimental ceiling (100% on all splits).

---

## 4. Evaluation Metrics & Statistics

1. **Accuracy**: Binary truth prediction accuracy ($p \ge 0.5 \implies \text{True}$) on `dev` and held-out test splits (`test_heldout`, `test_ambiguous`).
2. **Calibration & Brier Score**:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
   $$\text{Confidence} = \frac{1}{N} \sum_{i=1}^N \max(p_i, 1 - p_i)$$
3. **Model Size**: Total stored parameter count (number of lexical entries / senses).
4. **Restructuring Events**: Tally of Split, Merge, and Exception operations executed during learning.
5. **Retention**: Accuracy on untouched background concepts after restructuring events occur.
6. **Statistical Tests**:
   - Evaluated across **50 fresh seeds (`2000..2049`)**.
   - **Paired Bootstrap Differences**: 1,000 resamples for pairwise differences ($\Delta = A - B$) with 95% percentile confidence intervals.

---

## 5. Frozen Numeric Pass Criteria

Grounded in the Exhaustive Ideal Observer bounds established in [`AUDIT_P3.md`](file:///Users/laksharajjha/Downloads/NEXUS/AUDIT_P3.md):

| Criterion ID | Hypothesis / World | Metric | Pass Condition | Feasibility / Bound |
| :---: | :--- | :--- | :--- | :--- |
| **C1** | **H1: W1 Split** | `test_heldout` Acc | `NEXUS-Restructure` $\ge 0.900$, and $\Delta(\text{Restructure} - \text{NoRestructure}) \ge +0.200$ (95% CI strictly $> 0$). | Ideal observer ceiling = 1.000. $M_{\text{split}}$ probability = 0.9890. |
| **C2** | **H1: W5 Scale** | `test_heldout` Acc | `NEXUS-Restructure` $\ge 0.900$, and $\Delta(\text{Restructure} - \text{ExceptionsOnly}) \ge +0.200$ (95% CI strictly $> 0$). | Ideal observer ceiling = 1.000. Exceptions grow $O(N)$ and fail on unseen scenes. |
| **C3** | **H2: W2 Over-Split** | Operator & Model Size | In W2, `NEXUS-Restructure` invokes 0 splits on target concept; model size is strictly less than `NEXUS-SplitAlways` ($\Delta \le -1.0$ entries, 95% CI strictly $< 0$); `test_heldout` Acc $\ge 0.950$. | Ideal observer ceiling = 1.000. $M_{\text{exception}}$ probability = 0.9997. |
| **C4** | **H3: W3 Merge** | Operator & Model Size | In W3, `NEXUS-Restructure` merges synonym pair into shared concept; model size $\le 19.0$ entries; `test_heldout` Acc $\ge 0.950$. | Compression saves duplicate entry bit-cost without loss of accuracy. |
| **C5** | **H4: W4 Uncertainty** | Brier & Confidence | In W4, on `test_ambiguous` diagnostic probes, `NEXUS-Restructure` retains multiple hypotheses with mean confidence $\le 0.650$ and Brier score $\le 0.300$. | Ambiguous evidence warrants calibrated posterior entropy. |
| **C6** | **Retention** | Untouched Concepts | Background concept prediction accuracy on `dev` after restructuring $\ge 0.950$. | Preserves existing semantic representations without catastrophic forgetting. |

---

## 6. Execution Rules & Integrity Protocol

1. **Frozen Worlds & Test Splits**:
   Worlds W1–W5 defined in `nexus/world/restructure.py` are frozen. No worlds, split generators, or test splits may be modified during or after evaluation.
2. **One-Shot Evaluation**:
   Held-out test splits for seeds `2000..2049` are evaluated strictly ONCE during Stage P3-D. Tuning and debugging are restricted to `dev`.
3. **Honest Reporting**:
   If any criterion fails, the failure will be reported plainly with full forensic error categorization in `PHASE3_REPORT.md`. Modifying criteria or world parameters to achieve a pass is strictly prohibited.
