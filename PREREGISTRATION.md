# NEXUS-0: Phase 1 Pre-Registration

**Status**: FROZEN (Timestamped prior to Phase 1 implementation)  
**Date**: 2026-10-20  
**Target Repository**: `https://github.com/Laksharajjha/NEXUSBrain.git`  
**Phase**: Phase 1 (Grammar + Meaning Induction)

---

## 1. Research Question & Phase 1 Hypothesis

**Central Research Question**:  
Can an inspectable learning system induce word meanings and grammatical constructions from (scene, sentence) pairs, generalize to held-out combinations and novel lexical items, and retain early competencies, without pretrained weights, embeddings, or neural language models?

**Phase 1 Hypothesis**:  
Jointly maintaining cross-situational co-occurrence statistics and candidate syntactic-semantic combinators under Bayesian / description-length scoring enables a learner to:
1. Disambiguate word-to-concept mappings across ambiguous scenes.
2. Induce the correct constituent structure and argument alignment (e.g. within-NP modifier order, clause-level argument order).
3. Zero-shot generalize to novel combinations of known words (`test_novel_combination`).
4. Rapidly bind a newly encountered word within a known grammatical frame from minimal contrastive support (`test_novel_word_in_known_frame`).
5. Outperform both a memorizer baseline and an unstructured cross-situational baseline that lacks compositional grammar induction.

---

## 2. Hypothesis Space & Inductive Bias

The learner searches an explicit, bounded hypothesis space:
1. **Lexical Semantic Space**:
   - Every word token $w \in V$ maps to a predicate or relation symbol in the target ontology:
     - 1-place predicates: `kind:*`, `color:*`, `size:*`
     - 2-place relations: `rel:*`
     - Speech-act / functional markers: question particle `Q`
2. **Syntactic & Combinatory Space**:
   - Categorial grammar / compositional rewrite rules defining how syntactic categories compose and bind lambda variables:
     - $N$: Nominal predicate $\lambda x. P(x)$
     - $Adj$: Modifier functor $\lambda P \lambda x. [P(x) \land A(x)]$
     - $NP$: Saturated entity reference
     - $V$: Transitive relation functor $\lambda x \lambda y. R(x, y)$
     - $S$: Evaluated proposition / speech act
3. **Scoring Function**:
   - Scoring balances likelihood of the observed scene-sentence pairs against description length (MDL / penalized log-posterior):
     $$\mathcal{S}(H) = \sum_{(s, e)} \log P(e \mid s, H) - \lambda \cdot \text{Complexity}(H)$$
   - Candidate hypotheses are pruned only when an alternative hypothesis is supported by overwhelming posterior evidence ($\Delta \mathcal{S} > \tau$).

---

## 3. Evaluated Systems & Baselines

Four systems will be compared under identical harness protocols across 20 randomized seeds (`world_mode=randomized`):
1. **Baseline A (Memorizer)**:
   - Stores exact `(scene_signature, sentence_tokens)` pairs.
   - Provides Laplace-smoothed empirical empirical probabilities on exact matches; chance ($0.50$) on novel pairs.
2. **Baseline B (Cross-Situational Only)**:
   - Tracks token-feature co-occurrence counts without hierarchical syntax or constituent compositionality.
   - Predicts scene truth by checking if co-occurring features exist, without structural role binding.
3. **Baseline C (No-Revision Grammar Learner)**:
   - Commits eagerly to the first consistent lexical and grammatical hypotheses without maintaining competing alternatives or score-based pruning.
4. **NEXUS Learner**:
   - Full Phase 1 system: joint cross-situational semantics + categorial grammar induction + hypothesis scoring + full provenance tracking.

---

## 4. Evaluation Protocol, Splits & Primary Metrics

### Splits
- `train`: 180 examples across curriculum stages (`np_assert`, `trans_assert`, `np_ask`, `trans_ask`).
- `dev`: 60 balanced examples spanning training constructions with novel scenes.
- `test_novel_combination`: 60 balanced examples with seen words combined in unseen syntactic/semantic configurations.
- `test_novel_word_in_known_frame`: 40 balanced examples featuring a held-out noun in seen frames, evaluated across a 4-example contrastive support stream.
- `test_longer`: 40 balanced examples with multi-clause or conjoined structures exceeding the maximum training length.
- `test_contradiction`: 40 balanced examples evaluated across a 4-example corrective feedback stream.

### Primary Metrics
- **Split Accuracy**: Mean fraction of correct truth-value predictions ($\ge 0.50$ threshold for binary classification).
- **Logical Form (LF) Accuracy**: Exact match between learner's derived LF and ground-truth LF (when learner produces explicit derivations).
- **Retention**: Accuracy on early constructions (`np_assert`) measured at the end of training divided by accuracy at its curriculum boundary.
- **Calibration (ECE)**: Expected Calibration Error across 10 bins on test splits.
- **Model Size**: Parameter / rule count.

---

## 5. Frozen Pass Criteria

For Phase 1 to be declared successful, all of the following criteria must be met over 20 random seeds:

1. **Novel Combination Generalization**:
   $$\text{Mean Accuracy}(\text{NEXUS}, \text{test\_novel\_combination}) \ge 0.80$$
2. **Few-Shot Novel Word Acquisition**:
   $$\text{Mean Accuracy}(\text{NEXUS}, \text{test\_novel\_word\_in\_known\_frame}) \ge 0.70$$
3. **Statistical Superiority over Baselines**:
   - NEXUS must beat the best of Baseline A (Memorizer) and Baseline B (Cross-Situational) by at least **0.15 absolute** on both `test_novel_combination` and `test_novel_word_in_known_frame`.
   - The 95% paired bootstrap confidence interval (10,000 resamples over the 20 seeds) for the accuracy difference:
     $$\text{Acc}(\text{NEXUS}) - \max(\text{Acc}(A), \text{Acc}(B))$$
     must strictly exclude zero on both splits.
4. **Shortcut Immunity**:
   - The non-compositional shortcut baselines (`SentenceOnly` and `SceneOnly`) must stay within the 95% binomial interval of chance ($[0.37, 0.63]$ for $N=60$) on all test splits.
5. **Catastrophic Forgetting Resistance (Retention)**:
   $$\text{Retention}(\text{NEXUS}, \text{early\_constructions}) \ge 0.90$$

---

## 6. Pre-Registration Failure Protocol

If any of the above criteria are not satisfied:
- **Rule**: Report the failure truthfully and completely with diagnostic analyses of errors, hypothesis competition logs, and provenance traces.
- **Strict Prohibition**: Under no circumstances will the world generator, evaluation harness, test distributions, or pass criteria be altered post-hoc to force a pass. Any modification to hypotheses or learning mechanisms constitutes a separate Phase 1.1 iteration that must be documented as such.
