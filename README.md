# From arXiv Abstracts to a Knowledge Graph

> An end-to-end information extraction pipeline on AI research paper abstracts: **NER (BiLSTM-CRF) → Relation Extraction → Knowledge Graph → LLM benchmark**.



##  Pipeline overview

```
arXiv abstracts  →  spaCy EntityRuler (weak supervision)  →  BIO labels
                                                                  ↓
                                                  Train/Dev/Test (paper-level)
                                                                  ↓
                                              GloVe + BiLSTM-CRF (Part B)
                                                                  ↓
                                  Dependency parsing + Verb families (Part C)
                                                                  ↓
                                       NetworkX Knowledge Graph (Part C)
                                                                  ↓
                                vs. flan-t5-large benchmark (Part D)
```

---

##  What this project demonstrates

1. A reproducible **weak-supervision pipeline** for domain-specific NER without manually annotated training data
2. A **rigorous evaluation protocol** (paper-level splits, BIO tagging, `seqeval` entity-level scoring, early stopping on dev F1)
3. **Contextual generalization** evidence via stress tests on unseen sentences
4. A **structural Knowledge Graph** built from extracted relations
5. **A specialized pipeline outperforming a 780M-parameter LLM** on this domain-specific task

---

##  Dataset

| Split | Papers | Sentences | Note |
|-------|--------|-----------|------|
| Train | 3,500 | 25,546 → 26,666 (after injection) | augmentations added here only |
| Dev | 750 | 5,455 | untouched |
| Test | 750 | 5,602 | untouched |

- Source: arXiv metadata snapshot (5,000 papers sampled from `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.NE`)
- **270 patterns** added to the `EntityRuler` (lexical + syntactic)
- Vocabulary: **21,560 tokens** (built from train only)
- GloVe 100d coverage: **68.6%** (14,799/21,560 — the rest is randomly initialized; domain-specific scientific terms are out of GloVe 6B)
- Tag set: 12 tags (BIO scheme, 5 entity types: METHOD, TASK, METRIC, DATA, CHALLENGE)
- Model: **2,394,772 trainable parameters**

---

##  Results

### Part B — NER training

Early stopping triggered at epoch 15 (no dev-F1 improvement for 3 epochs).

| Epoch | Train loss | Dev F1 |
|-------|------------|--------|
| 1     | 3.3381     | 0.9758 |
| 5     | 0.1275     | 0.9951 |
| 10    | 0.0409     | 0.9971 |
| **12** | **0.0284** | **0.9974** ← best |
| 15    | 0.0212     | 0.9971 (early stop) |

### Part B — NER test set (entity-level, `seqeval`)

| Entity | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| CHALLENGE | 1.0000 | 1.0000 | 1.0000 | 951 |
| DATA | 0.9954 | 0.9950 | 0.9952 | 2,385 |
| METHOD | 0.9978 | 0.9960 | 0.9969 | 5,060 |
| METRIC | 0.9969 | 0.9956 | 0.9962 | 1,599 |
| TASK | 0.9981 | 0.9981 | 0.9981 | 2,135 |
| **micro avg** | **0.9974** | **0.9965** | **0.9969** | **12,130** |

>  **Read this caveat**: training and silver-test labels both come from the same `EntityRuler`. This 99.7% F1 measures **rule imitation**, not true NER quality. The honest evaluations are the stress test (below) and the LLM comparison (Part D).

### Part B — Stress Test (5 unseen sentences)

| # | Sentence | Result | Verdict |
|---|----------|--------|---------|
| 1 | *"We introduce **QuantumFormer**, a sparse encoder for long-range reasoning."* | `quantumformer` → **B-METHOD** ✅ (unseen word, predicted from context) | **Pass** — true context generalization |
| 2 | *"Our routing protocol minimizes latency and maximizes throughput on the testbed."* | Only `latency` → B-METRIC | Partial — vocabulary-limited |
| 3 | *"The model achieves 87.5 accuracy on the new **ZebraBench** corpus."* | `model`, `accuracy`, `corpus` ✅ — **`ZebraBench` missed** | Fail |
| 4 | *"**Wibblenet** outperforms baselines on the chaotic time series prediction task."* | `time series` → DATA, `prediction` → TASK ✅ — **`Wibblenet` missed** | Fail |
| 5 | *"The system suffers from overfitting and is vulnerable to adversarial attacks."* | `overfitting` → B-CHALLENGE ✅ + `adversarial attacks` → B-CHALLENGE I-CHALLENGE ✅ (multi-token entity correctly grouped) | **Pass** |

**Score: 2 pass, 1 partial, 2 fail.** The model handles known vocabulary and the `"We introduce X"` pattern well; multi-token CHALLENGE entities are now correctly grouped. The remaining fails are unseen named entities (`ZebraBench`, `Wibblenet`) which the model cannot infer without character-level features.

### Part C — Knowledge Graph

- **296 relation triples** extracted across 5,000 abstracts
- 4 relation types (`solves`, `uses`, `improves`, `yields`)
- Synonym fusion (`technique`/`approach`/`strategy` → `method`) + blacklist filtering
- Final graph: **329 nodes, 287 edges**

**Top-5 most central concepts** (by degree):

| Rank | Concept | Degree |
|------|---------|--------|
| 1 | method | 71 |
| 2 | model | 21 |
| 3 | system | 19 |
| 4 | performance | 14 |
| 5 | information | 8 |

Sample triples:
- `system --[uses]--> model`
- `method --[yields]--> performance`
- `fusion --[yields]--> image`
- `application --[uses]--> ngram`

### Part D — LLM Benchmark

Side-by-side comparison on 5 random abstracts: our custom pipeline vs `google/flan-t5-large` (780M params).

| Input abstract | flan-t5-large output | Custom pipeline output |
|---|---|---|
| Bayesian network on fuzzy knowledge bases | `"[TABLECONTEXT] [TITLE] Fuzzy Bayesian Network..."` (artifact tokens) | `fuzzification \| improves \| quality` ✅ |
| Multiple component matching for person reidentification | `"[TABLECONTEXT] [TITLE] ... [TABLECONTEXT] COORDINATOR ..."` (hallucinated metadata tokens) | No relation (strict rules) |
| Genetic programming for fingerprint matching | `"[TABLECONTEXT] [TITLE] ... [PRECEDED_DONE] [TABLECONTEXT] NUMBER_OF_PAGES ..."` (gibberish) | No relation (strict rules) |
| Stochastic portfolio optimization | `"Subject: ... \| Relationship: ... \| Relationship: ... \| Subject: ..."` (broken format) | No relation (strict rules) |
| Epistemic irrelevance in credal nets | `"[TABLECONTEXT] [TITLE] ... COORDINATION_OF_NODES MODEL_OF_CONSTRUCTION NUMBER_OF_NODES ..."` (gibberish) | No relation (strict rules) |

**Result**: 5/5 abstracts, the LLM hallucinates training-data artifacts (`[TABLECONTEXT]`, `[TITLE]`, `[NUMBER_OF_PAGES]`, ...) instead of extracting structured relations. The LLM is unable to follow the format instruction even with explicit `Subject | Relation | Object` examples.

The custom pipeline is **conservative**: it returns 1 clean triple where rules match, nothing where they don't — a much more usable signal for downstream KG construction. **Trading recall for usable precision is the right tradeoff for a knowledge-graph pipeline.**






