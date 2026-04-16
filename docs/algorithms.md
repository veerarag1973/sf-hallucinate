# Scoring Algorithms

`sf-hallucinate` uses a **hybrid similarity** metric that combines TF-IDF cosine similarity with token F1 overlap. This page explains the math behind each component and how they are combined.

---

## Overview

```
claim  ──┬── TF-IDF cosine ──┐
         │                   ├── weighted sum ──► hybrid_similarity ∈ [0, 1]
         └── token F1     ──┘
```

$$
\text{hybrid}(c, r) = w \cdot \text{tfidf\_cosine}(c, r) + (1 - w) \cdot \text{token\_f1}(c, r)
$$

where $w$ is `EvalConfig.tfidf_weight` (default `0.6`).

---

## TF-IDF Cosine Similarity

### Term Frequency (TF)

For a sentence tokenised into $n$ tokens, the TF of term $t$ is:

$$
\text{tf}(t) = \frac{\text{count}(t)}{n}
$$

### Inverse Document Frequency (IDF)

Given $N$ sentences (hypothesis + all reference sentences) and document frequency $\text{df}(t)$, smoothed IDF is:

$$
\text{idf}(t) = \log\!\left(\frac{N+1}{\text{df}(t)+1}\right) + 1
$$

The $+1$ smoothing avoids division by zero and prevents zero weights for terms that appear in all documents.

### TF-IDF Vector

$$
\text{tfidf}(t) = \text{tf}(t) \times \text{idf}(t)
$$

### Cosine Similarity

$$
\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}
$$

Empty vectors return `0.0`.

**Why TF-IDF?** It down-weights common function words (even without a stopword list) and up-weights rare, content-bearing terms. Two sentences mentioning "Eiffel Tower" score higher than two sentences both mentioning "the".

---

## Token F1

Token F1 is precision/recall on bags of content words (after stopword removal).

Let $P$ = set of tokens in the hypothesis, $R$ = set of tokens in the reference:

$$
\text{precision} = \frac{|P \cap R|}{|P|}, \quad
\text{recall}    = \frac{|P \cap R|}{|R|}
$$

$$
\text{token\_f1} = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

Returns `0.0` for empty inputs.

**Why token F1?** TF-IDF cosine can miss paraphrases that share content words but not identical forms. Token F1 adds explicit overlap counting. Together, the two metrics are complementary.

---

## Hybrid Similarity

$$
\text{hybrid}(c, r) = 0.6 \cdot \text{tfidf\_cosine}(c, r) + 0.4 \cdot \text{token\_f1}(c, r)
$$

Result is always in $[0.0, 1.0]$.

The weight $w = 0.6$ is the default; it can be changed via `EvalConfig(tfidf_weight=...)`.

---

## Faithfulness Scoring

For a given LLM output and reference document:

1. **Claim extraction** — split the output into sentences, filter out questions and meta-phrases.
2. **Best-match search** — for each claim $c_i$, compute $\text{hybrid}(c_i, r_j)$ against every reference sentence $r_j$; keep the maximum:

$$
s_i = \max_j \; \text{hybrid}(c_i, r_j)
$$

3. **Aggregate faithfulness** — mean over all claims:

$$
\text{faithfulness\_score} = \frac{1}{N} \sum_{i=1}^{N} s_i
$$

4. **Hallucination risk** — complement:

$$
\text{hallucination\_risk} = 1 - \text{faithfulness\_score}
$$

5. **Grounded flag per claim** — claim $c_i$ is *grounded* when:

$$
s_i \geq \text{grounding\_threshold} \quad (\text{default } 0.25)
$$

6. **Pipeline pass/fail**:

$$
\text{passed} = \text{hallucination\_risk} \leq \text{threshold}
$$

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| LLM output has no extractable claims (all questions / meta-phrases) | `faithfulness_score = 1.0`, `hallucination_risk = 0.0`, `passed = True` |
| Empty output or empty reference | Raises `EmptyOutputError` / `EmptyReferenceError` |
| Single-word claims | Filtered by `min_claim_length` (default 15 chars) |
| Identical output and reference | Score approaches 1.0 (depends on IDF context) |
| Completely unrelated output | Score approaches 0.0 |

---

## Additional Overlap Metrics

The `sf_hallucinate.scoring.overlap` module also exposes `bigram_f1` and `jaccard` for custom scorers or experiments:

**Bigram F1** uses consecutive word pairs:

$$
\text{bigram\_f1}(c, r) = F_1(\text{bigrams}(c), \text{bigrams}(r))
$$

**Jaccard similarity** on token sets:

$$
\text{jaccard}(c, r) = \frac{|P \cap R|}{|P \cup R|}
$$
---

## Pluggable Backends (v1.1.0)

The hybrid formula above is the **default** backend. Two additional backends are available:

### Embedding Backend

Uses `sentence-transformers` to encode claim and reference sentences as dense vectors, then computes cosine similarity:

$$
\\text{embedding}(c, r) = \\frac{\\mathbf{e}_c \\cdot \\mathbf{e}_r}{\\|\\mathbf{e}_c\\| \\cdot \\|\\mathbf{e}_r\\|}
$$

where $\\mathbf{e}_c$ and $\\mathbf{e}_r$ are the sentence embeddings. The model is configurable via `EvalConfig.embedding_model` (default `"all-MiniLM-L6-v2"`).

### LLM-NLI Backend

Uses an LLM as a Natural Language Inference judge. The model receives each claim alongside all reference sentences and returns a label:

| Label | Similarity mapping |
|---|---|
| `entailment` | $0.92 + 0.08 \\times \\text{confidence}$ |
| `contradiction` | $0.05 \\times (1 - \\text{confidence})$ |
| `neutral` | $0.35 + 0.30 \\times \\text{confidence}$ |

---

## Contradiction Detection (v1.1.0)

Three heuristic signals detect when a claim contradicts the reference:

### Negation Asymmetry

Detects cases where one sentence contains negation words (`not`, `no`, `never`, `n't`, etc.) and the other does not. Requires >20% vocabulary overlap to avoid false positives on unrelated sentences.

$$
\\text{confidence}_{\\text{negation}} = 0.8
$$

### Antonym Pairs

Checks 35 common antonym pairs (e.g. `true/false`, `increase/decrease`, `large/small`). If one term appears in the claim and its antonym in the reference:

$$
\\text{confidence}_{\\text{antonym}} = 0.6
$$

### Numeric Discrepancy

Extracts numeric values from both sentences. If different numbers appear in sentences with shared context:

$$
\\text{confidence}_{\\text{numeric}} = 0.7
$$

When a contradiction is detected with confidence ≥ 0.6, the claim's similarity is penalised:

$$
s_i = \\min(s_i,\\; 0.15)
$$

---

## Confidence Calibration (v1.1.0)

Each `ScorerResult` includes a calibrated confidence computed from three signals:

$$
\\text{confidence} = 0.3 \\times \\text{count\\_signal} + 0.3 \\times \\text{consistency} + 0.4 \\times \\text{backend\\_conf}
$$

where:

- **count\_signal** = $\\min(1.0,\\; N / 5)$ — saturates at 5 claims (more claims = more reliable estimate)
- **consistency** = $1.0 - \\text{std}(s_1, s_2, \\dots, s_N)$ — inverse standard deviation of per-claim scores
- **backend\_conf** = mean confidence from the similarity backend (1.0 for hybrid, backend-reported for LLM-NLI)

---

## Multi-Language Scoring (v1.1.0)

For non-English languages, the hybrid backend uses language-aware tokenization:

- **Latin-script languages** (es, fr, de, pt): Unicode NFC normalisation, lowercase, punctuation removal, language-specific stop word lists.
- **CJK languages** (zh, ja, ko): Character bigram tokenization — each pair of adjacent characters forms a token. No word segmentation required.
- **Arabic / Russian**: Unicode-aware tokenization with language-specific stop word lists.

The TF-IDF cosine and token F1 formulas remain the same; only the tokenization step differs.