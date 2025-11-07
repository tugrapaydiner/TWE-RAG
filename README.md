# TWE-RAG: Time-Weighted Evidence Retrieval-Augmented Generation

CPU-only retrieval system integrating temporal weighting, document corroboration via evidence graphs, hybrid retrieval scoring, and adaptive halting mechanisms. Addresses fundamental limitations in traditional RAG architectures through multi-dimensional document ranking with particular emphasis on temporal query resolution.

---

## 1. Problem Formulation

Traditional RAG systems suffer from three critical limitations:

**Time-blindness**: Documents are ranked independently of temporal context. A query "Who is the current CEO?" receives identical treatment of a 2019 document stating "Alice is CEO" and a 2024 document stating "Bob is CEO", despite the clear preference for recent information.

**Isolated document scoring**: Relevance scores are computed in isolation, without considering document corroboration. Multiple documents mentioning consistent facts should reinforce confidence in those facts.

**Fixed computational budget**: Retrieval always returns exactly $K$ documents regardless of confidence. Queries with high-confidence answers consume unnecessary computation.

<img width="3564" height="1764" alt="pro_fig03_decay" src="https://github.com/user-attachments/assets/e13a8a05-6469-4499-a042-b1df9ed2967a" />

---

## 2. Mathematical Framework 

### 2.1 Hybrid Retrieval Score

Final scoring combines sparse and dense retrieval modalities. Sparse retrieval via BM25Okapi:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

where 

$$
\text{IDF}(t) = \log\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right), \quad k_1 = 1.5, \quad b = 0.75
$$

Normalized to $[0,1]$ via max-pooling: 

$$
\text{BM25}_{\mathrm{norm}}(q,d) = \frac{\text{BM25}(q,d)}{\max_j \text{BM25}(q,d_j)}
$$

Dense retrieval via truncated SVD applied to TF-IDF matrix. Given corpus represented as TF-IDF matrix $\mathbf{X} \in \mathbb{R}^{N \times V}$:

$$
\mathbf{X} \approx \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$

Document embeddings (truncation at $d=128$):

$$
\mathbf{d}_i = \mathbf{U}_{i,1:d} \mathbf{\Sigma}_{1:d,1:d} \in \mathbb{R}^{d}
$$

Query embedding: 

$$
\mathbf{q} = \text{TF-IDF}(q) \cdot \mathbf{V}_{:,1:d}^T
$$

Cosine similarity normalized to $[0,1]$:

$$
\text{Dense}_{\mathrm{norm}}(q,d_i) = \frac{1}{2}\left(1 + \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \|\mathbf{d}_i\|}\right)
$$

Hybrid score (default $\alpha=\beta=1.0$):

$$
s_{\text{hybrid}}(q, d_i) = \alpha \cdot \text{BM25}_{\mathrm{norm}}(q, d_i) + \beta \cdot \text{Dense}_{\mathrm{norm}}(q, d_i)
$$

**Rationale for SVD over neural embeddings**: Computational determinism, reproducibility, CPU-compatibility, and empirical equivalence to BERT embeddings on mid-sized corpora (1K-100K documents).

<img width="4160" height="1446" alt="pro_fig06_scaling" src="https://github.com/user-attachments/assets/498066d4-5dc8-4e3b-aa6f-12679865dd0f" />

---

### 2.2 Evidence Graph Construction & Centrality Computation

Document corroboration captured through weighted evidence graph $G = (V, E, W)$.

**Graph construction via Jaccard similarity**

**Step 1** - Extract 3-grams (shingles) from tokenized text  
The shingle set is

$$
\mathcal{S}(d) = \{\mathrm{word}_i, \mathrm{word}_{i+1}, \mathrm{word}_{i+2}\}
$$

**Step 2** - Compute pairwise Jaccard similarity

$$
J(d_i, d_j) = \frac{|\mathcal{S}(d_i) \cap \mathcal{S}(d_j)|}{|\mathcal{S}(d_i) \cup \mathcal{S}(d_j)|}
$$

**Step 3** - Create edges for similarity threshold (default $\theta_{\text{edge}} = 0.05$)

**Graph components**  
Nodes: $V = \{d_1, d_2, \ldots, d_N\}$  
Edges: $(d_i, d_j) \in E$ if $J(d_i, d_j) > \theta_{\text{edge}}$  
Weights: $w_{ij} = J(d_i, d_j)$

**Degree centrality** (optimized variant)

$$
c_{\text{degree}}(d_i) = \frac{\sum_{j \neq i} w_{ij}}{\max_{k} \sum_{j \neq k} w_{kj}}
$$

Normalized to $[0,1]$ via max-pooling across corpus. Interpretation: weighted in-degree normalized to corpus maximum.

**Alternative PageRank centrality**

Higher computational cost $O(N^2 \times \text{iterations})$

$$
\text{PR}(d_i) = \frac{1-\lambda}{N} + \lambda \sum_{d_j \in \mathcal{N}(d_i)} \frac{w_{ji} \cdot \text{PR}(d_j)}{\sum_k w_{jk}}
$$

with damping factor $\lambda = 0.85$. Empirically performs similarly to degree centrality for corpora $N < 50K$.

**Justification**: High-centrality documents are "hubs" mentioning multiple shared facts, suggesting reliability. Jaccard similarity captures semantic overlap rather than lexical similarity alone.

<img width="3564" height="1769" alt="pro_fig04_scores" src="https://github.com/user-attachments/assets/9befa984-e112-439d-9e78-e3a76022739e" />

---

### 2.3 Adaptive Temporal Weighting 🕐

Time-aware scoring adapted to query temporal context.

**Step 1: Recency detection** via pattern matching

$$
r(q) = \begin{cases} 1.0 & \text{if } q \text{ matches } \{\text{current}, \text{latest}, \text{now}, \text{recent}, \ldots\} \\ 0.3 & \text{otherwise} \end{cases}
$$

**Step 2: Compute temporal parameters**

Delta amplitude

$$
\delta(q) = \delta_{\text{base}} \cdot r(q)
$$

Tau time constant

$$
\tau(q) = \tau_{\text{max}} - (\tau_{\text{max}} - \tau_{\text{min}}) \cdot r(q)
$$

Default hyperparameters: $\delta_{\text{base}} = 2.5$, $\tau_{\text{min}} = 90$ days, $\tau_{\text{max}} = 730$ days.

**Temporal dynamics**

- Recency query ($r(q)=1$) gives $\delta=2.5$, $\tau=90$
- General query ($r(q)=0.3$) gives $\delta=0.75$, $\tau=538$

**Step 3: Exponential decay function**

For document $d_i$ with timestamp $t_i$, referenced at time $t_{\text{now}}$

$$
\text{decay}(d_i, t_{\text{now}}) = \exp\left(-\frac{t_{\text{now}} - t_i}{\tau(q)}\right)
$$

Age measured in days. Decay properties

- At age $\tau$ the value is $e^{-1} \approx 0.368$
- At age $2\tau$ the value is $e^{-2} \approx 0.135$
- At age $3\tau$ the value is $e^{-3} \approx 0.050$

**Step 4: Temporal boost**

$$
\text{boost}_{\text{time}}(d_i, q) = \delta(q) \cdot \text{decay}(d_i, t_{\text{now}})
$$

Boost range is $[0, \delta_{\text{base}}] \approx [0, 2.5]$ for default parameters.

**Theoretical grounding**: Information utility for temporal queries decays exponentially. Parameter $\tau$ reflects the characteristic timescale over which information becomes obsolete (90 days for "current" queries, 2 years for general queries).

---

### 2.4 Unified Ranking Score

Final document ranking score combines all components:

$$
\boxed{S(q, d_i) = \alpha \cdot s_{\text{BM25}}(q,d_i) + \beta \cdot s_{\text{Dense}}(q,d_i) + \gamma \cdot c_{\text{degree}}(d_i) + \delta(q) \cdot \text{decay}(d_i, t_{\text{now}})}
$$

Default weights: $\alpha=1.0$, $\beta=1.0$, $\gamma=0.5$.

Score range: $[0, \sim 5.5]$ under typical hyperparameter settings.

---

### 2.5 Confidence-Based Early Halting

Adaptive retrieval budget based on result confidence.

**Margin statistic**

$$
m_k = S(d_1) - S(d_k)
$$

where $d_1$ is top-ranked document, $d_k$ is $k$-th ranked document.

**Agreement statistic**

$$
\text{agree}(k) = \frac{1}{k} \sum_{j=1}^{k} \mathbb{1}[d_j \text{ corroborates } d_1]
$$

measured via Jaccard similarity to top document.

**Halting criterion**

$$
\text{halt} = (m_k &gt; \theta_m) \wedge (\text{agree}(k) &gt; \theta_a)
$$

Default thresholds: $\theta_m = 0.5$, $\theta_a = 0.8$.

**Computation savings**: Marginal computation reduction $\sim 42\%$ for typical corpora with negligible accuracy degradation ($\Delta \text{Recall@5} \approx 0.01$).

<img width="3564" height="1764" alt="pro_fig05_halting" src="https://github.com/user-attachments/assets/4ef747db-95bb-42b6-b4fc-2cf9eed54903" />

---

## 3. Implementation Characteristics

### 3.1 Sparse Retrieval

BM25 computation via rank-bm25 library. Tokenization: whitespace-split with lowercase conversion. IDF capping prevents score outliers from dominating.

### 3.2 Dense Retrieval

SVD applied via scikit-learn's TruncatedSVD with $d=128$ components. TF-IDF vectorization: `max_features=100K`, `min_df=2`, `max_df=0.9`.

### 3.3 Graph Construction

Jaccard computation via set operations on 3-gram shingles. Graph stored as sparse adjacency matrix (COO format for memory efficiency). Centrality computation via weighted degree summation.

### 3.4 Temporal Weighting

Document timestamps stored as ISO 8601 strings, parsed to Unix timestamps. Decay computation vectorized across batch of documents using NumPy broadcasting.

---

## 4. Experimental Results

### 4.1 Hybrid Retrieval Analysis

Comparative performance on 5K-document corpus:

| Method | Recall@5 | MRR | Latency (ms) |
|--------|----------|-----|------|
| BM25 only | 0.68 | 0.52 | 40 |
| Dense (SVD) | 0.71 | 0.55 | 55 |
| **Hybrid** | **0.81** | **0.64** | 95 |

Hybrid retrieval improves Recall@5 by 13 percentage points over best single-modality approach. Latency trade-off justified by quality improvement.

### 4.2 Centrality Weight Study

Impact of centrality weight parameter $\gamma$:

| γ | Recall@5 | MRR |
|---|----------|-----|
| 0.0 | 0.81 | 0.64 |
| 0.3 | 0.83 | 0.66 |
| **0.5** | **0.85** | **0.68** |
| 1.0 | 0.82 | 0.65 |
| 2.0 | 0.77 | 0.59 |

Optimal $\gamma=0.5$ balances document-specific relevance against corroboration. Higher values over-rank generic hub documents.

### 4.3 Temporal Decay Parameter Optimization

500 temporal queries ("current CEO", "latest product", etc.):

| base_delta | min_tau (days) | Temporal Acc | General Acc |
|------------|---|---|---|
| 0.0 | — | 0.42 | 0.85 |
| 2.5 | 90 | 0.68 | 0.83 |
| **2.5** | **365** | **0.89** | **0.82** |
| 5.0 | 365 | 0.91 | 0.78 |

Optimal configuration: $\delta_{\text{base}}=2.5$, $\tau_{\text{min}}=365$ days. Achieves 0.89 accuracy on temporal queries with minimal degradation on general queries (0.82).

### 4.4 Compute Budget Analysis

Budgeted halting performance across corpus size:

| Halting | Avg K Retrieved | Compute Saved | Recall@5 | MRR |
|---------|---|---|---|---|
| Disabled (K=100) | 100 | 0% | 0.85 | 0.68 |
| **Enabled (30/60/100)** | **58** | **42%** | **0.84** | **0.67** |

Average document budget reduction from 100 to 58 with only 1% recall degradation. Statistically significant savings for large-scale deployments.

### 4.5 Parameter Sensitivity Analysis

<img width="4164" height="1459" alt="pro_fig08_sensitivity" src="https://github.com/user-attachments/assets/521360ef-5ad2-4148-8642-196c64950943" />

**Delta ($\delta$) sensitivity**: Temporal accuracy increases monotonically with $\delta \in [0,5]$. Peak at $\delta=2.5$ balances temporal preference against general query performance.

**Tau ($\tau$) sensitivity**: Optimal performance at $\tau=365$ days for one-year-old recent documents. Narrow ridge indicates sensitive parameter tuning required.

**Centrality weight ($\gamma$)**: Unimodal distribution, maximum at $\gamma=0.5$. Sharp degradation for $\gamma &gt; 1.0$ as generic documents over-weighted.

---

## 5. Complexity Analysis

**Graph construction**  
$O(N^2)$ Jaccard computation on tokenized documents. $O(N)$ centrality aggregation.

**Ranking**  
$O(N \log K)$ via heap for top-K selection.

**Temporal decay**  
$O(N)$ vectorized exponential computation.

**Query processing**  
$O(|V| + E)$ graph traversal for centrality lookups (pre-computed).

**Space complexity**  
$O(N \times d)$ for SVD embeddings, $O(N)$ for timestamps, $O(E)$ for graph edges.

Typical corpus (N=5K): Graph construction ≈ 2-5 minutes, query latency ≈ 100-200ms per query.

---

## 6. Ablation Studies

<img width="4170" height="2966" alt="pro_fig07_ablation" src="https://github.com/user-attachments/assets/103b0974-4bc4-4c86-a715-a5b17b201dc6" />

**Study 1: Retrieval method comparison**

- Hybrid improves over BM25-only by 13% Recall@5
- Dense-only outperforms BM25-only by 3%
- Complementarity: low correlation between methods suggests different error modes

**Study 2: Centrality weighting**

- Optimal $\gamma=0.5$ validates balanced weighting philosophy
- Over-weighting centrality ($\gamma &gt; 1.0$) hurts unique documents

**Study 3: Temporal decay impact**

- Time decay critical for temporal queries (+0.47 accuracy vs. no decay)
- Minimal impact on general queries (negligible degradation)

**Study 4: Budgeted halting**

- 42% compute reduction with 1% accuracy loss acceptable trade-off
- Margin-based halting effective proxy for confidence

---

## 7. Limitations & Future Work

**Current limitations**:
- Recency detection via regex patterns limited to English
- Jaccard similarity insensitive to semantic relationships (addresses "lexical" overlap only)
- Graph construction $O(N^2)$ prohibits very large corpora (N &gt; 1M)
- SVD embeddings less expressive than neural embeddings for semantic distinctions

**Enhancement directions**:

1. **Continuous temporal classification**: Replace binary recency detection with learned regression mapping queries to $r(q) \in [0,1]$.

2. **Multi-hop reasoning**: Iterate over evidence graph, following Jaccard edges to find connected facts.

3. **Approximate graph construction**: MinHash LSH reduces Jaccard computation to $O(N \log N)$ at cost of approximation error.

4. **Neural fallback mechanism**: Route low-confidence (margin &lt; 0.5) queries to BERT reranker for expensive verification.

5. **Dynamic parameter tuning**: Learn $\alpha, \beta, \gamma, \delta_{\text{base}}, \tau_{\text{min}}$ from labeled validation set via Bayesian optimization.
---
