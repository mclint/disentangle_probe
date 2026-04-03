# Methodology: Lookup-Dominant Transformer with TurboQuant-Enhanced Residual Compression

## 1. Overview

We investigate whether transformer hidden states can be factorized into:

$$
h_{l,t} \approx P_l[x_t] + r_{l,t}
$$

where:
- \(P_l[x_t]\): token-specific **prototype (context-free anchor)**
- \(r_{l,t}\): **contextual residual**

We further test whether:
1. A large fraction of tokens can be represented using **lookup-based symbolic states**
2. Residuals can be compressed efficiently using **TurboQuant-style quantization**
3. Computation can be performed partially or fully in **compressed space**

This leads to a **lookup-dominant architecture with quantized dense fallback**.

---

## 2. Prototype Bank Construction

For each layer \(l\), construct a vocabulary prototype bank:

$$
P_l[v] \in \mathbb{R}^d \quad \forall v \in \mathcal{V}
$$

### Procedure

For each token \(v\):
1. Input minimal sequence (e.g., `[BOS, v]`)
2. Run teacher forward pass
3. Extract hidden state at layer \(l\)
4. Store as \(P_l[v]\)

This captures **context-free latent anchors**.

---

## 3. Residual Extraction

For dataset \(D\), collect contextual hidden states:

$$
h_{l,t} = h_l(x_t \mid x_{<t})
$$

Compute residual:

$$
r_{l,t} = h_{l,t} - P_l[x_t]
$$

Residual captures **contextual deformation of token identity**.

---

## 4. TurboQuant Preconditioning (Optional / Core Variant)

We explore applying TurboQuant as a **geometry regularizer**:

### Variant A (Preferred)
$$
r_{l,t} \rightarrow \tilde r_{l,t} = \text{TurboQuant}(r_{l,t})
$$

### Variant B
$$
h_{l,t} \rightarrow \tilde h_{l,t} = \text{TurboQuant}(h_{l,t})
$$

Then operate in transformed space.

**Hypothesis:**
TurboQuant rotation + quantization produces:
- more isotropic residual distributions
- lower entropy residuals
- easier codebook fitting

---

## 5. Residual Compression Methods

### 5.1 Codebook (Lookup Residuals)

Cluster residuals:

$$
R_l = \{c_1, \dots, c_K\}
$$

Approximation:

$$
\hat h_{l,t} = P_l[x_t] + R_l[z_{l,t}]
$$

---

### 5.2 Top-k Mixture

$$
\hat h_{l,t} = P_l[x_t] + \sum_{i=1}^{k} \alpha_i R_l[z_i]
$$

Improves smoothness and reduces quantization error.

---

### 5.3 Low-Rank Residual

$$
r_{l,t} \approx U_l s_{l,t}
$$

$$
\hat h_{l,t} = P_l[x_t] + U_l s_{l,t}
$$

---

### 5.4 Hybrid (Recommended)

$$
\hat h_{l,t} = P_l[x_t] + R_l[z_{l,t}] + U_l s_{l,t}
$$

- codebook handles coarse structure
- low-rank captures fine detail

---

## 6. Compressed Representation

Each token is represented as:

```text
(token_id, code_id, optional low-rank coefficients)
```

No full dense vector required.

---

## 7. Router (Learned Assignment)

Predict residual code:

$$
z_{l,t} = \text{Router}_l(x_{\le t})
$$

Inputs:
- token embedding
- local context summary
- previous compressed state

Output:
- code index or mixture weights

---

## 8. Lookup-Dominant Computation

Instead of dense vectors:

$$
h_t = P[x_t] + R[z_t]
$$

Attention dot products become:

$$
(P[x_t] + R[z_t]) \cdot (P[x_s] + R[z_s])
$$

Expanded into lookupable components:
- prototype–prototype
- prototype–residual
- residual–residual

Enables **approximate attention without full vector reconstruction**.

---

## 9. Hybrid Execution Model

### Path A: Lookup Path (Cheap Tokens)
- stay in compressed symbolic form
- approximate attention via lookup decomposition
- no dense materialization

### Path B: Dense Path (Hard Tokens)
- reconstruct full hidden state
- compute exact attention
- compress K/V using TurboQuant

---

## 10. Error Control Mechanisms

### 10.1 Blockwise Compression
- compress every K layers, not every layer

### 10.2 Residual Carry
Maintain:
$$
h = P[x] + R[z] + U s
$$

### 10.3 Refresh Layers
- periodic dense reconstruction
- prevents drift accumulation

### 10.4 Confidence-Based Expansion
Expand token if:
- high residual norm
- router uncertainty high
- attention sensitivity high

---

## 11. Single-Layer Substitution Test

1. Replace \(h_{l,t}\) with \(\hat h_{l,t}\)
2. Run layers \(l+1 \dots L\)
3. Measure:
   - perplexity
   - token agreement

---

## 12. Evaluation Metrics

### Hidden-State Fidelity
- cosine similarity
- MSE
- explained variance

### Functional Metrics
- top-1 / top-k agreement
- perplexity

### Coverage
$$
\text{fraction of tokens approximated within threshold}
$$

### Memory
- bytes/token
- dense vs compressed ratio

### Efficiency
- FLOPs reduction
- bandwidth reduction

---

## 13. Experimental Setup

- Teacher: Qwen2.5 (1.5B–3B)
- Dataset: WikiText-2 -> long-context corpora
- Layers: early, mid, late
- Codebook sizes: \(K = 8, 32, 128, 512\)
- Low-rank dims: \(r = 4, 8, 16, 32\)

---

## 14. Key Hypotheses

1. Hidden states lie near **token-centered manifolds**
2. Residuals are **low-complexity and compressible**
3. TurboQuant improves residual structure
4. Majority of tokens can use **lookup-based computation**
5. Dense computation is only required for a minority of tokens

---

## 15. Expected Outcomes

- Early layers: high prototype explainability
- Middle layers: small residual codebooks sufficient
- Late layers: higher residual complexity
- TurboQuant reduces residual entropy
- Hybrid system achieves:
  - reduced memory footprint
  - reduced bandwidth
  - modest speedups

---

## 16. Core Insight

> Transformer hidden states can be decomposed into token identity + small contextual deviation, enabling a system where most computation is replaced by lookup and only difficult tokens require dense processing.

---

## 17. Research Direction

This work explores a new paradigm:

> **From dense vector computation -> structured symbolic + lookup computation with selective dense fallback**

---

## 18. Next Steps

- implement prototype + residual POC
- test TurboQuant on residuals
- build router
- test hybrid attention
- evaluate multi-layer stability
