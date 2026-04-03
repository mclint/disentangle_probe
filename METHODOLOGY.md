# Methodology: Lookup-Dominant Approximation of Transformer Hidden States

## 1. Overview

We investigate whether transformer hidden states can be approximated as a **token-centered prototype plus a small contextual residual**, enabling a large fraction of computation to be replaced by **lookup operations**.

For a token \( x_t \) at layer \( l \), we approximate the teacher hidden state:

\[
h_{l,t} \approx P_l[x_t] + \Delta_{l,t}
\]

where:
- \( P_l[x_t] \): context-free prototype from a vocabulary bank  
- \( \Delta_{l,t} \): context-dependent residual

We evaluate whether \( \Delta_{l,t} \) can be modeled using:
- discrete residual codebooks (lookup-based)
- low-rank latent structure

---

## 2. Prototype Bank Construction

For each layer \( l \), we construct a vocabulary prototype bank:

\[
P_l[v] \in \mathbb{R}^d \quad \forall v \in \mathcal{V}
\]

### Procedure

For each token \( v \):
1. Construct minimal valid input (e.g., `[BOS, v]`)
2. Run the teacher model forward
3. Extract hidden state at layer \( l \) for the token position
4. Store as \( P_l[v] \)

This bank represents **context-free latent anchors**.

---

## 3. Contextual Hidden State Collection

Given a dataset \( D \), we run the teacher model and collect:

\[
h_{l,t} = h_l(x_t \mid x_{<t})
\]

For each token position:
- token id \( x_t \)
- hidden state \( h_{l,t} \)

We define the residual:

\[
r_{l,t} = h_{l,t} - P_l[x_t]
\]

This isolates **contextual deviation from token identity**.

---

## 4. Approximation Families

### 4.1 Prototype-Only (Lookup Baseline)

\[
\hat h_{l,t} = P_l[x_t]
\]

Tests how much of the representation is explained by token identity alone.

---

### 4.2 Prototype + Discrete Residual Codebook

Cluster residuals \( \{r_{l,t}\} \) into \( K \) clusters:

\[
R_l = \{c_1, \dots, c_K\}
\]

Approximation:

\[
\hat h_{l,t} = P_l[x_t] + R_l[z_{l,t}]
\]

where \( z_{l,t} \) is the nearest cluster index (oracle assignment).

---

### 4.3 Prototype + Low-Rank Residual

Fit a low-rank basis (e.g., PCA):

\[
r_{l,t} \approx U_l s_{l,t}, \quad U_l \in \mathbb{R}^{d \times r}
\]

Approximation:

\[
\hat h_{l,t} = P_l[x_t] + U_l s_{l,t}
\]

---

## 5. Evaluation Metrics

### 5.1 Hidden-State Fidelity

- Cosine similarity:
  \[
  \cos(h_{l,t}, \hat h_{l,t})
  \]

- Mean squared error (MSE)

- Explained variance:
  \[
  1 - \frac{\sum \|h_{l,t} - \hat h_{l,t}\|^2}{\sum \|h_{l,t} - \bar h_l\|^2}
  \]

---

### 5.2 Token Prediction Consistency

Replace hidden states at layer \( l \) with \( \hat h_{l,t} \), then run layers \( l+1 \dots L \).

Measure:
- top-1 agreement
- top-k agreement
- perplexity degradation

---

### 5.3 Coverage

Define error threshold \( \epsilon \).

\[
\text{Coverage}_l = \frac{\#\{t : \text{error}(h_{l,t}, \hat h_{l,t}) < \epsilon\}}{T}
\]

Measures fraction of tokens handled by lookup approximation.

---

### 5.4 Compression Efficiency

Compute storage cost per token:
- prototype-only: token ID
- codebook: token ID + code index
- low-rank: token ID + latent vector

Compare against full hidden state.

---

## 6. Layer-wise Analysis

Evaluate across layers:
- early
- middle
- late

Hypothesis:
- early layers → high prototype explainability
- middle layers → moderate residual complexity
- late layers → higher residual complexity

---

## 7. Single-Layer Substitution Test

1. Replace \( h_{l,t} \) with \( \hat h_{l,t} \)
2. Run layers \( l+1 \dots L \)
3. Measure perplexity and prediction accuracy

Tests viability of replacing a transformer layer with lookup approximation.

---

## 8. Learned Routing (Optional)

Train a lightweight router:

\[
z_{l,t} = \text{Router}_l(x_{\le t})
\]

Inputs:
- token embedding
- projected previous-layer hidden state
- local context summary

Output:
- residual code index or mixture weights

Final approximation:

\[
\hat h_{l,t} = P_l[x_t] + R_l[z_{l,t}]
\]

---

## 9. Experimental Setup

- Teacher model: e.g., Qwen2.5-1.5B
- Dataset: WikiText-2 (initial), extended later
- Layers: selected across depth
- Codebook sizes: \( K \in \{8, 32, 128, 512\} \)
- Low-rank dimensions: \( r \in \{4, 8, 16, 32\} \)

---

## 10. Hypothesis

A significant fraction of hidden states can be approximated by token prototypes plus small residuals, enabling lookup-dominant computation.

---

## 11. Expected Outcomes

- Early layers: high prototype-only coverage
- Middle layers: effective codebook approximation
- Late layers: increased residual complexity

This supports a **lookup-dominant transformer architecture with conditional dense computation**.