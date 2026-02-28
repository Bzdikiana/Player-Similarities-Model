# ═══════════════════════════════════════════════════════════════════════════════
#                      MATHEMATICAL FOUNDATIONS
#              Player Similarity Model v2.0
# ═══════════════════════════════════════════════════════════════════════════════

## Table of Contents

1. [Notation](#1-notation)
2. [Contrastive Learning (InfoNCE)](#2-contrastive-learning-infonce)
3. [Graph Neural Networks](#3-graph-neural-networks)
4. [Temporal Transformer](#4-temporal-transformer)
5. [Similarity Computation](#5-similarity-computation)

---

## 1. Notation

### 1.1 General Symbols

| Symbol | Description |
|--------|-------------|
| $x, y$ | Pitch coordinates (yards, 0-120 × 0-80) |
| $n$ | Number of observations |
| $d$ | Dimensionality |
| $\tau$ | Temperature parameter |
| $\|\cdot\|$ | Euclidean norm |

### 1.2 Player & Event Notation

| Symbol | Description |
|--------|-------------|
| $p_i$ | Player $i$ |
| $e_j$ | Event $j$ |
| $z_i$ | Embedding for player $i$ (64-dim) |
| $s_k$ | Sequence $k$ of events |

### 1.3 Graph Notation

| Symbol | Description |
|--------|-------------|
| $G = (V, E)$ | Graph with vertices $V$, edges $E$ |
| $\mathbf{A}$ | Adjacency matrix |
| $h_v^{(l)}$ | Node $v$ embedding at layer $l$ |
| $\mathcal{N}(v)$ | Neighbors of node $v$ |
| $\alpha_{uv}$ | Attention weight from $u$ to $v$ |

---

## 2. Contrastive Learning (InfoNCE)

### 2.1 Core Idea

We want embeddings of the **same player** (from different sequences) to be similar, while embeddings of **different players** should be dissimilar.

### 2.2 InfoNCE Loss

For a batch of $N$ player embeddings where player $i$ has embedding $z_i$:

$$\mathcal{L}_{InfoNCE} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i, z_{i^+}) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}$$

Where:
- $z_{i^+}$ = another embedding of the same player (positive)
- $z_j$ = all other embeddings in batch (negatives)
- $\tau$ = temperature (we use 0.07)
- $\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$ (cosine similarity)

### 2.3 Temperature Effect

| Temperature $\tau$ | Effect |
|-------------------|--------|
| Low (0.01-0.05) | Sharp: strongly penalizes hard negatives |
| Medium (0.07-0.1) | Balanced (what we use) |
| High (0.5+) | Soft: all negatives treated similarly |

### 2.4 Why InfoNCE Works

The loss is minimized when:
1. Same-player pairs have high similarity → numerator large
2. Different-player pairs have low similarity → denominator small

This naturally learns a space where similar players cluster together.

---

## 3. Graph Neural Networks

### 3.1 Event as Graph

Each on-ball event becomes a graph:
- **Nodes**: On-ball player + visible teammates + visible opponents
- **Edges**: Connections between nearby players (within threshold distance)
- **Edge Features**: Distance, same_team flag, relative angle

### 3.2 Graph Attention (GAT)

For node $v$ at layer $l$:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u^{(l)}\right)$$

Where attention weights:

$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(a^T [Wh_v \| Wh_u \| e_{vu}]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(a^T [Wh_v \| Wh_k \| e_{vk}]))}$$

- $W$ = learnable weight matrix
- $a$ = attention vector
- $e_{vu}$ = edge features between $v$ and $u$
- $\|$ = concatenation

### 3.3 Multi-Head Attention

We use 4 attention heads, then concatenate:

$$h_v^{(l+1)} = \|_{k=1}^{4} \sigma\left(\sum_{u} \alpha_{vu}^{(k)} W^{(k)} h_u^{(l)}\right)$$

### 3.4 Graph Readout

After 3 GAT layers, we pool all nodes to get a single event embedding:

$$h_G = \text{MeanPool}(\{h_v^{(L)} : v \in V\})$$

---

## 4. Temporal Transformer

### 4.1 Input

A sequence of event embeddings for one player:

$$S = [h_{G_1}, h_{G_2}, ..., h_{G_T}]$$

Where $T$ = sequence length (typically 32 events).

### 4.2 Time2Vec Positional Encoding

Instead of fixed sinusoidal positions, we use learnable Time2Vec:

$$\text{Time2Vec}(t)[i] = \begin{cases}
\omega_i t + \phi_i & \text{if } i = 0 \\
\sin(\omega_i t + \phi_i) & \text{if } i > 0
\end{cases}$$

Where $\omega_i, \phi_i$ are learnable parameters.

**Why Time2Vec?**
- Captures periodic patterns (e.g., behavior in first vs second half)
- Learns game-time-specific representations

### 4.3 Transformer Encoder

Standard transformer encoder with:
- 2 layers
- 4 attention heads
- Hidden dim 128

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 4.4 Hierarchical Pooling

Instead of just taking the [CLS] token or mean pooling, we use hierarchical pooling:

1. **Segment pooling**: Divide sequence into segments, pool each
2. **Global pooling**: Pool segment representations

This captures both local (within-possession) and global (across-match) patterns.

---

## 5. Similarity Computation

### 5.1 Cosine Similarity

For two player embeddings $z_a, z_b$:

$$\text{sim}(z_a, z_b) = \frac{z_a \cdot z_b}{\|z_a\| \|z_b\|} = \cos(\theta)$$

Range: $[-1, 1]$ where 1 = identical, 0 = orthogonal, -1 = opposite.

### 5.2 Interpretation

| Similarity | Interpretation |
|------------|----------------|
| > 0.8 | Very similar playing style |
| 0.6 - 0.8 | Similar with some differences |
| 0.4 - 0.6 | Moderately similar |
| 0.2 - 0.4 | Different styles |
| < 0.2 | Very different players |

### 5.3 Similarity Gap Metric

We track the "similarity gap" during training:

$$\text{Gap} = \mathbb{E}[\text{sim}(z_i, z_{i^+})] - \mathbb{E}[\text{sim}(z_i, z_{j})]$$

Where $i^+$ = same player, $j$ = different player.

A higher gap means the model better discriminates between players.
- Our final gap: **0.624** (good discrimination)

---

## 6. Multi-Task Learning

### 6.1 Combined Loss

$$\mathcal{L}_{total} = \mathcal{L}_{InfoNCE} + \lambda \cdot \mathcal{L}_{position}$$

Where:
- $\mathcal{L}_{InfoNCE}$ = contrastive loss (main objective)
- $\mathcal{L}_{position}$ = cross-entropy position prediction
- $\lambda = 0.01$ (position loss weight)

### 6.2 Why Multi-Task?

Position prediction acts as regularization:
- Forces embeddings to encode positional information
- Prevents collapse to trivial solutions
- Adds supervised signal to guide learning

### 6.3 Planned Improvement

Current position prediction (14 classes) is too granular. We plan to:
1. Reduce to 7 position groups (see FUTURE_IMPROVEMENTS.md)
2. Add action distribution prediction as additional task

---

**Author:** Armen Bzdikian  
**Contact:** bzdikiana11@gmail.com  
**Last Updated:** February 2026
