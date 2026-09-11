# ThaiMouth Architecture Reference

This document explains the foundational theory and engineering implementation of **ThaiMouth**, focusing on how **recurrent latent computation** enables small models to trade inference time for depth without expanding parameter budgets.

---

## 1. Fundamentals of Causal Language Models

### 1.1 Tokens & Vocabulary
Computers do not process raw text strings directly. Text is sliced into atomic fragments called **Tokens** indexed by integers $0 \le t < V$, where $V$ is the vocabulary size.
In ThaiMouth, a Byte-Level Byte-Pair Encoding (BPE) tokenizer transforms Thai sentences (which lack inter-word whitespace) into token IDs without generating Out-Of-Vocabulary (OOV) errors.

### 1.2 Token Embeddings
The token embedding matrix $W_e \in \mathbb{R}^{V \times D}$ maps discrete token IDs into continuous vectors of dimension $D$ (`d_model`):
$$\mathbf{X} = \text{Embedding}(\mathbf{t}) \in \mathbb{R}^{B \times T \times D}$$
Where:
- $B$: Batch size
- $T$: Sequence length (number of context tokens)
- $D$: Hidden representation size

### 1.3 Hidden States
A **hidden state** $\mathbf{h}_t \in \mathbb{R}^D$ is the continuous vector representation of the sequence at position $t$ after passing through one or more layers. It stores semantic context, syntactic structure, and reasoning state.

---

## 2. Multi-Head Attention & Scaled Dot-Product

### 2.1 Query, Key, Value Projections
At each attention layer with $H$ heads and head dimension $D_h = D / H$:
- **Queries ($\mathbf{Q}$)**: What each token is looking for ($\mathbf{Q} = \mathbf{X} W_Q$).
- **Keys ($\mathbf{K}$)**: What information each token possesses ($\mathbf{K} = \mathbf{X} W_K$).
- **Values ($\mathbf{V}$)**: The actual content transmitted ($\mathbf{V} = \mathbf{X} W_V$).

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times T \times D_h}$$

### 2.2 Rotary Position Embedding (RoPE)
RoPE injects relative positional information directly into $\mathbf{Q}$ and $\mathbf{K}$ by rotating 2D sub-vectors in the complex plane:
$$\mathbf{q}_m = \mathbf{R}_m \mathbf{q}_m, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{k}_n$$
This preserves translational invariance and enables length extrapolation.

### 2.3 Causal Attention Mask
To preserve causality in autoregressive language generation (so token $t$ cannot look ahead at future tokens $t+1 \dots T$), we apply an upper-triangular mask $-\infty$ to the attention logits:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{D_h}} + \mathbf{M}_{\text{causal}}\right) \mathbf{V}$$

---

## 3. The Three Model Architectures in ThaiMouth

```
========================================================================================
1. BASELINE (Fixed Depth)      2. RECURRENT TRANSFORMER       3. RECURRENT LATENT LM
========================================================================================
Tokens -> Embedding            Tokens -> Embedding            Tokens -> Embedding (X)
         ↓                               ↓                              ↓
Transformer Block 1            Input Block(s)                 Input Encoder -> Z_0
         ↓                               ↓                              ↓
Transformer Block 2            ┌───► [Shared ThinkBlock]      ┌───► [Shared LatentBlock]
         ↓                     │            ↓                 │     Z_{t+1} = Z_t +
Transformer Block 3            └──── (Repeat R times)         └────  alpha * F(Z_t, X)
         ↓                               ↓                              ↓
Transformer Block N                     Z_R                            Z_R
         ↓                               ↓                              ↓
    Final Norm                       Final Norm                     Final Norm
         ↓                               ↓                              ↓
      LM Head                         LM Head                        LM Head
```

---

## 4. Parameter Sharing vs. Inference Compute

### 4.1 The Fundamental Distinction
In machine learning, it is vital to distinguish between **Learned Parameters** and **Inference FLOPs**:

| Concept | Definition | ThaiMouth Implementation |
|---|---|---|
| **Learned Weights ($\theta$)** | Physical floating-point numbers stored in memory/disk. | Shared across all recurrent loop iterations. |
| **Compute / FLOPs** | Mathematical multiply-accumulate operations executed during inference. | Scales linearly with recurrent steps $R$ ($\mathcal{O}(R \cdot T \cdot D^2)$). |
| **Latent State ($Z_t$)** | Ephemeral activation tensor updated step-by-step. | Allocated in VRAM during the forward pass. |

### 4.2 Why Parameter Count Stays Constant
In a standard $N$-layer transformer, layer $k$ has its own private weights $\theta_k$:
$$\text{Params}_{\text{Baseline}} = V \cdot D + N \cdot (4 D^2 + 8 D^2) = \mathcal{O}(N \cdot D^2)$$

In ThaiMouth's **Recurrent Latent LM**, the `ThinkBlock` has exactly **1 set of weights** $\theta_{\text{think}}$ executed $R$ times:
$$\text{Params}_{\text{Recurrent}} = V \cdot D + 1 \cdot (4 D^2 + 8 D^2) = \text{Constant with respect to } R!$$

Whether $R=1, 4, 8$, or $16$, the model file on disk is identically **1.2 MB (or 30M parameters)**, but the effective computational depth expands dynamically at runtime.

---

## 5. Mathematical Formulation of Recurrent Latent Updates

1. **Initial Embedding & Encoding:**
   $$X = \text{tok\_embeddings}(tokens)$$
   $$Z_0 = \text{InputEncoder}(X)$$

2. **Iterative Latent Thought Refinement ($t = 0 \dots R-1$):**
   $$\Delta Z_t = \text{ThinkBlock}(\text{RMSNorm}(Z_t), \text{Context}=X)$$
   $$Z_{t+1} = Z_t + \alpha \cdot \Delta Z_t$$

3. **Decoding Next-Token Probabilities:**
   $$\hat{y}_{t+1} = \text{softmax}\left(W_{\text{lm}} \cdot \text{RMSNorm}(Z_R)\right)$$
