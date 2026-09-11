# ThaiMouth Research & Experimental Hypotheses

## 1. Research Question

> **"How small can a Thai-native conversational language model be if we trade parameter count for recurrent latent compute?"**

Modern Large Language Models (LLMs) scale reasoning by expanding physical parameter size (e.g., 7B $\to$ 70B parameters) or generating long textual chain-of-thought tokens. 
ThaiMouth investigates an alternative: **recurrent depth within shared latent hidden representations** before emitting tokens.

---

## 2. Core Hypotheses

### 2.1 Primary Hypothesis ($H_1$)
Small parameter models (e.g., 1M – 30M parameters) can acquire multi-hop transitive reasoning capabilities by recycling a shared `ThinkBlock` across multiple recurrent steps $R$ ($1 \to 2 \to 4 \to 8$), achieving reasoning accuracy comparable to deeper fixed-weight models at equivalent parameter budgets.

### 2.2 Null Hypothesis ($H_0$)
Increasing inference-time recurrent depth on fixed shared parameters does not meaningfully improve reasoning accuracy, or improvements only occur on step counts explicitly witnessed during training without general depth extrapolation.

### 2.3 Secondary Hypotheses
1. **Recurrent Depth Curriculum Hypothesis ($H_{2a}$):** Training models with a stochastic step curriculum (e.g. randomly sampling $R \in \{1, 2, 4\}$ during training) prevents representation explosion and allows smooth test-time depth scaling up to $R=8$ or $16$.
2. **Thai Specialization Efficiency ($H_{2b}$):** Training Byte-Level BPE tokenizers specifically on Thai morphology preserves token-per-character efficiency and avoids multi-token fracturing typical of multilingual tokenizers, yielding higher quality per parameter.
3. **Latent Saturation / Degradation Boundary ($H_{2c}$):** Beyond an optimal recurrent depth $R^*$, latent hidden states may experience representation collapse or drift unless stabilized by residual damping $\alpha \le 1.0$ or layer normalization.

---

## 3. Established Facts vs. Hypotheses

| Statement | Status | Evidence / Notes |
|---|---|---|
| Causal Transformers process sequences through layered hidden state transformations. | **Established Fact** | Standard Vaswani et al. (2017) transformer architecture. |
| Reusing a `ThinkBlock` in a loop does not increase model parameter count. | **Established Fact** | Proven by unit tests and PyTorch weight reference verification. |
| Increasing recurrent steps linearly increases inference time / FLOPs. | **Established Fact** | Measured empirically in `thaimouth.evaluation.efficiency`. |
| Recurrent latent compute improves multi-hop reasoning over a fixed baseline in Thai conversational domains. | **Active Hypothesis** | Under experimental measurement in ThaiMouth benchmark suites. |
| Recurrent curriculum prevents out-of-distribution step degradation. | **Active Hypothesis** | Under validation via `compare_models.sh`. |

---

## 4. Key Metrics for Falsification

To test $H_1$ vs $H_0$, experiments must log:
1. **Perplexity ($PPL$) vs Steps ($R$):** Does validation loss decrease monotonically as $R$ increases from 1 to 8?
2. **Multi-Hop Reasoning Exact Match:** Does transitive accuracy ($A > B > C > D$) scale with $R$?
3. **Accuracy per Parameter ($\frac{\text{Acc}}{\text{Params}}$):** Does Recurrent Latent LM achieve superior performance per unit memory footprint compared to fixed-layer baselines?

## 5. Milestone 0.3.1 Exit Criteria

Variant B remains frozen during this milestone. A single checkpoint per seed
is evaluated at `R = 1, 2, 4, 8, 16, 32`; inference alpha is swept over
`0.1, 0.25, 0.5, 1.0`, with `0.5` pre-registered as the primary setting.

The milestone passes only when all configured gates pass:

1. Numerical stability: latent measurements remain finite, norm growth stays
   within the registered bound, and generated outputs do not collapse.
2. Adaptive computation: the same checkpoint improves from low to high `R`
   across at least four of five seeds and the mean curve is predominantly
   non-decreasing.
3. Useful computation: Variant B beats independently trained fixed-depth
   controls at matched inference FLOPs, both overall and on hard tasks.
4. Efficiency: accuracy gain per additional GFLOP exceeds the fixed-depth
   control curve.
5. Dynamics: relative latent updates decrease across recurrent iterations
   without beginning at an effectively dead fixed point.

Passing stability alone supports only a stability claim. Passing stability and
adaptive computation supports evidence for test-time recurrent computation.
The project advances to the 5M tier only after the compute-matched advantage is
also reproduced.
