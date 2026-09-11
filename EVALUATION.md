# ThaiMouth Evaluation Suite

This document describes how to benchmark and compare ThaiMouth models across recurrent depth spectra.

---

## 1. Evaluation Dimensions

1. **Language Modeling Perplexity ($PPL$):**
   $$PPL = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i})\right)$$
   Measures fluency and next-token predictive certainty on held-out Thai text.

2. **Reasoning Exact Match Accuracy (%):**
   Evaluates multi-hop transitive questions ($1 \to 2 \to 3 \to 4$ hops) generated from out-of-distribution entity names.

3. **Inference Compute & Efficiency:**
   - Latency per forward pass (ms)
   - Milliseconds per generated token (ms/token)
   - Throughput (tokens/sec)
   - Memory / Parameter footprint (MB / Param Count)

---

## 2. Running Evaluation

### 2.1 Single Checkpoint Multi-Depth Evaluation
```bash
./scripts/evaluate.sh checkpoints/smoke/model_final.pt checkpoints/tokenizer_smoke.json data/synthetic/test.jsonl
```

### 2.2 Cross-Model Comparison Across Recurrent Depths (1, 2, 4, 8, 16)
```bash
./scripts/compare_models.sh checkpoints/baseline_1m/model_final.pt checkpoints/latent_1m/model_final.pt
```

Outputs summary reports in `results/`:
- `results/comparison_summary.csv`
- `results/comparison_summary.json`
- `results/comparison_summary.md`

## 3. Milestone 0.3.1 Locked Protocol

Milestone 0.3.1 freezes Variant B and tests whether additional recurrent
inference compute improves capability rather than only numerical stability.
The pre-registered protocol is in `configs/milestone_0_3_1.yaml`.

Prepare the five-seed Variant B, parameter-matched baseline, and fixed-depth
compute controls:

```bash
PYTHONPATH=src python scripts/prepare_milestone_0_3_1.py
```

Train missing checkpoints and run the complete benchmark:

```bash
PYTHONPATH=src python scripts/run_milestone_0_3_1.py --train-missing
```

The compute control for Variant B at recurrent depth `R` uses the same width
and `input_layers + R` independent Transformer blocks. FLOPs count one
multiply and one add as two operations and include attention projections,
attention score/value products, MLP projections, and the LM head. Embedding
lookups, normalization, activations, masking, and softmax are omitted for all
models. The configured maximum compute gap is 5%.

All five training seeds use the same pre-registered benchmark seed so model
variance is not mixed with test-set sampling variance. Fixed-depth controls use
the same data and optimizer-step budget; this gives them more training FLOPs
as depth increases and makes the comparison conservative for Variant B.

The run emits raw generated completions, per-hop and difficulty-stratified
accuracy, PPL, measured wall-clock latency, estimated FLOPs, latent norms,
relative latent updates, five alpha-by-R heatmaps, and a machine-readable exit
assessment. `primary_alpha` is registered before evaluation; the other alpha
values are explanatory ablations and cannot be selected after seeing results.
