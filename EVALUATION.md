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
