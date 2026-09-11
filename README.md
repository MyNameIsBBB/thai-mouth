# 🇹🇭 ThaiMouth: Recurrent Latent Language Models for Thai

**ThaiMouth** is a research framework investigating the limits of model size and compute efficiency for Thai-native conversational and reasoning language models.

---

## 🔬 Research Question

> **"How small can a Thai-native conversational model be if we trade parameter count for recurrent latent compute?"**

Instead of scaling physical parameter counts (which increases disk and memory footprint), ThaiMouth investigates recycling a shared `ThinkBlock` across multiple recurrent steps $R$ ($1 \to 2 \to 4 \to 8 \to 16$) in latent representation space $Z$ before decoding next-token distributions.

---

## 🏛️ Model Architectures

1. **`TinyTransformerBaseline`:** Standard decoder-only causal Transformer (fixed depth $N$).
2. **`RecurrentTransformer`:** Input embedding + non-recurrent input block + shared `ThinkBlock` executed $R$ times + LM Head.
3. **`RecurrentLatentLM`:** Input embedding $X \to$ Input Encoder $Z_0 \to$ Iterative refinement $Z_{t+1} = Z_t + \alpha \cdot F_\theta(Z_t, X) \to$ LM Head.

All recurrent architectures maintain **constant parameter counts** regardless of recurrent compute depth $R$.

---

## 🚀 Quickstart

### 1. Installation
```bash
cd thaimouth
pip install -r requirements.txt
pip install -e .
```

### 2. Run All Unit Tests
```bash
pytest tests
```

### 3. Run End-to-End Smoke Training Pipeline
Generates synthetic data, trains the tokenizer, trains a tiny recurrent-latent model, generates sample Thai text, and evaluates reasoning across depths:
```bash
./scripts/train_smoke.sh
```

### 4. Interactive Text Generation
```bash
python -m thaimouth.generation.generate \
  --checkpoint checkpoints/smoke/model_final.pt \
  --tokenizer checkpoints/tokenizer_smoke.json \
  --prompt "<s><user> วันนี้เหนื่อยมากเลย </user><assistant>" \
  --recurrent_steps 4
```

### 5. Multi-Depth Evaluation & Comparison
```bash
./scripts/compare_models.sh \
  checkpoints/smoke/model_final.pt
```

---

## 📂 Repository Structure

```
thaimouth/
├── README.md                # Project overview and quickstart
├── ARCHITECTURE.md          # In-depth architectural & mathematical reference
├── RESEARCH.md              # Research hypotheses and verification metrics
├── TRAINING.md              # Training workflows and curriculum scheduling
├── DATA.md                  # Data schemas and synthetic dataset generator
├── EVALUATION.md            # Benchmark and efficiency metrics
├── AGENTS.md                # Guidelines for future AI coding agents
├── pyproject.toml           # Package metadata and pytest configuration
├── requirements.txt         # Core dependencies (PyTorch, tokenizers, yaml, etc.)
├── configs/                 # Model and training YAML configurations
│   ├── smoke.yaml           # Tiny smoke config (~164K params)
│   ├── baseline_1m.yaml     # 1M parameter standard baseline
│   ├── recurrent_1m.yaml    # 1M parameter recurrent transformer
│   ├── latent_1m.yaml       # 1M parameter recurrent latent model
│   ├── baseline_30m.yaml    # 30M parameter baseline
│   └── latent_30m.yaml      # 30M parameter recurrent latent model
├── src/thaimouth/           # Core library
│   ├── config.py            # Configuration dataclasses
│   ├── tokenizer/           # Byte-level BPE tokenizer training and wrapper
│   ├── data/                # Dataset, collation, synthetic reasoning generator
│   ├── models/              # Pure PyTorch attention, blocks, and model variants
│   ├── training/            # Trainer, optimizer, scheduler, checkpointing
│   ├── generation/          # Autoregressive text generation
│   ├── evaluation/          # Reasoning, perplexity, and efficiency benchmarks
│   └── utils/               # Seed and device selection
├── scripts/                 # Shell execution scripts
├── tests/                   # Pytest test suite
├── data/                    # Raw, processed, and synthetic datasets
├── checkpoints/             # Saved model weights
└── results/                 # Evaluation outputs (CSV, JSON, Markdown)
```

---

## 📊 Empirical Findings & Preliminary Results

> [!NOTE]
> **Preliminary result:** Increasing recurrent depth from 1 to 16 at inference reduced held-out test perplexity from 31.82 to 13.79 for the same 647K-parameter checkpoint. Most gains occurred by 4–8 recurrent steps. Further controlled experiments are required to determine whether this represents genuine test-time reasoning scaling or alignment with the recurrent depth used during training.

### Test-Time Latency vs. Throughput Profile (Apple Silicon MPS)

| Recurrent Steps ($R$) | Perplexity | Forward Latency | Throughput | Trade-off Characteristic |
|---|---|---|---|---|
| **1 Step** | 31.82 | 0.82 ms | 155,206 tok/s | Baseline speed, initial latent representation |
| **2 Steps** | 18.76 | 1.28 ms | 100,168 tok/s | ~41% perplexity reduction |
| **4 Steps** | 14.67 | 1.99 ms | 64,473 tok/s | Major quality jump with sub-2ms latency |
| **8 Steps** | 13.84 | 3.11 ms | 41,142 tok/s | Diminishing returns curve begins |
| **16 Steps** | 13.79 | 5.34 ms | 23,949 tok/s | Latent representation convergence plateau |

*Throughput scales sub-linearly with $1/R$ because non-recurrent modules (embedding, input encoder, output LM head) execute only once per sequence.*

---

## 🔬 Latent Dynamics & Scientific Status

> *Output distributions converge with increasing recurrence, while latent-state norms diverge rapidly. This suggests output-space stabilization despite unstable latent magnitude dynamics. Further experiments are required to determine whether normalization or recurrent update design can produce stable iterative computation.*

---

## 🗺️ Roadmap & Milestones

- [x] **Milestone 0.1:** Core causal Transformer, recurrent, and latent architectures with parameter weight tying.
- [x] **Milestone 0.2:** Multi-hop reasoning depth benchmark (1, 2, 4, 8, 16-hops), multi-seed evaluation, and latent trajectory tracking.
- [ ] **Milestone 0.3:** Recurrent PreNorm / Gated residual stabilization ($Z_{t+1} = \text{RMSNorm}(Z_t + \alpha F(Z_t))$) to bound latent magnitude.
- [ ] **Milestone 0.4:** Scale to 5M parameter tier with compute-matched baseline comparisons.
- [ ] **Milestone 0.5:** Scale to 30M parameter conversational Thai model.

# thai-mouth
