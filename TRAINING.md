# ThaiMouth Training Guide

This guide covers tokenizer training, dataset preprocessing, full model training, curriculum scheduling, and troubleshooting.

---

## 1. Quickstart Commands

### 1.1 End-to-End Smoke Training (CPU / Apple Silicon / CUDA)
```bash
./scripts/train_smoke.sh
```

### 1.2 Training the Tokenizer
```bash
./scripts/train_tokenizer.sh 4096 checkpoints/tokenizer.json
```

### 1.3 Training 1M Parameter Models
```bash
# Baseline standard transformer
./scripts/train_baseline.sh configs/baseline_1m.yaml

# Recurrent transformer
./scripts/train_recurrent.sh configs/recurrent_1m.yaml

# Recurrent latent LM
./scripts/train_latent.sh configs/latent_1m.yaml
```

---

## 2. Recurrent Depth Curriculum

To train recurrent models stably across varying compute depths, ThaiMouth supports step curriculums in YAML configs:

```yaml
training:
  curriculum:
    - progress: 0.2    # First 20% of training
      steps: [1]       # Train only single-step
    - progress: 0.5    # 20% to 50%
      steps: [1, 2]    # Randomly sample between 1 or 2 steps
    - progress: 0.8    # 50% to 80%
      steps: [1, 2, 4] # Sample across 1, 2, or 4 steps
    - progress: 1.0    # Final 20%
      steps: [1, 2, 4, 8] # Allow up to 8 recurrent iterations
```

---

## 3. Hardware & Mixed Precision

- **Apple Silicon (MPS):** Supported natively.
- **CUDA (NVIDIA):** Uses PyTorch `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` and `GradScaler`.
- **CPU:** Automatic fallback with standard 32-bit floats.

---

## 4. Debugging & Stability Checklist

1. **NaN / Loss Divergence:**
   - Verify `grad_clip` is enabled (default `1.0`).
   - Check `latent_alpha` scaling factor (default `1.0`). If latents explode, reduce to `0.5` or `0.25`.
   - Ensure RMSNorm / LayerNorm epsilon is at least `1e-6`.
2. **Out of Memory (OOM):**
   - Reduce `batch_size` and increase `grad_accum_steps` proportionally (e.g. `batch_size: 16`, `grad_accum_steps: 4`).
3. **Parameter Verification:**
   - Parameter counts are printed at initialization. Ensure `ThinkBlock` parameters are not duplicated when `recurrent_steps > 1`.
