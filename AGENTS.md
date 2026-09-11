# Agent Guidelines for ThaiMouth

Instructions and rules for AI coding assistants modifying or extending the **ThaiMouth** codebase.

---

## 1. Architectural Guardrails

1. **Strict Parameter Sharing:**
   - When modifying `ThinkBlock` or `LatentThinkBlock`, ensure the exact same PyTorch `nn.Module` parameters are reused in the recurrent loop.
   - **Never create separate parameter layers inside the recurrent iteration loop.**
   - Always run `pytest tests/test_recurrent_weight_sharing.py` after touching model definitions.

2. **No Unnecessary External Dependencies:**
   - Maintain pure PyTorch core implementations for attention and transformer blocks.
   - Do NOT add heavy framework dependencies (e.g. Megatron, DeepSpeed, Ray, Lightning) without explicit user instruction.

3. **Empirical Honesty & Scientific Rigor:**
   - **Never fabricate benchmark scores, accuracies, or evaluation numbers.**
   - Clearly delineate between working empirical results and theoretical hypotheses.
   - All evaluation scripts must log measured wall-clock latencies and actual token outputs.

4. **Reproducibility:**
   - Keep random seed logic deterministic (`thaimouth.utils.seed.set_seed`).
   - Store exact YAML configs alongside saved model checkpoints (`config.yaml`).

5. **Test-Driven Architecture Extensions:**
   - If adding a new model variant (e.g. cross-attention latent memory), create corresponding unit tests for output tensor shapes, causality masks, and parameter count invariance.
