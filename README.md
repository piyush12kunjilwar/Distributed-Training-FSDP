# Distributed-Training-FSDP# Distributed Training with FSDP 🚀

> LLaMA-style transformer training with PyTorch FSDP,
> Mixed Precision, Gradient Accumulation and NCCL —
> proving every claim from production ML experience

## Hardware
- **GPU:** NVIDIA L4 (23.7GB, 58 SMs)
- **RAM:** 56.9GB High RAM runtime
- **CUDA:** 12.8 | **PyTorch:** 2.10

---

## Model — LLaMA-style Transformer (37.4M params)

Built from scratch with exact LLaMA design choices:

| Component | Choice | Why |
|-----------|--------|-----|
| Normalization | RMSNorm | Faster than LayerNorm |
| Position encoding | RoPE | Better long context |
| Activation | SwiGLU | Better than ReLU/GELU |
| Bias terms | None | Cleaner gradients |
| Weight tying | embed ↔ lm_head | Fewer parameters |
| Architecture | Pre-norm | More stable training |

---

## Results

### Memory Profiling — Where Memory Goes
