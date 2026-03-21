# Pico-GPT Architecture

## Overview

Pico-GPT is a decoder-only transformer language model with approximately 49M parameters. It follows GPT architecture design principles with modern optimizations.

## Model Architecture

```
Input Token IDs (B, T)
        ↓
Token Embedding (50257 x 384)
Positional Embedding (128 x 384)
        ↓
Embedding Sum + Dropout
        ↓
┌─────────────────────────────────┐
│  Transformer Block × 6          │
│  ┌─────────────────────────┐    │
│  │ LN1 → Attention        │    │
│  │ (Fused QKV + Flash Attn)│    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │ LN2 → MLP (4x hidden)  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
        ↓
Final LayerNorm
        ↓
LM Head (384 x 50257) - separate from token embedding
        ↓
Logits (B, T, 50257)
```

> **Note:** The model uses separate embedding (`wte`) and output (`lm_head`) layers. Weight tying is not implemented, which increases parameter count but simplifies the architecture.

## Components

### Attention Module

- **Type:** Multi-head self-attention with causal masking
- **QKV Projection:** Fused single linear layer (3*C)
- **Flash Attention:** Enabled via PyTorch SDPA
- **Heads:** 6
- **Head Dim:** 64 (384 / 6)

### Transformer Block

- **Norm:** Pre-LayerNorm (before attention and MLP)
- **Residual Connections:** Both attention and MLP
- **MLP:** Linear → GELU → Linear (4x hidden dimension)

### Embeddings

- **Token Embedding:** Learned (vocab_size x n_embd)
- **Positional Embedding:** Learned (context_length x n_embd)

## Training

### Optimization

- **Optimizer:** AdamW
- **Weight Decay:** 0.1
- **Learning Rate:** 3e-4 (constant)
- **Betas:** (0.9, 0.95)
- **Gradient Clipping:** 1.0

> **Note:** The current implementation uses a constant learning rate. Cosine decay with warmup is planned but not yet implemented.

### Data Pipeline

- **Format:** Memory-mapped binary shards (uint16)
- **Shard Size:** 5M tokens
- **Sampling:** Random window sampling

### Regularization

- **Dropout:** 0.1 (embeddings, residual connections, attention)

### Training Outputs

- **Checkpoints:** PyTorch `.pt` format (checkpoint_step_N.pt)
- **Safetensors:** Exported automatically for Hugging Face compatibility
- **Training Log:** CSV file with step, loss, and elapsed time

## Inference

### Generation

- **Method:** Temperature sampling
- **Default Temperature:** 0.8
- **Stopping:** EOS token or max_new_tokens
- **Context Management:** Truncate from left when exceeding context_length

## Key Design Decisions

1. **Pre-LayerNorm:** Improves training stability
2. **Fused QKV:** Reduces memory allocations, improves efficiency
3. **Flash Attention:** Leverages optimized kernels on A100
4. **Learned Positional Embeddings:** Simple, effective for short context
5. **Separate Output Layer:** No weight tying between embeddings and LM head for simplicity

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)
- GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- nanoGPT (karpathy, 2023)
