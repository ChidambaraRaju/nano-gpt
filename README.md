# Pico-GPT

A **GPT-style decoder-only Small Language Model (~49M parameters)** built from scratch using PyTorch. This project implements a clean, scalable, and research-grade training pipeline for pretraining a transformer language model from first principles.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset & Data Collection](#dataset--data-collection)
- [Training Pipeline](#training-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)
- [Training Guide](#training-guide)
- [Text Generation](#text-generation)
- [Hugging Face Export](#hugging-face-export)
- [API Reference](#api-reference)
- [Design Philosophy](#design-philosophy)
- [Technical Documentation](#technical-documentation)
- [References](#references)
- [License](#license)

---

## Overview

Pico-GPT is an educational and practical implementation of a GPT-style language model. It demonstrates the complete pipeline from raw text preprocessing to model pretraining, generation, and deployment. The project focuses on:

- **Simplicity**: Minimal dependencies, clear code structure
- **Correctness**: Following established transformer architecture patterns
- **Efficiency**: Memory-mapped data loading, Flash Attention
- **Reproducibility**: Deterministic preprocessing, checkpointing, detailed logging

### Key Numbers

| Metric | Value |
|--------|-------|
| Model Parameters | ~49M |
| Training Tokens | 1B |
| Validation Tokens | 50M |
| Transformer Layers | 6 |
| Context Length | 128 tokens |
| Training Time (A100) | ~10-12 hours |

---

## Features

### Architecture
- **Pre-LayerNorm** design for training stability
- **Fused QKV projection** (GPT-2 style) for efficiency
- **Flash Attention** via PyTorch SDPA
- **Learned positional embeddings**
- **Standard GPT-style decoder-only transformer**

### Training
- **Memory-mapped binary dataset** for efficient loading
- **AdamW optimizer** with constant learning rate
- **Gradient clipping** for training stability
- **Automatic checkpointing** (periodic + final)
- **Training logs** with loss metrics and elapsed time

### Inference
- **Temperature sampling** for controlled generation
- **Autoregressive decoding loop**
- **Context-aware prompt handling**

### Deployment
- **Hugging Face Hub** export support
- **Safetensors** format for secure serialization
- **Custom model config** for architecture reconstruction

---

## Model Architecture

Pico-GPT follows the classic GPT architecture with modern optimizations:

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
LM Head (384 x 50257)
        ↓
Logits (B, T, 50257)
```

### Architecture Specifications

| Component | Configuration |
|-----------|---------------|
| Layers | 6 transformer blocks |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Feedforward Dimension | 1,536 (4 × n_embd) |
| Context Length | 128 tokens |
| Vocabulary Size | 50,257 (GPT-2) |
| Head Dimension | 64 (384 / 6) |
| Dropout | 0.1 |
| Attention Bias | False |

### Transformer Block Details

Each transformer block consists of:

1. **Pre-LayerNorm** (Layer Normalization before attention)
2. **Multi-Head Self-Attention** with:
   - Fused QKV projection (single linear layer)
   - Causal masking for autoregressive generation
   - Flash Attention via PyTorch SDPA
3. **Pre-LayerNorm** (Layer Normalization before MLP)
4. **Feedforward Network**:
   - Linear layer (384 → 1536)
   - GELU activation
   - Linear layer (1536 → 384)
   - Dropout (0.1)

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pre-LayerNorm | Improves training stability, avoids vanishing gradients |
| Fused QKV | Reduces memory allocations, improves GPU efficiency |
| Flash Attention | Leverages optimized kernels on A100, faster training |
| Learned Positional Embeddings | Simple and effective for short context (128) |
| No Weight Tying | Explicit embedding and output layers for clarity |

---

## Dataset & Data Collection

Pico-GPT uses the **OpenWebText** dataset, a collection of web text from Reddit outbound links.

### Data Collection Strategy

The dataset pipeline follows a scalable, memory-efficient design:

1. **Streaming**: Dataset is streamed, not fully loaded into RAM
2. **Incremental Tokenization**: Text is tokenized on-the-fly
3. **Binary Sharding**: Tokens saved in fixed-size binary files
4. **Memory Mapping**: Enables fast loading during training
5. **Resume Capability**: State is saved for interrupted preprocessing

### Preprocessing Pipeline

```
OpenWebText (streaming)
        ↓
Text Cleaning (minimal)
        ↓
Tokenization (GPT-2 tiktoken)
        ↓
EOS Token Append
        ↓
Token Buffer Accumulation
        ↓
Binary Shard Writing (5M tokens/shard)
        ↓
Train/Validation Split (95%/5%)
```

### Text Cleaning Rules

Minimal preprocessing is applied to preserve natural language distribution:

- Strip leading/trailing whitespace
- Skip empty strings
- Normalize multiple spaces → single space

**Not applied:**
- Punctuation removal
- Lowercasing
- Aggressive URL filtering

### Sharding Strategy

| Property | Value |
|----------|-------|
| Shard Size | 5 million tokens |
| Data Type | `uint16` |
| Total Shards | ~190 (training) + 1 (validation) |
| Naming | `train_000.bin`, `train_001.bin`, ... |

### Train/Validation Split

Deterministic split based on token count:

- **Training**: First 950 million tokens
- **Validation**: Last 50 million tokens

Validation tokens are saved separately in `data/val.bin`.

### Resume Capability

The preprocessing state is saved in `data/preprocessing_state.json`:

```json
{
  "current_shard_index": 10,
  "tokens_in_current_shard": 3200000,
  "total_tokens_processed": 53200000
}
```

If interrupted, the script resumes from the last saved state without reprocessing completed shards.

---

## Training Pipeline

### Training Overview

Pico-GPT uses a custom PyTorch training loop with the following features:

- **Memory-mapped dataset loading** for efficient batch sampling
- **Random window sampling** from the token stream
- **AdamW optimizer**
- **Gradient clipping** for training stability
- **Periodic checkpointing** for resumability

### Training Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Batch Size | 64 | Batch size |
| Learning Rate | 3e-4 | Learning rate |
| Weight Decay | 0.1 | L2 regularization |
| Betas | (0.9, 0.95) | AdamW momentum parameters |
| Max Steps | 200,000 | Total training steps |
| Gradient Clipping | 1.0 | Gradient norm threshold |
| Checkpoint Interval | 1,000 | Checkpoint saving frequency |
| Log Interval | 100 | Logging frequency |

> **Note**: Learning rate is constant (no scheduler). A learning rate schedule with warmup and cosine decay is planned for future implementation.

### Batch Sampling Strategy

For each training step:

```python
# Randomly sample start index
i = random_start_index

# Create (x, y) pairs for causal language modeling
x = tokens[i : i + context_length]
y = tokens[i + 1 : i + context_length + 1]
```

This random sampling ensures diverse training data across epochs.

> **Note**: Validation loop is not yet implemented. A validation strategy with periodic evaluation and best checkpoint saving is planned for future implementation.

### Checkpointing Strategy

| Checkpoint | When Saved | Description |
|-----------|------------|-------------|
| `checkpoint_step_N.pt` | Every 1,000 steps | Periodic checkpoints |
| `model_step_N.safetensors` | At training end | Final model in safetensors format |

Each checkpoint contains:
- Model state dict
- Step number
- Model config
- Final checkpoint includes training config

### Training Logs

Training metrics are logged to `outputs/training_log.csv`:

```csv
step,loss,elapsed_time
1,10.8234,0.5
100,9.4567,45.2
...
```

---

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (recommended)
- ~50GB disk space for dataset and checkpoints

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| tiktoken | >=0.5.0 | GPT-2 tokenizer |
| numpy | >=1.24.0 | Numerical operations |
| datasets | >=2.14.0 | OpenWebText dataset streaming |
| safetensors | >=0.4.0 | Safe model serialization |
| matplotlib | >=3.7.0 | Plotting training curves |
| huggingface-hub | >=0.17.0 | Model upload to HF Hub |

---

## Quick Start

### 1. Prepare Dataset

```bash
python scripts/prepare_data.py
```

This streams OpenWebText, tokenizes it, and creates binary shards.

**Flags:**
- `--output-dir data`: Output directory for shards
- `--shard-size 5000000`: Tokens per shard (default: 5M)
- `--total-tokens 1000000000`: Total tokens to process (default: 1B)
- `--val-tokens 50000000`: Validation tokens (default: 50M)
- `--no-resume`: Start from scratch

### 2. Train Model

```bash
python scripts/train.py
```

**Flags:**
- `--data-dir data`: Dataset directory
- `--output-dir outputs`: Output directory
- `--max-steps 200000`: Maximum training steps

### 3. Generate Text

```bash
python scripts/generate.py --model outputs/model_step_200000.safetensors --prompt "The future of AI is"
```

**Flags:**
- `--model`: Model checkpoint path
- `--prompt`: Input text prompt
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)

### 4. Export to Hugging Face

```bash
python scripts/export_hf.py --checkpoint outputs/model_step_200000.safetensors --upload username/pico-gpt
```

**Flags:**
- `--checkpoint`: Path to model checkpoint
- `--output hf_model`: Output directory
- `--upload`: HuggingFace repo_id to upload
- `--private`: Create private repository

---

## Project Structure

```
pico-gpt/
├── pico_gpt/                      # Core implementation
│   ├── __init__.py
│   ├── config.py                  # Configuration classes
│   ├── model.py                   # Model architecture
│   ├── tokenizer.py               # GPT-2 tokenizer wrapper
│   ├── dataloader.py              # Memory-mapped dataset loader
│   ├── trainer.py                 # Training loop
│   └── export.py                  # HuggingFace export utilities
├── scripts/                       # Executable scripts
│   ├── prepare_data.py            # Dataset preprocessing
│   ├── train.py                   # Training script
│   ├── generate.py                # Text generation
│   └── export_hf.py               # Export to HuggingFace
├── docs/                          # Technical documentation
│   ├── ARCHITECTURE.md            # Architecture details
│   ├── API.md                     # API reference
│   ├── TRAINING_MANUAL.md         # Complete training guide
│   └── USAGE.md                   # Usage examples
├── instructions/                  # Implementation specs
│   ├── 01_project_overview.md
│   ├── 02_dataset_preparation.md
│   ├── 03_tokenizer_usage.md
│   ├── 04_model_architecture.md
│   ├── 05_training_pipeline.md
│   ├── 06_text_generation.md
│   └── 07_huggingface_export.md
├── data/                          # Prepared dataset (generated)
│   ├── train_000.bin
│   ├── train_001.bin
│   ├── ...
│   ├── val.bin
│   └── preprocessing_state.json
├── outputs/                       # Training outputs (generated)
│   ├── checkpoint_step_*.pt        # Periodic checkpoints
│   ├── model_step_*.safetensors   # Final model
│   └── training_log.csv            # Training metrics
├── hf_model/                      # HuggingFace export (generated)
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── requirements.txt
├── README.md
└── CLAUDE.md                      # Project instructions
```

---

## Configuration

All configurations are defined in `pico_gpt/config.py`:

### ModelConfig

```python
@dataclass
class ModelConfig:
    n_layer: int = 6              # Transformer layers
    n_head: int = 6               # Attention heads
    n_embd: int = 384             # Embedding dimension
    vocab_size: int = 50257       # GPT-2 vocabulary
    context_length: int = 128     # Maximum sequence length
    ffn_dim: int = 1536           # Feedforward dimension (4×n_embd)
    dropout: float = 0.1          # Dropout rate
    bias: bool = False            # Attention bias
    flash_attention: bool = True  # Use Flash Attention
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    data_dir: str = "data"
    shard_size: int = 5_000_000
    total_tokens: int = 1_000_000_000   # 1B tokens
    val_tokens: int = 50_000_000        # 50M tokens
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_steps: int = 200_000
    checkpoint_interval: int = 1000
    grad_clip: float = 1.0
    output_dir: str = "outputs"
```

### GenerationConfig

```python
@dataclass
class GenerationConfig:
    model_path: str = "outputs/model_step_200000.safetensors"
    max_new_tokens: int = 100
    temperature: float = 0.8
    prompt: str = "The future of artificial intelligence is"
```

---

## Hardware Requirements

### Recommended Setup

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A100 (20GB VRAM) |
| CPU | Modern multi-core processor |
| RAM | 32GB+ |
| Storage | 50GB+ SSD |
| Training Time | ~10-12 hours for 1B tokens |

### Alternative GPUs

| GPU | Expected Performance |
|-----|---------------------|
| RTX 4090 (24GB) | ~12-15 hours |
| RTX 3090 (24GB) | ~15-18 hours |
| RTX 3080 (10GB) | Requires reduced batch size, ~20-24 hours |

### Memory Usage

| Operation | Approximate Memory |
|-----------|-------------------|
| Dataset (1B tokens) | ~2GB (binary) |
| Model (49M params) | ~200MB (FP32) |
| Training (batch=64) | ~10GB (including gradients) |
| Checkpoint | ~400MB per file |

---

## Training Guide

For complete training instructions, see `docs/TRAINING_MANUAL.md`.

### Quick Training Steps

1. **Prepare dataset** (2-4 hours for 1B tokens):
   ```bash
   python scripts/prepare_data.py
   ```

2. **Start training** (10-12 hours on A100):
   ```bash
   python scripts/train.py
   ```

3. **Monitor progress**:
   - Check `outputs/training_log.csv` for metrics
   - Loss values printed during training

> **Note**: Resume functionality is not yet implemented. If training is interrupted, you'll need to restart from the beginning.

---

## Text Generation

### Generation Method

Pico-GPT uses **temperature sampling** for autoregressive text generation:

```python
for _ in range(max_new_tokens):
    # Get logits for last token
    logits = model(idx)[:, -1, :]

    # Scale by temperature
    logits = logits / temperature

    # Sample next token
    probs = softmax(logits)
    next_token = sample(probs)

    # Append and continue
    idx = torch.cat([idx, next_token], dim=1)
```

### Temperature Guide

| Temperature | Behavior | Use Case |
|-------------|-----------|----------|
| 0.3-0.5 | Very focused, deterministic | Factual generation |
| 0.6-0.8 | Balanced creativity | General purpose |
| 0.9-1.2 | More diverse | Creative writing |
| 1.5+ | Highly random | Exploration |

### Example

```bash
python scripts/generate.py \
    --model outputs/model_step_200000.safetensors \
    --prompt "Machine learning is transforming" \
    --max-tokens 50 \
    --temperature 0.8
```

---

## Hugging Face Export

Pico-GPT can export trained models to Hugging Face Hub for sharing and deployment.

### Export Format

```
hf_model/
├── model.safetensors          # Model weights
├── config.json                 # Architecture config
├── tokenizer_config.json       # Tokenizer metadata
├── special_tokens_map.json     # Special tokens
└── README.md                   # Model card
```

### Export Command

```bash
# Export locally
python scripts/export_hf.py \
    --checkpoint outputs/model_step_200000.safetensors \
    --output hf_model

# Export and upload
export HF_TOKEN=your_token_here

python scripts/export_hf.py \
    --checkpoint outputs/model_step_200000.safetensors \
    --output hf_model \
    --upload username/pico-gpt
```

---

## API Reference

For complete API documentation, see `docs/API.md`.

### Core Modules

| Module | Description |
|--------|-------------|
| `pico_gpt.config` | Configuration classes |
| `pico_gpt.model` | Model architecture (`GPT`, `TransformerBlock`, `Attention`) |
| `pico_gpt.tokenizer` | GPT-2 tokenizer wrapper |
| `pico_gpt.dataloader` | Memory-mapped dataset loader |
| `pico_gpt.trainer` | Training loop |
| `pico_gpt.export` | HuggingFace export utilities |

### Example: Using the Model

```python
import torch
from pico_gpt.model import GPT
from pico_gpt.config import ModelConfig
from pico_gpt.tokenizer import GPT2Tokenizer

# Initialize model
config = ModelConfig()
model = GPT(config)

# Load checkpoint
checkpoint = torch.load("outputs/model_step_200000.safetensors")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate text
tokenizer = GPT2Tokenizer()
prompt = "The future of AI is"
tokens = torch.tensor([tokenizer.encode(prompt)])

with torch.no_grad():
    generated = model.generate(tokens, max_new_tokens=100, temperature=0.8)

text = tokenizer.decode(generated[0].tolist())
print(text)
```

---

## Design Philosophy

Pico-GPT follows these guiding principles:

1. **Minimal but Correct**: Prioritize clarity and correctness over unnecessary optimization
2. **No Framework Magic**: Implement core components from first principles
3. **Reproducibility**: Deterministic preprocessing and training
4. **Research-Style**: Follow patterns from nanoGPT and other research implementations
5. **Educational**: Clear, well-documented code suitable for learning
6. **Modern**: Use PyTorch best practices (Flash Attention, etc.)

---

## Technical Documentation

- **[Architecture Details](docs/ARCHITECTURE.md)** - Complete architecture specification
- **[API Reference](docs/API.md)** - Module-level API documentation
- **[Training Manual](docs/TRAINING_MANUAL.md)** - Complete training guide
- **[Usage Examples](docs/USAGE.md)** - Code examples and use cases
- **[Implementation Specs](instructions/)** - Detailed implementation instructions

---

## References

Pico-GPT is inspired by and builds upon the following work:

- **Attention Is All You Need** (Vaswani et al., 2017) - Transformer architecture
- **Improving Language Understanding by Generative Pre-Training** (Radford et al., 2018) - GPT
- **GPT-2: Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2
- **nanoGPT** (karpathy, 2023) - Clean GPT implementation
- **OpenWebText** (Skylion007) - Web text dataset

---

## License

MIT License

---

Made with ❤️ and PyTorch
