# Pico-GPT Training Manual

Complete guide to train a GPT-style Small Language Model (~30M parameters) from scratch using the Pico-GPT pipeline.

---

## Prerequisites

Before starting, ensure you have:

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (A100/RTX 3090/4090 recommended)
- ~50GB disk space for dataset and checkpoints

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Pipeline Overview

The pipeline consists of four steps:

| Step | Script | Output | Time |
|------|--------|--------|------|
| 1. Dataset Preparation | `prepare_data.py` | Binary shards | 2-4 hours |
| 2. Training | `train.py` | Model checkpoints | 20-24 hours |
| 3. Generation | `generate.py` | Text samples | Seconds |
| 4. HuggingFace Export | `export_hf.py` | Uploadable model | Minutes |

---

## Step 1: Dataset Preparation

The `prepare_data.py` script streams the OpenWebText dataset, tokenizes it using GPT-2 tokenizer, and saves binary shards for efficient memory-mapped loading during training.

### Default Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `--output-dir` | `data` | Directory for binary shards |
| `--shard-size` | `5,000,000` | Tokens per shard |
| `--total-tokens` | `1,000,000,000` | Total tokens to process (1B) |
| `--val-tokens` | `50,000,000` | Validation tokens (last 5%) |
| `--no-resume` | False | Start from scratch if set |

### Running the Script

```bash
python scripts/prepare_data.py
```

This will:
- Stream OpenWebText dataset (~80GB compressed for 1B tokens)
- Tokenize ~1B tokens using GPT-2 tokenizer
- Create 190 training shards (`train_000.bin` through `train_189.bin`)
- Create 1 validation shard (`val.bin`)
- Save state to `data/preprocessing_state.json` for resume capability

### Expected Output

```
Loading OpenWebText dataset (streaming)...
Processing up to 1,000,000,000 tokens...
Validation split: 50,000,000 tokens

Dataset preparation complete!
Total tokens processed: 1,000,000,000
Train tokens: 950,000,000
Val tokens: 50,000,000
```

### Resume Capability

If the script is interrupted, simply run it again. It will automatically resume from the last saved state.

To start from scratch:

```bash
python scripts/prepare_data.py --no-resume
```

### Adjusting Token Count

For faster testing, reduce the token count:

```bash
python scripts/prepare_data.py --total-tokens 10000000 --val-tokens 500000
```

This processes only 10M tokens for quick iteration.

Note: Full dataset preparation uses 1B tokens (1,000,000,000) with 190 training shards and 50M validation tokens.

---

## Step 2: Model Training

The `train.py` script trains the model using the prepared dataset.

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layer` | 6 | Transformer layers |
| `n_head` | 6 | Attention heads |
| `n_embd` | 384 | Embedding dimension |
| `vocab_size` | 50,257 | GPT-2 vocabulary |
| `context_length` | 128 | Maximum sequence length |
| `ffn_dim` | 1,536 | Feedforward dimension (4×n_embd) |
| **Total Parameters** | **~30M** |  |

### Training Configuration

| Parameter | Default | Description |
|-----------|----------|-------------|
| `batch_size` | 64 | Total batch size |
| `micro_batch_size` | 8 | Batch size per GPU step |
| `gradient_accumulation` | 8 | Accumulation steps (64/8) |
| `learning_rate` | `3e-4` | Peak learning rate |
| `weight_decay` | `0.1` | L2 regularization |
| `warmup_steps` | 2,000 | Linear warmup steps |
| `min_lr` | `3e-5` | Minimum learning rate |
| `max_steps` | 200,000 | Total training steps |
| `eval_interval` | 200 | Validation every 200 steps |
| `checkpoint_interval` | 1,000 | Checkpoint every 1k steps |
| `grad_clip` | `1.0` | Gradient clipping threshold |
| `use_bf16` | True | Mixed precision training |

### Running the Script

```bash
python scripts/train.py
```

### Expected Output

```
Model config: ModelConfig(n_layer=6, n_head=6, n_embd=384, ...)
Training config: TrainingConfig(batch_size=64, learning_rate=3e-4, ...)
Model parameters: 29,974,656
Training tokens: 950,000,000
Validation tokens: 50,000,000

Starting training for 200000 steps...
Device: cuda
BF16: True
Batch size: 64 (micro: 8, grad_acc: 8)

Step     10 | Loss: 10.8234 | LR: 1.53e-06 | Time: 2.3s
Step     20 | Loss: 10.4567 | LR: 3.06e-06 | Time: 4.6s
  -> Val Loss: 10.2345
  Saved checkpoint: best_model.pt
...
```

### Checkpointing Strategy

| Checkpoint | When Saved | Description |
|-----------|------------|-------------|
| `best_model.pt` | On improved val loss | Best performing checkpoint |
| `checkpoint_1000.pt` | Every 1,000 steps | Periodic checkpoints |
| `checkpoint_2000.pt` | Every 1,000 steps | Periodic checkpoints |
| ... | ... | ... |
| `final_model.pt` | At training end | Final model state |

### Training Logs

The script creates `checkpoints/training_log.csv` with step-by-step metrics:

```csv
step,train_loss,val_loss,lr
0,10.8234,10.2345,1.53e-06
10,10.4567,9.9876,3.06e-06
...
```

### Resume Training

If training is interrupted, resume from a checkpoint:

```bash
python scripts/train.py --resume checkpoints/checkpoint_1000.pt
```

### Override Training Steps

For shorter training runs:

```bash
python scripts/train.py --max-steps 10000
```

### Expected Training Time

On an NVIDIA A100 (20GB):

| Metric | Value |
|--------|-------|
| Steps per second | ~2.5 |
| Time per step | ~0.4s |
| Total training time | ~22 hours |
| Estimated final loss | ~3.2-3.5 |

---

## Step 3: Text Generation

The `generate.py` script generates text from a trained model.

### Default Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `--model` | `checkpoints/best_model.pt` | Model checkpoint path |
| `--prompt` | `The future of artificial intelligence is` | Input prompt |
| `--max-tokens` | 100 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |

### Running the Script

```bash
python scripts/generate.py --model checkpoints/best_model.pt --prompt "Machine learning is" --max-tokens 50
```

### Expected Output

```
Loading model from checkpoints/best_model.pt...
Generating with temperature 0.8...
Prompt: Machine learning is
--------------------------------------------------
Machine learning is a powerful tool for understanding data and making predictions. It has revolutionized fields like computer vision, natural language processing, and robotics.
--------------------------------------------------
```

### Temperature Guide

| Temperature | Behavior |
|-------------|-----------|
| 0.5-0.7 | More focused, conservative |
| 0.8-1.0 | Balanced creativity |
| 1.0-1.5 | More random, diverse |

---

## Step 4: HuggingFace Export

The `export_hf.py` script exports the model in HuggingFace-compatible format.

### Default Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--output` | `hf_model` | Output directory |
| `--upload` | None | Upload to HuggingFace (repo_id) |
| `--private` | False | Make repository private |

### Export to Local Directory

```bash
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output hf_model
```

This creates:
```
hf_model/
├── model.safetensors      # Model weights
├── config.json             # Model configuration
├── tokenizer_config.json   # Tokenizer metadata
└── special_tokens_map.json # Special tokens
```

### Upload to HuggingFace

First, set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```

Then upload:

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/best_model.pt \
  --output hf_model \
  --upload username/pico-gpt \
  --private
```

This creates a public or private repository on HuggingFace Hub with your model.

---

## Troubleshooting

### Out of Memory During Training

Reduce the batch size:

```bash
# Edit pico_gpt/config.py or pass different config
# Or reduce micro_batch_size
python scripts/train.py --batch-size 32 --micro-batch-size 4
```

### Dataset Preparation Fails

Check disk space:

```bash
df -h .
```

You need at least 50GB free for the full dataset.

### Checkpoint Loading Fails

Ensure the checkpoint exists:

```bash
ls checkpoints/
```

Verify the checkpoint format:

```python
import torch
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
print(checkpoint.keys())
# Should see: 'model_state_dict', 'optimizer_state_dict', etc.
```

### Slow Training

Enable BF16 mixed precision (default) for faster training:

```bash
python scripts/train.py  # use_bf16=True by default
```

If BF16 is not supported, training will automatically fall back to FP32.

---

## Quick Reference

### All Commands Summary

```bash
# 1. Prepare dataset
python scripts/prepare_data.py

# 2. Train model
python scripts/train.py

# 3. Generate text
python scripts/generate.py --model checkpoints/best_model.pt

# 4. Export to HuggingFace
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --upload username/repo
```

### Test Run (Small Scale)

```bash
# Prepare 10M tokens instead of 100M
python scripts/prepare_data.py --total-tokens 10000000 --val-tokens 500000

# Train for 1000 steps instead of 200k
python scripts/train.py --max-steps 1000

# Generate
python scripts/generate.py
```

### Production Run

```bash
# Full dataset preparation (1B tokens)
python scripts/prepare_data.py

# Full training (200k steps, ~10-12 hours on A100)
python scripts/train.py

# Export and upload
python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --upload username/pico-gpt
```

---

## File Structure After Training

```
pico-gpt/
├── data/                          # Prepared dataset
│   ├── train_000.bin              # Training shard (5M tokens)
│   ├── train_001.bin
│   ├── ...
│   ├── train_018.bin
│   ├── val.bin                    # Validation shard (5M tokens)
│   └── preprocessing_state.json   # Resume state
├── checkpoints/                    # Training outputs
│   ├── best_model.pt              # Best checkpoint
│   ├── checkpoint_1000.pt
│   ├── checkpoint_2000.pt
│   ├── ...
│   └── training_log.csv          # Training metrics
└── hf_model/                      # HuggingFace export
    ├── model.safetensors
    ├── config.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

---

## Next Steps

After successful training:

1. **Analyze training curves** - Plot loss from `training_log.csv`
2. **Generate samples** - Test model with various prompts
3. **Upload to HuggingFace** - Share your model with the community
4. **Fine-tune** - Consider fine-tuning on specific domains
5. **Scale up** - Increase model size or training tokens

---

## Contact & Issues

For issues or questions, refer to:
- `docs/ARCHITECTURE.md` - Model architecture details
- `docs/API.md` - Module-level API documentation
- GitHub Issues - Report bugs or feature requests
