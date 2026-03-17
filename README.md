# Pico-GPT

A GPT-style decoder-only Small Language Model (~35M parameters) built from scratch using PyTorch.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python scripts/prepare_data.py
python scripts/train.py
python scripts/generate.py
```

## Project Structure

See `docs/plans/2026-03-17-pico-gpt-implementation.md` for detailed implementation plan.

## Hardware Requirements

- NVIDIA A100 GPU (~20 GB VRAM) recommended
- BF16 mixed precision training supported
- Training time: ~22 hours on A100
