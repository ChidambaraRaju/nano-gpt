"""Configuration module for Pico-GPT model and training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    vocab_size: int = 50257
    context_length: int = 128

    # Feedforward
    ffn_dim: int = 1536  # 4 * n_embd

    # Regularization
    dropout: float = 0.1

    # Attention
    bias: bool = False
    flash_attention: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.ffn_dim == 4 * self.n_embd, "ffn_dim should be 4 * n_embd"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Dataset
    data_dir: str = "data"
    shard_size: int = 5_000_000  # tokens per shard
    total_tokens: int = 1_000_000_000  # total training tokens (1B)
    val_tokens: int = 50_000_000  # validation tokens (50M)

    # Training hyperparameters
    batch_size: int = 64

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)

    # Training
    max_steps: int = 200_000
    checkpoint_interval: int = 1000
    grad_clip: float = 1.0

    # Logging
    log_interval: int = 1
    output_dir: str = "outputs"

    def __post_init__(self):
        """Validate configuration."""
        pass


@dataclass
class GenerationConfig:
    """Text generation configuration."""

    model_path: str = "checkpoints/best_model.pt"
    max_new_tokens: int = 100
    temperature: float = 0.8
    prompt: str = "The future of artificial intelligence is"
