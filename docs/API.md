# Pico-GPT API Reference

## Core Modules

### `pico_gpt.config`

Configuration classes for model and training.

#### `ModelConfig`

```python
@dataclass
class ModelConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    vocab_size: int = 50257
    context_length: int = 128
    ffn_dim: int = 1536
    dropout: float = 0.1
    bias: bool = False
    flash_attention: bool = True
```

#### `TrainingConfig`

```python
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
```

> **Note:** The current implementation uses a simplified training loop. Advanced features like gradient accumulation, learning rate scheduling, and validation loops are not yet implemented.

#### `GenerationConfig`

```python
@dataclass
class GenerationConfig:
    """Text generation configuration."""

    model_path: str = "checkpoints/best_model.pt"
    max_new_tokens: int = 100
    temperature: float = 0.8
    prompt: str = "The future of artificial intelligence is"
```

### `pico_gpt.tokenizer`

Tokenizer wrapper for OpenAI tiktoken.

#### `GPT2Tokenizer`

```python
class GPT2Tokenizer:
    """Wrapper for tiktoken GPT-2 tokenizer."""

    def __init__(self) -> None:
        """Initialize tokenizer."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""

    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text."""

    def truncate(self, tokens: List[int], max_length: int) -> List[int]:
        """Truncate tokens from left to max_length."""
```

### `pico_gpt.model`

Model architecture components.

#### `GPT`

```python
class GPT(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize model."""

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: Input token ids (B, T)
            targets: Target token ids (B, T), optional

        Returns:
            logits (B, T, vocab_size), loss or None
        """

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: Input token ids (B, T)
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature
            eos_token_id: EOS token ID to stop generation (optional)

        Returns:
            Generated tokens (B, T + max_new_tokens)
        """
```

### `pico_gpt.dataloader`

Memory-mapped dataset loader.

#### `MemoryMappedDataset`

```python
class MemoryMappedDataset:
    """Memory-mapped dataset for efficient training."""

    def __init__(
        self,
        data_dir: str | Path,
        context_length: int,
        batch_size: int,
        split: str = "train"
    ) -> None:
        """Initialize dataset loader."""

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch of (x, y) pairs."""

    def __len__(self) -> int:
        """Number of valid start positions."""

    @property
    def n_tokens(self) -> int:
        """Total number of tokens in dataset."""
```

### `pico_gpt.data`

Dataset preprocessing utilities.

#### `PreprocessingState`

```python
@dataclass
class PreprocessingState:
    """State for dataset preprocessing resume capability."""

    shard_index: int  # Current shard being filled
    tokens_written: int  # Tokens in current shard
    total_tokens: int  # Total tokens processed across all shards
    total_processed: int = 0  # Cumulative tokens processed

    def save(self, path: Path) -> None:
        """Save state to JSON file."""

    @classmethod
    def load(cls, path: Path) -> "PreprocessingState":
        """Load state from JSON file."""
```

#### `TokenBuffer`

```python
class TokenBuffer:
    """Buffer for accumulating tokens during streaming preprocessing."""

    def __init__(
        self,
        output_dir: Path,
        shard_size: int = 5_000_000,
        total_tokens: int = 100_000_000,
        val_tokens: int = 5_000_000,
    ) -> None:
        """Initialize token buffer."""

    def add_tokens(self, tokens: List[int]) -> bool:
        """Add tokens to buffer and write shards as needed."""

    def finalize(self) -> None:
        """Write any remaining tokens in buffer."""
```

### `pico_gpt.trainer`

Training loop utilities.

#### `Trainer`

```python
class Trainer:
    """Minimal training loop for Pico-GPT."""

    def __init__(
        self,
        model: GPT,
        train_loader: MemoryMappedDataset,
        output_dir: str,
        config: ModelConfig,
        max_steps: int,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        checkpoint_interval: int = 1000,
        log_interval: int = 100,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: GPT model to train
            train_loader: Training dataset loader
            output_dir: Directory for outputs and checkpoints
            config: Model configuration
            max_steps: Maximum training steps
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            checkpoint_interval: Save checkpoint every N steps
            log_interval: Print logs every N steps
        """

    def train(self) -> None:
        """Main training loop."""

    def save_checkpoint(self, step: int, training_config: dict = None) -> None:
        """Save model checkpoint in PyTorch format."""

    def save_safetensors(self, step: int) -> None:
        """Save model in safetensors format for Hugging Face export."""
```

> **Note:** The current trainer implementation is minimal. It uses basic AdamW optimizer with gradient clipping. Validation loop, learning rate scheduling, and gradient accumulation are not yet implemented.

## Scripts

### `scripts/prepare_data.py`

Preprocess OpenWebText dataset.

```bash
python scripts/prepare_data.py \
    --output-dir data \
    --shard-size 5000000 \
    --total-tokens 100000000 \
    --val-tokens 5000000
```

### `scripts/train.py`

Train model.

```bash
python scripts/train.py \
    --data-dir data \
    --output-dir checkpoints \
    --max-steps 200000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Directory containing binary shards |
| `--output-dir` | `checkpoints` | Output directory for checkpoints |
| `--max-steps` | `200000` | Maximum training steps |
| `--lr` | `3e-4` | Learning rate |
| `--checkpoint-interval` | `1000` | Save checkpoint every N steps |

### `scripts/generate.py`

Generate text from trained model.

```bash
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "The future of AI is" \
    --max-tokens 100 \
    --temperature 0.8
```

### `scripts/export_hf.py`

Export to Hugging Face format.

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/best_model.pt \
    --output hf_model \
    --training-log outputs/training_log.csv \
    --upload username/pico-gpt \
    --private
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--output` | `hf_model` | Output directory |
| `--training-log` | `None` | Path to training_log.csv file |
| `--upload` | `None` | Upload to Hugging Face (repo_id) |
| `--private` | `False` | Make repository private |

## Additional Modules

### `pico_gpt.export`

Hugging Face export utilities.

#### `export_to_huggingface`

```python
def export_to_huggingface(
    checkpoint_path: str,
    output_dir: str,
    training_log_path: Optional[str] = None,
) -> None:
    """
    Export model to Hugging Face format.

    Creates:
    - model.safetensors: Model weights
    - config.json: Model architecture configuration
    - training_config.json: Training hyperparameters
    - training_log.csv: Training metrics (if provided)
    - samples.txt: Generated text samples
    - tokenizer_config.json: Tokenizer configuration
    - special_tokens_map.json: Special tokens
    - README.md: Model card
    """
```

#### `upload_to_hub`

```python
def upload_to_hub(
    repo_id: str,
    model_dir: str,
    private: bool = False,
) -> None:
    """
    Upload model to Hugging Face Hub.

    Requires huggingface-hub package and authentication.
    """
```

### `pico_gpt.tokenizer_utils`

Tokenizer metadata export utilities.

#### `export_tokenizer_metadata`

```python
def export_tokenizer_metadata(
    output_dir: str,
    model_max_length: int = 128
) -> None:
    """
    Export tokenizer metadata files for Hugging Face compatibility.

    Creates:
    - tokenizer_config.json
    - special_tokens_map.json
    """
```
