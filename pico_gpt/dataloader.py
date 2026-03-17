"""Memory-mapped dataset loader for training.

Reference: @instructions/05_training_pipeline.md
"""

from pathlib import Path
from typing import Tuple
import numpy as np


class MemoryMappedDataset:
    """
    Memory-mapped dataset for efficient training.

    Loads binary token shards using NumPy memmap for zero-copy access.
    Supports random sampling of context windows.

    Reference: @instructions/05_training_pipeline.md
    """

    def __init__(
        self,
        data_dir: str | Path,
        context_length: int,
        batch_size: int,
        split: str = "train",
    ):
        """
        Initialize dataset loader.

        Args:
            data_dir: Directory containing binary shards
            context_length: Context window size
            batch_size: Batch size for sampling
            split: "train" or "val"
        """
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.batch_size = batch_size
        self.split = split

        # Load shards
        self.tokens = self._load_shards()

    def _load_shards(self) -> np.memmap:
        """Load and concatenate shards into a single memmap."""
        if self.split == "val":
            # Load validation data
            val_path = self.data_dir / "val.bin"
            if not val_path.exists():
                raise FileNotFoundError(f"Validation shard not found: {val_path}")
            return np.memmap(val_path, dtype=np.uint16, mode="r")
        else:
            # Load and concatenate training shards
            shard_files = sorted(self.data_dir.glob("train_*.bin"))
            if not shard_files:
                raise FileNotFoundError(f"No training shards found in {self.data_dir}")

            # Load all shards
            shards = []
            for shard_file in shard_files:
                shard = np.memmap(shard_file, dtype=np.uint16, mode="r")
                shards.append(shard)

            # Concatenate shards
            return np.concatenate(shards)

    @property
    def n_tokens(self) -> int:
        """Get total number of tokens in dataset."""
        return len(self.tokens)

    def __len__(self) -> int:
        """Get number of valid starting positions for context windows."""
        return self.n_tokens - self.context_length

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch of (x, y) pairs.

        Returns:
            x: Input tokens of shape (batch_size, context_length)
            y: Target tokens of shape (batch_size, context_length)
            where y[i, j] = x[i, j+1]
        """
        # Sample random start positions
        max_start = self.n_tokens - self.context_length
        starts = np.random.randint(0, max_start, size=self.batch_size)

        # Sample tokens
        x = np.stack([self.tokens[s:s + self.context_length] for s in starts])
        y = np.stack([self.tokens[s + 1:s + self.context_length + 1] for s in starts])

        return x.astype(np.int64), y.astype(np.int64)

    def __iter__(self):
        """Iterate over random batches."""
        while True:
            yield self.get_batch()
