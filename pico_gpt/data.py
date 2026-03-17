"""Dataset preprocessing utilities.

Reference: @instructions/02_dataset_preparation.md
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import numpy as np


@dataclass
class PreprocessingState:
    """State for dataset preprocessing resume capability."""

    shard_index: int  # Current shard being filled
    tokens_written: int  # Tokens in current shard
    total_tokens: int  # Total tokens processed across all shards

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PreprocessingState":
        """Load state from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def is_shard_complete(self, shard_size: int) -> bool:
        """Check if current shard is complete."""
        return self.tokens_written >= shard_size

    def tokens_until_shard_complete(self, shard_size: int) -> int:
        """Get remaining tokens needed to complete current shard."""
        remaining = shard_size - self.tokens_written
        return max(0, remaining)


class TokenBuffer:
    """
    Buffer for accumulating tokens during streaming preprocessing.

    Handles:
    - Accumulating tokens from documents
    - Writing complete shards to disk
    - Managing train/validation split

    Reference: @instructions/02_dataset_preparation.md
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size: int = 5_000_000,
        total_tokens: int = 100_000_000,
        val_tokens: int = 5_000_000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.total_tokens = total_tokens
        self.val_tokens = val_tokens
        self.train_tokens = total_tokens - val_tokens

        self.buffer: List[int] = []
        self.total_processed = 0
        self.shard_index = 0

    def add_tokens(self, tokens: List[int]) -> bool:
        """
        Add tokens to buffer and write shards as needed.

        Returns:
            True if more tokens can be added, False if target reached
        """
        if self.total_processed >= self.total_tokens:
            return False

        remaining_needed = self.total_tokens - self.total_processed
        tokens_to_add = tokens[:remaining_needed]
        self.buffer.extend(tokens_to_add)
        self.total_processed += len(tokens_to_add)

        # Write shards as they fill up
        while len(self.buffer) >= self.shard_size and self.total_processed < self.total_tokens:
            shard_tokens = self.buffer[:self.shard_size]
            self.buffer = self.buffer[self.shard_size:]
            self._write_shard(shard_tokens)

        return True

    def _write_shard(self, tokens: List[int]) -> None:
        """Write a shard to disk (either train or validation)."""
        # Determine if this should be train or validation
        tokens_so_far = self.total_processed - len(self.buffer)
        train_tokens_so_far = min(tokens_so_far, self.train_tokens)

        if train_tokens_so_far + len(tokens) <= self.train_tokens:
            # Pure training shard
            self._write_train_shard(tokens, self.shard_index)
        elif train_tokens_so_far >= self.train_tokens:
            # Pure validation shard
            self._write_val_shard(tokens)
        else:
            # Split shard - part train, part validation
            remaining_train = self.train_tokens - train_tokens_so_far
            train_tokens = tokens[:remaining_train]
            val_tokens = tokens[remaining_train:]
            self._write_train_shard(train_tokens, self.shard_index)
            self._write_val_shard(val_tokens)

        self.shard_index += 1

    def _write_train_shard(self, tokens: List[int], index: int) -> None:
        """Write training shard to disk."""
        path = self.output_dir / f"train_{index:03d}.bin"
        np.array(tokens, dtype=np.uint16).tofile(path)

    def _write_val_shard(self, tokens: List[int]) -> None:
        """Write validation shard to disk."""
        path = self.output_dir / "val.bin"
        np.array(tokens, dtype=np.uint16).tofile(path)

    def finalize(self) -> None:
        """Write any remaining tokens in buffer."""
        if self.buffer and self.total_processed <= self.total_tokens:
            # Determine if remaining tokens are train or validation
            train_tokens_so_far = min(self.total_processed, self.train_tokens)
            if train_tokens_so_far < self.train_tokens:
                self._write_train_shard(self.buffer, self.shard_index)
            else:
                self._write_val_shard(self.buffer)
