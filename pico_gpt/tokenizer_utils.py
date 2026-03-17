"""Utilities for exporting tokenizer metadata."""

import json
from pathlib import Path


def export_tokenizer_metadata(output_dir: str, model_max_length: int = 128):
    """
    Export tokenizer metadata files for Hugging Face compatibility.

    Since we use tiktoken's GPT-2 tokenizer directly, we only export
    metadata files that reference the tokenizer.

    Args:
        output_dir: Directory to save metadata files
        model_max_length: Maximum context length for the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "eos_token": "<|endoftext|>",
        "model_max_length": model_max_length,
        "tokenizer_type": "gpt2",
    }

    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # special_tokens_map.json
    special_tokens_map = {
        "eos_token": "<|endoftext|>",
    }

    with open(output_path / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    print(f"Tokenizer metadata exported to {output_dir}")
