"""
Export script for Hugging Face Hub.
Reference: @instructions/07_huggingface_export.md
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from pico_gpt.export import export_to_huggingface, upload_to_hub


def main():
    parser = argparse.ArgumentParser(description="Export Pico-GPT to Hugging Face format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="hf_model", help="Output directory")
    parser.add_argument("--upload", type=str, default=None, help="Upload to Hugging Face (repo_id)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    # Export model
    export_to_huggingface(args.checkpoint, args.output)

    # Upload if requested
    if args.upload:
        upload_to_hub(args.upload, args.output, private=args.private)


if __name__ == "__main__":
    main()
