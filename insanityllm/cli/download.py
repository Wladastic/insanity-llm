"""
insanityllm.cli.download
Dataset downloader CLI for Insanity LLM.
"""
import argparse
import json
from pathlib import Path
from typing import Optional
from insanityllm.config import setup_environment

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Please install it with: pip install datasets")
    exit(1)

def download_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> None:
    config = setup_environment()
    if output_dir is None:
        output_dir = config['DATASETS_DIR']
    print(f"Downloading dataset: {dataset_name}")
    if subset:
        print(f"Subset: {subset}")
    print(f"Split: {split}")
    print(f"Output directory: {output_dir}")
    dataset = load_dataset(dataset_name, subset, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")
    else:
        print(f"Downloaded {len(dataset)} samples")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename_parts = [dataset_name.replace("/", "_")]
    if subset:
        filename_parts.append(subset)
    filename_parts.append(split)
    if max_samples:
        filename_parts.append(f"max{max_samples}")
    filename = "_".join(filename_parts) + ".jsonl"
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Dataset saved to: {output_file}")
    print("\nFirst sample:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets for DPO training")
    parser.add_argument("dataset_name", help="Hugging Face dataset name (e.g., sam-paech/gutenbergs_1_2_3_4-antislop-dpo)")
    parser.add_argument("--subset", help="Dataset subset/configuration (if applicable)")
    parser.add_argument("--split", default="train", help="Dataset split to download (default: train)")
    parser.add_argument("--output-dir", help="Output directory (default: from .env DATASETS_DIR)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to download")
    args = parser.parse_args()
    download_dataset(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
