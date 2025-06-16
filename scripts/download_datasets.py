#!/usr/bin/env python3
"""
Dataset downloader for DPO training.
Allows downloading any Hugging Face dataset and converting it to JSONL format.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Please install it with: pip install datasets")
    exit(1)


def download_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    output_dir: str = "data",
    max_samples: Optional[int] = None
) -> None:
    """
    Download a dataset from Hugging Face and save it as JSONL.
    
    Args:
        dataset_name: Name of the Hugging Face dataset (e.g., "microsoft/orca-dpo")
        subset: Subset/configuration name (if applicable)
        split: Dataset split to download (default: "train")
        output_dir: Directory to save the dataset
        max_samples: Maximum number of samples to download (None for all)
    """
    print(f"Downloading dataset: {dataset_name}")
    if subset:
        print(f"Subset: {subset}")
    print(f"Split: {split}")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, subset, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"Limited to {len(dataset)} samples")
        else:
            print(f"Downloaded {len(dataset)} samples")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename_parts = [dataset_name.replace("/", "_")]
        if subset:
            filename_parts.append(subset)
        filename_parts.append(split)
        if max_samples:
            filename_parts.append(f"max{max_samples}")
        
        filename = "_".join(filename_parts) + ".jsonl"
        output_file = output_path / filename
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Dataset saved to: {output_file}")
        
        # Show first few samples
        print("\nFirst sample:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets for DPO training")
    parser.add_argument("dataset_name", help="Hugging Face dataset name (e.g., microsoft/orca-dpo)")
    parser.add_argument("--subset", help="Dataset subset/configuration (if applicable)")
    parser.add_argument("--split", default="train", help="Dataset split to download (default: train)")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
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
