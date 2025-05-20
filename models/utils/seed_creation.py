#!/usr/bin/env python3
"""
seed_creation_adni.py

Generate reproducible train/val/test splits for ADNI MRI preprocessed volumes.

Features:
  - Reads all .pt files in a specified input directory (filename without extension = subject ID)
  - Supports configurable train/validation/test ratios
  - Uses a fixed random seed for reproducibility
  - Saves splits to JSON or plain-text
  - Computes and prints age and sex statistics for each split based on a metadata CSV

Usage:
    python seed_creation_adni.py \
        --input_dir /path/to/preprocessed_adni \
        --metadata_csv /path/to/adni_metadata.csv \
        --subject_column subject \
        --output_split splits.json \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed 2025
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits for ADNI MRI dataset"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory containing preprocessed .pt volumes'
    )
    parser.add_argument(
        '--metadata_csv', type=str, required=True,
        help='CSV file with subject metadata (age, sex, etc.)'
    )
    parser.add_argument(
        '--subject_column', type=str, default='subject',
        help='Column name in metadata CSV corresponding to subject ID'
    )
    parser.add_argument(
        '--output_split', type=str, default='splits.json',
        help='Path to save JSON splits file'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.7,
        help='Fraction of data to use for training'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.15,
        help='Fraction of data to use for validation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Set random seed
    np.random.seed(args.seed)

    # List all preprocessed subject files
    files = os.listdir(args.input_dir)
    pt_files = [f for f in files if f.endswith('.pt')]
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {args.input_dir}")

    # Extract subject IDs (filename without .pt)
    subjects = np.array([os.path.splitext(f)[0] for f in pt_files])

    # Shuffle subjects
    np.random.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_ids = subjects[:n_train]
    val_ids = subjects[n_train:n_train + n_val]
    test_ids = subjects[n_train + n_val:]

    # Save splits
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    with open(args.output_split, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved split file to {args.output_split}")

    # Load metadata and compute statistics
    meta = pd.read_csv(args.metadata_csv)
    subj_col = args.subject_column
    if subj_col not in meta.columns:
        raise KeyError(f"Column '{subj_col}' not found in {args.metadata_csv}")

    def describe_split(name, ids):
        subset = meta[meta[subj_col].astype(str).isin(ids)]
        print(f"\n{name} set: {len(ids)} subjects")
        if 'age' in subset.columns:
            print(f"  Age mean: {subset['age'].mean():.2f}, std: {subset['age'].std():.2f}")
        if 'sex' in subset.columns:
            counts = subset['sex'].value_counts()
            print(f"  Sex counts:\n{counts.to_string()}")

    describe_split('Train', train_ids)
    describe_split('Validation', val_ids)
    describe_split('Test', test_ids)

if __name__ == '__main__':
    main()
