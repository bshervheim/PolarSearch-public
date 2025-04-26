#!/usr/bin/env python
"""
Train a Vision Transformer model for semantic product image clustering
using supervised contrastive learning.

Usage:
    python train_model.py --csv_file path/to/your/dataset.csv --output_dir path/to/output

Author: Bradley Shervheim
"""

import argparse
import os
import pandas as pd
import torch
from datetime import datetime

from vit_product_clustering import (
    ImageSetDataset,
    get_transforms,
    train_model
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Vision Transformer for product image clustering"
    )
    
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to CSV file with image URLs and set_ids"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model outputs. Default is a timestamped directory."
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. If not specified, auto-determined based on dataset size."
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size. If not specified, auto-determined based on GPU memory."
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter for contrastive loss"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loading workers. Default is 4 on GPU, 2 on CPU."
    )
    
    parser.add_argument(
        "--no_preload",
        action="store_true",
        help="Disable image preloading to save memory at the cost of training speed"
    )
    
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision training"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for training script."""
    args = parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"vit_supcon_model_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    
    # Load CSV data
    try:
        print(f"Loading data from {args.csv_file}...")
        csv_df = pd.read_csv(args.csv_file)
        print(f"CSV loaded with {len(csv_df)} rows")
        print("CSV columns:", csv_df.columns.tolist())
        
        # Verify required columns
        required_columns = ['set_id', 'image_url']
        for col in required_columns:
            if col not in csv_df.columns:
                raise ValueError(f"CSV file must contain '{col}' column")
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Create image cache directory
    cache_dir = os.path.join(output_dir, 'image_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create dataset
    transform = get_transforms()
    dataset = ImageSetDataset(
        dataframe=csv_df,
        transform=transform,
        cache_dir=cache_dir,
        preload=not args.no_preload
    )
    
    # Train model
    use_amp = not args.no_amp and torch.cuda.is_available()
