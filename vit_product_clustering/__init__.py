"""
Vision Transformer (ViT) for Semantic Product Image Clustering
=============================================================

This package implements a supervised contrastive learning framework built upon 
the Vision Transformer (ViT) architecture for semantic product image clustering.

The implementation adds a multi-layer projection head to map the ViT's [CLS]
token representation to a lower-dimensional embedding space optimized for
semantic similarity learning.

For more information, see the README.md file in the repository root.

Author: Bradley Shervheim, Chief Science Officer at Polar Search Inc
Licensed under MIT License
"""

__version__ = '0.1.0'
__author__ = 'Bradley Shervheim'
__email__ = 'brad@polarsearch.io'

from .model import ViTContrastive
from .loss import SupConLoss
from .utils import (
    ImageSetDataset, 
    get_transforms, 
    create_dataloader,
    preload_images
)
from .training import train_model

__all__ = [
    'ViTContrastive',
    'SupConLoss',
    'ImageSetDataset',
    'get_transforms',
    'create_dataloader',
    'preload_images',
    'train_model'
]
