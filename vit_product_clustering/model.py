import torch
import torch.nn as nn
from transformers import ViTModel

class ViTContrastive(nn.Module):
    """
    Vision Transformer model for contrastive learning with a projection head
    
    Args:
        pretrained_model_name (str): Name of the pretrained ViT model
        projection_dims (list): List of dimensions for projection head layers
    """
    def __init__(self, 
                 pretrained_model_name='google/vit-base-patch16-224-in21k',
                 projection_dims=[768, 256, 128]):
        super(ViTContrastive, self).__init__()
        
        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        
        # Create projection head layers
        layers = []
        for i in range(len(projection_dims) - 1):
            layers.append(nn.Linear(projection_dims[i], projection_dims[i+1]))
            if i < len(projection_dims) - 2:  # Add ReLU for all but the last layer
                layers.append(nn.ReLU())
                
        self.projection = nn.Sequential(*layers)

    def forward(self, pixel_values):
        """
        Forward pass
        
        Args:
            pixel_values (torch.Tensor): Image tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Embedding vectors of shape [batch_size, projection_dims[-1]]
        """
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0]  # Get CLS token output
        embeddings = self.projection(cls_output)
        return embeddings
    
    def get_embedding(self, pixel_values):
        """
        Get normalized embedding vectors for inference
        
        Args:
            pixel_values (torch.Tensor): Image tensor
            
        Returns:
            torch.Tensor: Normalized embedding vectors
        """
        with torch.no_grad():
            embeddings = self.forward(pixel_values)
            # Normalize embeddings to unit length
            embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings
