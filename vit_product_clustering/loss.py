import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss as described in 
    https://arxiv.org/abs/2004.11362
    
    Args:
        temperature (float): A scaling factor in the exponential. Default is 0.07.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Calculate the supervised contrastive loss
        
        Args:
            features (torch.Tensor): Feature tensor of shape [batch_size, feature_dim]
            labels (torch.Tensor): Label tensor of shape [batch_size]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        device = features.device
        
        # Reshape labels for comparison [batch_size, 1]
        labels = labels.contiguous().view(-1, 1)
        
        # Create mask for positive pairs (same label) [batch_size, batch_size]
        mask = torch.eq(labels, labels.T).float().to(device)

        # Normalize features to unit vectors
        features = nn.functional.normalize(features, dim=1)

        # Calculate similarity matrix
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Subtract max for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Create positive mask (excluding self-contrasting)
        mask_pos = mask.clone()
        mask_pos.fill_diagonal_(0)
        
        # Count positive examples per sample
        pos_per_sample = mask_pos.sum(dim=1)
        
        # Handle samples with no positives (only one example of a class)
        # Replace zeros with ones to avoid division by zero
        pos_per_sample = torch.where(pos_per_sample > 0, 
                                      pos_per_sample, 
                                      torch.ones_like(pos_per_sample))

        # Compute the average log-likelihood for positive pairs
        loss = -(mask_pos * log_prob).sum(dim=1) / pos_per_sample
        
        # Return mean loss across the batch
        return loss.mean()
