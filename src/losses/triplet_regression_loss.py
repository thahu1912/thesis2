import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TripletRegressionLoss(nn.Module):
    """Triplet Regression Loss for distance learning.
    
    This loss function learns a regression model to predict distances between embeddings
    instead of using fixed margins. It uses a dropout rate of 0.2 for uncertainty estimation.
    
    Args:
        distance: Distance metric to use ('euclidean' or 'cosine')
        dropout_rate: Dropout rate for uncertainty estimation (default: 0.2)
    """
    
    def __init__(self, distance: str = 'euclidean', dropout_rate: float = 0.2):
        super().__init__()
        self.distance = distance
        self.dropout_rate = dropout_rate
        
        # Regression network to predict distances
        self.regression_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance between two sets of embeddings."""
        if self.distance == 'euclidean':
            return torch.cdist(x, y, p=2)
        elif self.distance == 'cosine':
            return 1 - F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
            
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass computing the triplet regression loss.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)
            
        Returns:
            Loss value
        """
        # Get unique labels
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        
        # Initialize lists to store distances
        pos_distances = []
        neg_distances = []
        
        # For each class
        for label in unique_labels:
            # Get indices for current class
            pos_indices = (labels == label).nonzero().squeeze()
            neg_indices = (labels != label).nonzero().squeeze()
            
            if len(pos_indices.shape) == 0:
                pos_indices = pos_indices.unsqueeze(0)
            if len(neg_indices.shape) == 0:
                neg_indices = neg_indices.unsqueeze(0)
                
            # Get embeddings for current class
            pos_embeddings = embeddings[pos_indices]
            neg_embeddings = embeddings[neg_indices]
            
            # Compute distances
            pos_dist = self.compute_distance(pos_embeddings, pos_embeddings)
            neg_dist = self.compute_distance(pos_embeddings, neg_embeddings)
            
            # Add to lists
            pos_distances.append(pos_dist)
            neg_distances.append(neg_dist)
            
        # Concatenate all distances
        pos_distances = torch.cat(pos_distances, dim=0)
        neg_distances = torch.cat(neg_distances, dim=0)
        
        # Create input features for regression network
        pos_features = torch.cat([pos_distances, torch.zeros_like(pos_distances)], dim=1)
        neg_features = torch.cat([neg_distances, torch.ones_like(neg_distances)], dim=1)
        
        # Predict distances
        pos_pred = self.regression_net(pos_features)
        neg_pred = self.regression_net(neg_features)
        
        # Compute loss (MSE between predicted and actual distances)
        pos_loss = F.mse_loss(pos_pred, pos_distances)
        neg_loss = F.mse_loss(neg_pred, neg_distances)
        
        # Total loss
        loss = pos_loss + neg_loss
        
        return loss 