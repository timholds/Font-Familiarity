import torch
from torch import nn

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Basic CNN architecture for font classification.
    Input: (batch_size, 1, 256, 256)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
# class FontEncoder(nn.Module):
#     def __init__(self, latent_dim: int = 128):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 64, 512),
#             nn.ReLU(),
#             nn.Linear(512, latent_dim)
#         )
    
#     def forward(self, x):
#         embeddings = self.encoder(x)
#         # Normalize embeddings to lie on unit hypersphere
#         return F.normalize(embeddings, p=2, dim=1)

# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature: float = 0.07):
#         super().__init__()
#         self.temperature = temperature
        
#     def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         # Create similarity matrix
#         sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
#         # Create mask for positive pairs
#         labels = labels.view(-1, 1)
#         mask = torch.eq(labels, labels.T).float()
        
#         # Remove diagonal elements (self-similarity)
#         mask = mask - torch.eye(mask.shape[0], device=mask.device)
        
#         # Compute log_prob
#         exp_sim = torch.exp(sim_matrix)
#         log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
#         # Compute mean of positive pair similarities
#         mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
#         return -mean_log_prob.mean()