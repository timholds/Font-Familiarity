import torch
from torch import nn

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    """
    Basic CNN architecture for font classification.
    Input: (batch_size, 1, 256, 256)
    """
    def __init__(self, num_classes: int):
        super().__init__()     
        self.transform = transforms.Resize((64, 64))

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()

        # Calculate flattened dim dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_size, self.input_size)
            dummy_output = self.features(dummy_input)
            self.flatten_dim = self.flatten(dummy_output).shape[1]
        

        # Calculate dimensions by hand:
        # resolution /(2^num_downsamples)
        # After resizing to 64x64 and going through the CNN layers:
        # Ex: input 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # Final channels = 256
        # Final feature map 256 * 4 * 4 = 4096
        # TODO clean this up 
        #self.flatten_dim = 128 * 4 * 4 
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
        
        self.classifier = nn.Linear(1024, num_classes)



    def get_embedding(self, x):
        x = self.transform(x)
        x = self.features(x)
        x = self.flatten(x)
        embeddings = self.embedding_layer(x)
        return embeddings

    def forward(self, x):
        embeddings = self.get_embedding(x)
        x = self.classifier(embeddings)
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