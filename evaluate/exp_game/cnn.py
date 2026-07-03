import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, emb_size):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.emb_size = emb_size

        layers = [
            nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2), # (64, 64) -> (32, 32)
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2), # 32 -> 16
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2), # 16 -> 8
            
            nn.Conv2d(32, emb_size, kernel_size=3, stride=1, padding=1),
        ]
        
        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):
        # TODO: compute forward pass
        x = self.model(x)
        return x