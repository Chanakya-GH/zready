import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        self.net = nn.Sequential(
            nn.Linear(29, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):                      # returns logits
        return self.net((x - self.mean) / self.std).squeeze(-1)
