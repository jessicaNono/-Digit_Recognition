import torch

import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # 784 inputs to 128 outputs
        self.fc2 = nn.Linear(128, 64)  # 128 inputs to 64 outputs
        self.fc3 = nn.Linear(64, 10)   # 64 inputs to 10 outputs (for 10 classes)

    def forward(self, x):
        # Define forward pass
        x = F.relu(self.fc1(x))  # Apply ReLU to layer 1
        x = F.relu(self.fc2(x))  # Apply ReLU to layer 2
        x = self.fc3(x)          # Output layer
        return F.log_softmax(x, dim=1)  # Apply log softmax to the outputs
