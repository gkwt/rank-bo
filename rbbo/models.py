import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Standard fully connected multi-layer perceptron.
    Only allows training with 
    """
    def __init__(self, hidden_dim: int = 100, num_layers: int = 3, output_dim: int = 1):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
        for _ in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        output = self.layers(x)
        return output

# class 