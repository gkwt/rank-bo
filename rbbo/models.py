from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bayesian_torch.layers import LinearFlipout, LinearReparameterization

import warnings
warnings.simplefilter("ignore", category=Warning)


class OrthoLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


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
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        output = self.layers(x)
        return output
    
    def predict(self, x):
        return self.forward(x), None

    def train_step(self, data, loss_type: str, loss_func: Callable, device):
        if loss_type == 'mse':
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = self.forward(x)
            loss = loss_func(outputs.flatten(), y.flatten())
        elif loss_type == 'ranking':
            x1, x2, y = data
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y1 = self.forward(x1)
            y2 = self.forward(x2)
            loss = loss_func(y1.flatten(), y2.flatten(), y)
        else:
            raise ValueError('Invalid loss type.')
        
        return loss



class BNN(nn.Module):
    def __init__(self, hidden_dim: int = 100, num_layers: int = 1, output_dim: int = 1):
        super(BNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.bayes_layer = LinearReparameterization(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layers(x)
        output, kl = self.bayes_layer(x)
        return output, kl

    def predict(self, x, samples=10):
        preds = [self.forward(x)[0] for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        var = preds.var(axis=0)
        return means, var

    def train_step(self, data, loss_type: str, loss_func, device, beta: float = 0.1):
        if loss_type == 'mse':
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs, kld = self.forward(x)
            loss = loss_func(outputs.flatten(), y.flatten()) + beta * kld / len(y)
        elif loss_type == 'ranking':
            x1, x2, y = data
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y1, kld1 = self.forward(x1)
            y2, kld2 = self.forward(x2)
            loss = loss_func(y1.flatten(), y2.flatten(), y) + beta * (kld1 + kld2) / len(y)
        else:
            raise ValueError('Invalid loss type.')
        
        return loss


def get_loss_function(loss_fn: str):
    if loss_fn == 'mse':
        return nn.MSELoss()
    elif loss_fn == 'ranking':
        return nn.MarginRankingLoss(margin=1.0)
    else:
        raise ValueError('Invalid loss_fn name.')
