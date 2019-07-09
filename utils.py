import torch
from torch import nn


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x)
        b = -1.0 * b.sum(dim=1).mean()
        return b
