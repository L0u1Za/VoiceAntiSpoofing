import torch
from torch import Tensor
import torch.nn.functional as F

class ASoftmax(torch.nn.Module):
    def __init__(self, m, **args):
        super().__init__()

        self.m = m
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, label, W,
                **batch) -> Tensor:

        x = prediction
        W = F.normalize(W, p=2, dim=1)
        x_norm = torch.norm(x, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        tetta = torch.arccos(x * W)
        x = x_norm * torch.cos(self.m * tetta)
        return self.loss(x, label)