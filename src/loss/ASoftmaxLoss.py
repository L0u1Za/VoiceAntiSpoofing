import torch
from torch import Tensor

class FinalLoss(torch.nn.Module):
    def __init__(self, m=1.0, **args):
        super().__init__()

        self.m = m

    def forward(self, prediction, label,
                **batch) -> Tensor:


        return 0.0