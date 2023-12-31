import torch
from torch import Tensor

class FinalLoss(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, label,
                **batch) -> Tensor:

        return self.loss(prediction, label)