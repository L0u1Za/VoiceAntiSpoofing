import torch
from torch import Tensor


class AdvLoss(torch.nn.Module):
    def forward(self, disc_audio_pred,
                **batch) -> Tensor:
        loss = 0.0
        for dg in disc_audio_pred:
            loss += torch.mean((dg - 1)**2)
        return loss
