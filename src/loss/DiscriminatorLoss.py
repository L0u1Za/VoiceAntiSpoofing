import torch
from torch import Tensor


class DiscriminatorLoss(torch.nn.Module):
    def forward(self, disc_audio_pred, disc_audio,
                **batch) -> Tensor:
        loss = 0.0
        for d, dg in zip(disc_audio, disc_audio_pred):
            loss += (torch.mean((d - 1)**2) + torch.mean(dg**2))
        return loss