import torch
from torch import Tensor


class MelLoss(torch.nn.Module):
    def forward(self, spectrogram_audio_pred, spectrogram,
                **batch) -> Tensor:
        return torch.mean(torch.abs(spectrogram_audio_pred[:,:,:spectrogram.shape[2]] - spectrogram))
