import torch
from torch import Tensor

from src.loss.DiscriminatorLoss import DiscriminatorLoss
from src.loss.GeneratorLoss import GeneratorLoss

class FinalLoss(torch.nn.Module):
    def __init__(self, alpha_fm, alpha_mel, **args):
        super().__init__()

        self.gen_loss = GeneratorLoss(alpha_fm, alpha_mel)
        self.disc_loss = DiscriminatorLoss()

    def forward(self, audio_pred, spectrogram, disc_audio_pred, disc_audio, feat_pred, feat_target,
                **batch) -> Tensor:

        return self.gen_loss(audio_pred, spectrogram, disc_audio_pred, feat_pred, feat_target), self.disc_loss(disc_audio_pred, disc_audio)