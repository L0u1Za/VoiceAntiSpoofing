import torch
from torch import Tensor

from src.loss.AdvLoss import AdvLoss
from src.loss.FeatureLoss import FeatureLoss
from src.loss.MelLoss import MelLoss
from src.base.base_mel import MelSpectrogram

class GeneratorLoss(torch.nn.Module):
    def __init__(self, alpha_fm, alpha_mel, **args):
        super().__init__()
        self.alpha_fm = alpha_fm
        self.alpha_mel = alpha_mel

        self.adv = AdvLoss()
        self.feat = FeatureLoss()
        self.mel = MelLoss()

        self.mel_transform = MelSpectrogram()

    def forward(self, audio_pred, spectrogram, disc_audio_pred, feat_pred, feat_target,
                **batch) -> Tensor:

        spectrogram_audio_pred = self.mel_transform(audio_pred)
        return self.adv(disc_audio_pred) + self.alpha_fm * self.feat(feat_target, feat_pred) + self.alpha_mel * self.mel(spectrogram_audio_pred, spectrogram)
