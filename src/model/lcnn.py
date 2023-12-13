from torch import nn
import torch
from src.base.base_model import BaseModel

class MFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, inputs):
        outputs = inputs.split(self.channels, 1)
        outputs = torch.max(outputs[0], outputs[1])
        return outputs

class LCNN(BaseModel):
    def __init__(self):
        super().__init__()

        self.feats = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                MFM(32),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),
                MFM(32),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                MFM(48),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(48),
            ),
            nn.Sequential(
                nn.Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1)),
                MFM(48),
                nn.BatchNorm2d(48),
                nn.Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                MFM(64),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),
                MFM(64),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                MFM(32),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),
                MFM(32),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                MFM(32),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        ])
        self.fc = nn.Sequential(
            nn.Linear(53 * 37 * 32, 160),
            MFM(80),
            nn.Dropout(0.75),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2)
        )

    def forward(self, spectrogram, **batch):
        print(spectrogram.shape)
        outputs = spectrogram.unsqueeze(1)
        for layer in self.feats:
            outputs = layer(outputs)
        outputs = outputs.flatten(1, -1)
        outputs = self.fc(outputs)
        return outputs