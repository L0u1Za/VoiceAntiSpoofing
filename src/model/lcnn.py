from torch import nn
import torch

class MFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, inputs):
        outputs = inputs.split(self.channels, 1)
        outputs = torch.max(outputs[0], outputs[1])
        return outputs

class FrontEnd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        outputs = torch.stft(inputs, n_fft=1724, win_length=1724, return_complex=False)
        return outputs

class LCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.front = FrontEnd()

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
            nn.Dropout(0.75),
            MFM(80),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2)
        )

    def forward(self, audio, spectrogram, **batch):
        outputs = spectrogram.unsqueeze(1)
        for layer in self.feats:
            print(outputs.shape)
            outputs = layer(outputs)
        print(outputs.shape)
        outputs = outputs.flatten(1, -1)
        print(outputs.shape)
        outputs = self.fc(outputs)
        print(outputs.shape)
        return outputs