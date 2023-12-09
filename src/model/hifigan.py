from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def get_pad(kernel, dilation):
    return (kernel*dilation - dilation) // 2

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilations[l], padding=get_pad(kernel_size, dilations[l])))
            ) for l in range(len(dilations))
        ])
    def forward(self, inputs):
        res = inputs
        output = inputs
        for layer in self.net:
            output = layer(output)
        output += res
        return output

class MultiReceptiveField(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()

        self.net = nn.ModuleList([
            ResBlock(channels=channels, kernel_size=kernel_size[n], dilations=dilations[n]) for n in range(len(dilations))
        ])

    def forward(self, inputs):
        outputs = torch.zeros(inputs.shape).to(inputs.device)
        for layer in self.net:
            outputs += layer(inputs)
        return outputs

class Generator(nn.Module):
    def __init__(self, h_u, k_u, k_r, d_r):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(80, h_u, kernel_size=7, stride=1, dilation=1, padding=3))
        self.upsampling = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(nn.ConvTranspose1d(h_u // 2 ** l, h_u // 2 ** (l + 1), kernel_size=k_u[l], stride=k_u[l] // 2, padding=(k_u[l]- k_u[l] // 2)//2)),
                MultiReceptiveField(h_u // 2 ** (l + 1), k_r, d_r)
            ) for l in range(len(k_u))
        ])
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(nn.Conv1d(h_u // 2 ** (len(k_u)), 1, kernel_size=7, stride=1, padding=3)),
            nn.Tanh()
        )

    def forward(self, inputs, **batch):
        outputs = self.conv1(inputs)
        for layer in self.upsampling:
            outputs = layer(outputs)
        outputs = self.conv2(outputs)
        return outputs

class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256)),
                nn.LeakyReLU()
            )
        ])

        self.conv2 = nn.Sequential(
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1))
        )

    def forward(self, inputs):
        outputs = inputs
        feats = []
        for layer in self.conv1:
            outputs = layer(outputs)
            feats.append(outputs)
        outputs = self.conv2(outputs)
        feats.append(outputs)
        return outputs.squeeze(1), feats

class NothingModule(nn.Module):
    def forward(self, inputs):
        return inputs

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.desc = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator()
        ])
        self.pooling = nn.ModuleList([
            NothingModule(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, target, pred):
        out_target = []
        out_pred = []
        feat_target = []
        feat_pred = []
        for i, layer in enumerate(self.desc):
            target, pred = self.pooling[i](target), self.pooling[i](pred)
            o_t, f_t = layer(target)
            out_target.append(o_t), feat_target.append(f_t)
            o_p, f_p = layer(pred)
            out_pred.append(o_p), feat_pred.append(f_p)
        return out_target, out_pred, feat_target, feat_pred

class PeriodDiscriminator(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.period = p
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv2d(1, 2 ** 5, kernel_size=(5, 1), stride=(3, 1), padding=(get_pad(5, 1), 0))),
                nn.LeakyReLU()
            )] +
            [
            nn.Sequential(
                weight_norm(nn.Conv2d(2 ** (5 + l - 1), 2 ** (5 + l), kernel_size=(5, 1), stride=(3, 1), padding=(get_pad(5, 1), 0))),
                nn.LeakyReLU()
            ) for l in range(1, 5)
        ])
        self.conv2 = nn.Sequential(
            weight_norm(nn.Conv2d(2 ** (5 + 4), 1024, kernel_size=(5, 1), padding=(get_pad(5, 1), 0))),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(get_pad(5, 1), 0)))
        )
    def pad_and_reshape(self, outputs):
        b, c, t = outputs.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            outputs = F.pad(outputs, (0, n_pad), "reflect")
            t = t + n_pad
        return outputs.view(b, c, t // self.period, self.period)

    def forward(self, inputs):
        outputs = self.pad_and_reshape(inputs)
        feats = []
        for layer in self.conv1:
            outputs = layer(outputs)
            feats.append(outputs)
        outputs = self.conv2(outputs)
        feats.append(outputs)
        return outputs.squeeze(1), feats

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.desc = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11)
        ])

    def forward(self, target, pred):
        out_target = []
        out_pred = []
        feat_target = []
        feat_pred = []
        for layer in self.desc:
            o_t, f_t = layer(target)
            out_target.append(o_t), feat_target.append(f_t)
            o_p, f_p = layer(pred)
            out_pred.append(o_p), feat_pred.append(f_p)
        return out_target, out_pred, feat_target, feat_pred

class HiFiGAN(nn.Module):
    def __init__(self, h_u, k_u, k_r, d_r):
        super().__init__()

        self.generator = Generator(h_u, k_u, k_r, d_r)

        self.discriminator1 = MultiPeriodDiscriminator()
        self.discriminator2 = MultiScaleDiscriminator()

    def forward(self, spectrogram, audio, **batch):
        pred = self.generator(spectrogram)
        out_desc1 = self.discriminator1(audio.unsqueeze(1), pred)
        out_desc2 = self.discriminator2(audio.unsqueeze(1), pred)
        return pred.squeeze(1), out_desc1, out_desc2
