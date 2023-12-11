from torch.utils.data import Sampler
import torch

class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        assert batch_size >= 2

        super().__init__(data_source)
        self.data_source = data_source
        self.length = 0
        self.batch_size = batch_size

    def __iter__(self):
        length = 0
        bonafide, spoof = [], []
        indices = torch.randperm(len(self.data_source)).tolist()
        for idx in indices:
            s = self.data_source[idx]
            if s["label"] == 0:
                bonafide.append(idx)
            else:
                spoof.append(idx)

            while (len(bonafide) != 0 and len(spoof) != 0) and (len(bonafide) + len(spoof) >= self.batch_size):
                if (len(bonafide) < len(spoof)):
                    batch = bonafide + spoof[:self.batch_size - len(bonafide)]
                    if self.batch_size - len(bonafide):
                        spoof = spoof[self.batch_size - len(bonafide):]
                else:
                    batch = bonafide[:self.batch_size - len(spoof)] + spoof
                    if self.batch_size - len(spoof):
                        bonafide = bonafide[self.batch_size - len(spoof):]
                yield batch
                length += 1
                batch = []

        self.length = length

    def __len__(self):
        if self.length:
            return self.length
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size