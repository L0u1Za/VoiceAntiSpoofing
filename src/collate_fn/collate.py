import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    num_items = len(dataset_items)
    max_audio_length = max([item['audio'].shape[1] for item in dataset_items])
    max_spec_length = 600

    audio, spectrogram = torch.zeros(num_items, max_audio_length), torch.zeros(num_items, dataset_items[0]['spectrogram'].shape[1], max_spec_length)
    duration, audio_path = [], []

    spectrogram_length = torch.tensor([item['spectrogram'].shape[2] for item in dataset_items], dtype=torch.int32)
    label = torch.zeros(num_items, dtype=torch.long)
    for i, item in enumerate(dataset_items):
        audio[i, :item['audio'].shape[1]] = item['audio'].squeeze(0)
        spectrogram[i, :, :min(max_spec_length, item['spectrogram'].shape[2])] = item['spectrogram'].squeeze(0)[:, :min(max_spec_length, item['spectrogram'].shape[2])]
        label[i] = item['label']

        duration.append(item['duration'])
        audio_path.append(item['audio_path'])

    result_batch = {
        "audio": audio,
        "spectrogram": spectrogram,
        "duration": duration,
        "audio_path": audio_path,
        "spectrogram_length": spectrogram_length,
        "label": label
    }
    return result_batch