import logging
from pathlib import Path

import torchaudio

from src.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert "path" in entry
            assert Path(entry["path"]).exists(), f"Path {entry['path']} doesn't exist"
            entry["path"] = str(Path(entry["path"]).absolute().resolve())
            entry["label"] = entry.get("label", "")
            t_info = torchaudio.info(entry["path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)
    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "label" in entry, (
                "Each dataset item should include field 'label'"
                " - bonafide/spoof of the audio."
            )
