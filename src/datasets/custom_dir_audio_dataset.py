import logging
from pathlib import Path

from src.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + '.txt')
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["label"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "label": 0 if data_dict["label"] == 'bonafide' else 1,
            "audio_path": audio_path,
        }
