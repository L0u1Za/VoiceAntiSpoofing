import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://datashare.ed.ac.uk/download/DS_10283_3336.zip"
}


class ASVSpoof2019Dataset(BaseDataset):
    def __init__(self, part, data_dir=None, index_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "asvspoof"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "datasets"
            index_dir.mkdir(exist_ok=True, parents=True)
        else:
            index_dir = Path(index_dir)
        self._index_dir = index_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "DS_10283_3336.zip"
        print(f"Loading ASVSpoof2019")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "DS_10283_3336").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "DS_10283_3336"))

        #files = [file_name for file_name in (self._data_dir / "LA"/ "flac").iterdir()]

        #for i, fpath in enumerate((self._data_dir / "flac").iterdir()):
        #    shutil.move(str(fpath), str(self._data_dir / "data" / fpath.name))
        #shutil.rmtree(str(self._data_dir / "flac"))


    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        prefix = "ASVspoof2019_LA_"

        if not self._data_dir.exists():
            self._load_dataset()

        protocol_path = self._data_dir / "LA" / "LA" / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.{'trn' if part == 'train' else 'trl'}.txt"
        labels = dict()
        with protocol_path.open() as f:
            for line in f:
                _, filename, _, _, label = line.replace('\n', '').split(' ')
                labels[filename] = label

        index = []
        split_dir = self._data_dir / "LA" / "LA" / (prefix + part) / "flac"

        flac_dirs = dict()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs[dirpath] = filenames

        for flac_dir in tqdm(
                list(flac_dirs.keys()), desc=f"Preparing ASVSpoof2019 folders: {part}"
        ):
            for f in flac_dirs[flac_dir]:
                filename = f.split('.')[0]
                flac_dir = Path(flac_dir)
                try:
                    label = labels[filename]
                    flac_path = flac_dir / f"{filename}.flac"
                    if not flac_path.exists(): # elem in another part
                        continue
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "label": label,
                            "audio_len": length,
                        }
                    )
                except:
                    continue
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "label": 0 if data_dict["label"] == 'bonafide\n' else 1,
            "audio_path": audio_path,
        }

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