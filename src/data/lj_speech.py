from pathlib import Path

import numpy as np
import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(
        self,
        segment_size,
        root='data/lj_speech',
        indices_path=None
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root=root, download=True)

        self.segment_size = segment_size
        if indices_path is not None:
            indices = np.loadtxt(indices_path, dtype=np.int32)
        else:
            indices = np.arange(super().__len__())

        self._data = []
        for index in indices:
            wav, _, _, _ = super().__getitem__(index)
            self._data.append(wav.squeeze())

    def __getitem__(self, index: int):
        wav = self._data[index]
        return self._cut(wav)

    def __len__(self):
        return len(self._data)

    def _cut(self, wav):
        if self.segment_size is None or len(wav) <= self.segment_size:
            return wav
        start_ind = torch.randint(len(wav) - self.segment_size, (1,)).item()
        return wav[start_ind:start_ind + self.segment_size]
