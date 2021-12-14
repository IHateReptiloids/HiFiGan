from pathlib import Path

import numpy as np
import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(
        self,
        segment_size,
        root='data/lj_speech',
        indices_path=None,
        device=torch.device('cpu')
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root=root, download=True)

        self.segment_size = segment_size
        self._indices = None
        if indices_path is not None:
            self._indices = np.loadtxt(indices_path, dtype=np.int32)
        self.device = device

        self._cache = {}

    def __getitem__(self, index: int):
        if self._indices is not None:
            index = self._indices[index]

        wav = self._cache.get(index)
        if wav is None:
            wav, _, _, _ = super().__getitem__(index)
            wav = wav.to(self.device).squeeze()
            self._cache[index] = wav

        wav = self._cut(wav)
        return wav

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return super().__len__()

    def _cut(self, wav):
        if self.segment_size is None or len(wav) <= self.segment_size:
            return wav
        start_ind = torch.randint(len(wav) - self.segment_size, (1,)).item()
        return wav[start_ind:start_ind + self.segment_size]
