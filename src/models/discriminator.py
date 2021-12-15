import torch.nn as nn

from .mpd import MPD
from .msd import MSD


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mpd = MPD(config)
        self.msd = MSD(config)
