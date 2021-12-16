import torch.nn as nn

from .mpd import MPD
from .msd import MSD


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mpd = MPD(config)
        self.msd = MSD(config)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
