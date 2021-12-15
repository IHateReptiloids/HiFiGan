import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .utils import pairwise


class MPD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pds = nn.ModuleList()
        for period in config.mpd_periods:
            self.pds.append(PeriodDiscriminator(period, config))

    def forward(self, x):
        activations = []
        outputs = []
        for pd in self.pds:
            output, activations_ = pd(x)
            activations.append(activations_)
            outputs.append(output)
        return outputs, activations


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, config):
        super().__init__()
        self.period = period
        self.relu_slope = config.pd_relu_slope
        self.convs = nn.ModuleList()

        channels = [1] + list(config.pd_channels)
        for in_c, out_c in pairwise(channels):
            conv = nn.Conv2d(in_c, out_c, (config.pd_ksize, 1),
                             (config.pd_stride, 1), (config.pd_padding, 0))
            self.convs.append(weight_norm(conv))

        postnet = nn.Conv2d(channels[-1], config.pd_postnet_channels,
                            (config.pd_ksize, 1), 1, (config.pd_padding, 0))
        self.convs.append(weight_norm(postnet))

        self.proj = weight_norm(nn.Conv2d(config.pd_postnet_channels, 1,
                                          (config.pd_proj_ksize, 1), 1,
                                          (config.pd_proj_padding, 0)))

    def forward(self, x):
        '''
        x is of shape (bs, time)
        '''
        assert x.dim() == 2
        if x.shape[1] % self.period != 0:
            x = F.pad(x, (0, self.period - (x.shape[1] % self.period)))
        x = x.view(x.shape[0], 1, x.shape[1] // self.period, self.period)

        activations = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.relu_slope)
            activations.append(x)
        x = self.proj(x)
        activations.append(x)
        return x.flatten(1, -1), activations
