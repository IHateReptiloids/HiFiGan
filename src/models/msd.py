import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

from .utils import pairwise


class MSD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sds = nn.ModuleList()
        for scale in config.msd_scales:
            self.sds.append(ScaleDiscriminator(scale, config))

    def forward(self, x):
        activations = []
        outputs = []
        for sd in self.sds:
            output, activations_ = sd(x)
            activations.append(activations_)
            outputs.append(output)
        return outputs, activations


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale, config):
        super().__init__()

        self.pool = nn.Identity()
        if scale > 1:
            self.pool = nn.AvgPool1d(2 * scale, scale, padding=scale)
        self.relu_slope = config.sd_relu_slope

        self.convs = nn.ModuleList()
        norm_func = spectral_norm if scale == 1 else weight_norm
        channels = [1] + list(config.sd_channels)
        assert len(config.sd_channels) == len(config.sd_groups) == \
            len(config.sd_ksizes) == len(config.sd_paddings) == \
            len(config.sd_strides)
        for (in_c, out_c), groups, ksize, stride in zip(
            pairwise(channels),
            config.sd_groups,
            config.sd_ksizes,
            config.sd_strides
        ):
            self.convs.append(norm_func(
                nn.Conv1d(in_c, out_c, ksize, stride,
                          padding='same', groups=groups)
            ))

        self.proj = norm_func(nn.Conv1d(channels[-1], 1, config.sd_proj_ksize,
                                        config.sd_proj_stride, padding='same'))

    def forward(self, x):
        '''
        x is of shape (bs, time)
        '''
        assert x.dim() == 2
        x = x.unsqueeze(1)
        activations = []
        x = self.pool(x)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.relu_slope)
            activations.append(x)
        x = self.proj(x)
        activations.append(x)
        return x.flatten(1, -1), activations
