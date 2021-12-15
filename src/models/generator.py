import torch.nn as nn
from torch.nn.utils import weight_norm

from .utils import init_conv, pairwise


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []

        prenet = nn.Conv1d(config.n_mels, config.gen_channels[0],
                           config.gen_prenet_ksize, padding='same')
        init_conv(prenet)
        layers.append(weight_norm(prenet))

        assert config.gen_n_layers == len(config.gen_channels) - 1 == \
            len(config.gen_ksizes) == len(config.gen_paddings) == \
            len(config.gen_strides)
        for (in_c, out_c), ksize, padding, stride in zip(
            pairwise(config.gen_channels),
            config.gen_ksizes,
            config.gen_paddings,
            config.gen_strides
        ):
            layers.append(nn.LeakyReLU(config.relu_slope))
            conv = nn.ConvTranspose1d(in_c, out_c, ksize, stride, padding)
            init_conv(conv)
            layers.append(weight_norm(conv))
            layers.append(MRF(out_c, config))

        layers.append(nn.LeakyReLU(config.relu_slope))
        postnet = nn.Conv1d(config.gen_channels[-1], 1,
                            config.gen_postnet_ksize, padding='same')
        init_conv(postnet)
        layers.append(weight_norm(postnet))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MRF(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        assert len(config.resblock_dilations) == len(config.resblock_ksizes)
        self.res_blocks = nn.ModuleList()
        for dilations, kernel_size in zip(config.resblock_dilations,
                                          config.resblock_ksizes):
            self.res_blocks.append(
                ResBlock(channels, config.relu_slope,
                         kernel_size, dilations)
            )

    def forward(self, x):
        sum_ = 0
        for res_block in self.res_blocks:
            sum_ = sum_ + res_block(x)
        return sum_ / len(self.res_blocks)


class ResBlock(nn.Module):
    def __init__(self, channels, relu_slope, kernel_size, dilations):
        super().__init__()

        self.convs = nn.ModuleList()
        # dilations is Tuple[Tuple[int]]
        for dilations_ in dilations:
            layers = []
            for dilation in dilations_:
                layers.append(nn.LeakyReLU(relu_slope))
                conv = nn.Conv1d(channels, channels, kernel_size,
                                 padding='same', dilation=dilation)
                init_conv(conv)
                layers.append(weight_norm(conv))
            self.convs.append(nn.Sequential(*layers))

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x
