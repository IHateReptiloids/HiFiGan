from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DefaultConfig:
    # Meta params
    checkpointing_freq: int = 1000
    checkpoints_path: str = 'checkpoints/'
    data_path: str = 'data/lj_speech/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed: int = 3407
    wandb_file_name: str = None
    wandb_run_path: str = None
    # AdamW params
    betas: Tuple[float] = (0.8, 0.99)
    initial_lr: float = 2e-4
    lr_decay: float = 0.999
    weight_decay: float = 0.01
    # Model params
    channels: Tuple[int] = (512, 256, 128, 64, 32)
    ksizes: Tuple[int] = (16, 16, 4, 4)
    n_layers: int = 4
    paddings: Tuple[int] = (4, 4, 1, 1)
    postnet_ksize: int = 7
    prenet_ksize: int = 7
    relu_slope: float = 0.1
    resblock_dilations: Tuple[Tuple[Tuple[int]]] = \
        (((1, 1), (3, 1), (5, 1)),) * 3
    resblock_ksizes: Tuple[int] = (3, 7, 11)
    strides: Tuple[int] = (8, 8, 2, 2)
    # MelSpec params
    f_min: int = 0
    f_max: int = 8000
    hop_length: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    pad_mode: str = 'constant'
    power: float = 1.0
    sample_rate: int = 22050
    segment_size: int = 8192
    win_length: int = 1024
    # Training params
    num_epochs: int = 1000
    train_batch_size: int = 16
    train_log_freq: int = 3000
    val_batch_size: int = 4
    val_log_freq: int = 40
