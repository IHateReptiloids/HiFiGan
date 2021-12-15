from pathlib import Path

from argparse_dataclass import ArgumentParser
import torch
from torchinfo import summary
import wandb

from src.configs import DefaultConfig
from src.data import collate, LJSpeechDataset
from src.models import Discriminator, Generator
from src.trainers import DefaultTrainer
from src.utils import seed_all
from src.wav2spec import Wav2Spec


TRAIN_INDICES = Path('data/lj_speech/train_indices.txt')
VAL_INDICES = Path('data/lj_speech/val_indices.txt')


config = ArgumentParser(DefaultConfig).parse_args()
seed_all(config.random_seed)


train_ds = LJSpeechDataset(config.segment_size, config.data_path,
                           TRAIN_INDICES, config.device)

val_ds = LJSpeechDataset(None, config.data_path,
                         VAL_INDICES, config.device)

train_loader = torch.utils.data.DataLoader(train_ds, config.train_batch_size,
                                           shuffle=True, collate_fn=collate,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, config.val_batch_size,
                                         shuffle=True, collate_fn=collate)

g = Generator(config).to(config.device)
summary(g)
d = Discriminator(config).to(config.device)
summary(d)
opt_g = torch.optim.AdamW(g.parameters(), lr=config.initial_lr,
                          betas=config.betas,
                          weight_decay=config.weight_decay)
opt_d = torch.optim.AdamW(d.parameters(), lr=config.initial_lr,
                          betas=config.betas,
                          weight_decay=config.weight_decay)

epoch_num_iters = len(train_ds) // config.train_batch_size
scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, epoch_num_iters,
                                              config.lr_decay)
scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, epoch_num_iters,
                                              config.lr_decay)

wav2spec = Wav2Spec(config)

wandb.init(config=config)
wandb.watch((d, g), log='all', log_freq=config.train_log_freq)

trainer = DefaultTrainer(
    config,
    g,
    d,
    opt_g,
    opt_d,
    scheduler_g,
    scheduler_d,
    wav2spec,
    train_loader,
    None
)
trainer.train(config.num_epochs)
