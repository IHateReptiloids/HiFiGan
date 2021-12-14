from collections import OrderedDict
from pathlib import Path

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


def normalize_img(img):
    img -= img.min()
    return img / img.max()


class DefaultTrainer:
    def __init__(
        self,
        config,
        g,
        d,
        opt_g,
        opt_d,
        scheduler_g,
        scheduler_d,
        wav2spec,
        train_loader,
        val_loader
    ):
        self.config = config
        self.g = g
        self.d = d
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.wav2spec = wav2spec
        self.train_loader = train_loader
        self.val_loader = val_loader

        self._checkpointing_freq = config.checkpointing_freq
        self._train_log_age = 0
        self._train_log_freq = config.train_log_freq
        self._val_log_age = 0
        self._val_log_freq = config.val_log_freq

        self._checkpoints_path = Path(config.checkpoints_path)

        if (
            config.wandb_file_name is not None and
            config.wandb_run_path is not None
        ):
            f = wandb.restore(config.wandb_file_name, config.wandb_run_path,
                              root=config.checkpoints_path)
            sd = torch.load(f.name, map_location=config.device)
            sd['opt_d']['param_groups'][0]['initial_lr'] = config.initial_lr
            sd['opt_g']['param_groups'][0]['initial_lr'] = config.initial_lr
            sd['scheduler_d']['base_lrs'] = [config.initial_lr]
            sd['scheduler_g']['base_lrs'] = [config.initial_lr]
            self.load_state_dict(sd)

    def load_state_dict(self, sd):
        self.d.load_state_dict(sd['discriminator'])
        self.g.load_state_dict(sd['generator'])
        self.opt_d.load_state_dict(sd['opt_d'])
        self.opt_g.load_state_dict(sd['opt_g'])
        self.scheduler_d.load_state_dict(sd['scheduler_d'])
        self.scheduler_g.load_state_dict(sd['scheduler_g'])

    def process_batch(self, specs, wavs, train: bool):
        out_wavs = self.g(specs).squeeze(dim=1)
        out_specs = self.wav2spec(out_wavs).squeeze(dim=1)
        if train:
            assert out_specs.shape == specs.shape
            assert out_wavs.shape == wavs.shape

        loss = F.l1_loss(out_specs, specs) * 45.0
        if train:
            self.opt_g.zero_grad()
            loss.backward()
            self.opt_g.step()
            self.scheduler_g.step()

            self.scheduler_d.step()

        return loss.item(), out_specs.detach().cpu(), out_wavs.detach().cpu()

    def state_dict(self):
        sd = OrderedDict()
        sd['discriminator'] = self.d.state_dict()
        sd['generator'] = self.g.state_dict()
        sd['opt_d'] = self.opt_d.state_dict()
        sd['opt_g'] = self.opt_g.state_dict()
        sd['scheduler_d'] = self.scheduler_d.state_dict()
        sd['scheduler_g'] = self.scheduler_g.state_dict()
        return sd

    def train(self, num_epochs):
        for i in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            print(f'Epoch {i} train loss: {train_loss}')
            if self.val_loader is not None:
                val_loss = self.validation()
                print(f'Epoch {i} validation loss: {val_loss}')
            if i % self._checkpointing_freq == 0:
                assert self.scheduler_d.last_epoch == \
                    self.scheduler_g.last_epoch
                fname = f'state{self.scheduler_d.last_epoch}.pth'
                path = self._checkpoints_path / fname
                print(f'Saving checkpoint after epoch {i} to {str(path)}')
                torch.save(self.state_dict(), path)
                wandb.save(path)
            print('-' * 100)

    def train_epoch(self):
        self.g.train()
        self.d.train()
        total_loss = 0
        for wavs in tqdm(self.train_loader):
            specs = self.wav2spec(wavs)
            batch_loss, out_specs, out_wavs = self.process_batch(specs, wavs,
                                                                 train=True)
            total_loss += batch_loss

            self._train_log_age += 1
            wandb_data = {
                'train/g_loss': batch_loss,
                'train/g_lr': self.scheduler_g.get_last_lr()[0]
            }
            if self._train_log_age == self._train_log_freq:
                self._train_log_age = 0
                gt_spec, gt_wav, out_spec, out_wav = self._log_audio(
                    specs.cpu(), wavs.cpu(), out_specs, out_wavs
                )
                wandb_data.update({
                    'train/gt_spec': gt_spec,
                    'train/gt_wav': gt_wav,
                    'train/out_spec': out_spec,
                    'train/out_wav': out_wav
                })
            assert self.scheduler_d.last_epoch == self.scheduler_g.last_epoch
            wandb.log(wandb_data, step=self.scheduler_d.last_epoch)
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validation(self):
        self.g.eval()
        self.d.eval()
        total_loss = 0
        table = wandb.Table(columns=['gt_spec', 'gt_wav',
                                     'out_spec', 'out_wav'])
        for wavs in tqdm(self.val_loader):
            specs = self.wav2spec(wavs)
            batch_loss, out_specs, out_wavs = self.process_batch(specs, wavs,
                                                                 train=False)
            total_loss += batch_loss

            self._val_log_age += 1
            if self._val_log_age == self._val_log_freq:
                table.add_data(*self._log_audio(specs.cpu(), wavs.cpu(),
                                                out_specs, out_wavs))
        wandb_data = {
            'val/table': table,
            'val/loss': total_loss / len(self.val_loader)
        }
        assert self.scheduler_d.last_epoch == self.scheduler_g.last_epoch
        wandb.log(wandb_data, step=self.scheduler_d.last_epoch)
        return total_loss / len(self.val_loader)

    def _log_audio(self, gt_specs, gt_wavs, out_specs, out_wavs):
        assert len(gt_specs) == len(gt_wavs) == len(out_specs) == len(out_wavs)
        index = torch.randint(len(gt_specs), (1,)).item()

        viridis = plt.get_cmap('viridis')
        gt_spec = wandb.Image(viridis(normalize_img(gt_specs[index])))
        out_spec = wandb.Image(viridis(normalize_img(out_specs[index])))

        gt_wav = wandb.Audio(gt_wavs[index],
                             sample_rate=self.config.sample_rate)
        out_wav = wandb.Audio(out_wavs[index],
                              sample_rate=self.config.sample_rate)

        return gt_spec, gt_wav, out_spec, out_wav
