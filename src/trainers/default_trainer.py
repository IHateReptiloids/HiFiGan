from collections import OrderedDict
from itertools import chain
from pathlib import Path

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


def fm_loss(fake_acts, real_acts):
    '''
    fake_acts is List[List[Tensor]] with feature maps
    so is real_acts
    '''
    total_loss = 0
    fake_acts = tuple(chain(*fake_acts))
    real_acts = tuple(chain(*real_acts))
    assert len(fake_acts) == len(real_acts)
    for fake, real in zip(fake_acts, real_acts):
        assert fake.shape == real.shape
        total_loss += F.l1_loss(fake, real)
    return total_loss


def lsgan_d_loss(fake_out, real_out):
    '''
    fake_out is List[Tensor]
    so is real_out
    '''
    total_loss = 0
    assert len(fake_out) == len(real_out)
    for fake, real in zip(fake_out, real_out):
        assert fake.shape == real.shape
        total_loss += F.mse_loss(fake, torch.zeros_like(fake))
        total_loss += F.mse_loss(real, torch.ones_like(real))
    return total_loss


def lsgan_g_loss(fake_out):
    '''
    fake_out is List[Tensor]
    '''
    total_loss = 0
    for fake in fake_out:
        total_loss += F.mse_loss(fake, torch.ones_like(fake))
    return total_loss


def melspec_loss(fake, real):
    return F.l1_loss(fake, real)


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
        out_specs = self.wav2spec(out_wavs)
        if train:
            assert out_specs.shape == specs.shape
            assert out_wavs.shape == wavs.shape
            self.d.unfreeze()
            self.opt_d.zero_grad()

        mpd_real_outs, _ = self.d.mpd(wavs)
        mpd_fake_outs, _ = self.d.mpd(out_wavs.detach())
        mpd_loss = lsgan_d_loss(mpd_fake_outs, mpd_real_outs)

        msd_real_outs, _ = self.d.msd(wavs)
        msd_fake_outs, _ = self.d.msd(out_wavs.detach())
        msd_loss = lsgan_d_loss(msd_fake_outs, msd_real_outs)

        d_loss = (mpd_loss + msd_loss) * self.config.gan_coef
        if train:
            d_loss.backward()
            self.opt_d.step()
            self.scheduler_d.step()
            self.d.freeze()

        mel_loss = melspec_loss(out_specs, specs) * self.config.mel_coef

        mpd_real_outs, mpd_real_acts = self.d.mpd(wavs)
        mpd_fake_outs, mpd_fake_acts = self.d.mpd(out_wavs)
        msd_real_outs, msd_real_acts = self.d.msd(wavs)
        msd_fake_outs, msd_fake_acts = self.d.msd(out_wavs)

        fm_loss_ = fm_loss(mpd_fake_acts, mpd_real_acts)
        fm_loss_ += fm_loss(msd_fake_acts, msd_real_acts)
        fm_loss_ *= self.config.fm_coef

        mpd_loss = lsgan_g_loss(mpd_fake_outs) * self.config.gan_coef
        msd_loss = lsgan_g_loss(msd_fake_outs) * self.config.gan_coef

        g_loss = mel_loss + fm_loss_ + mpd_loss + msd_loss
        if train:
            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()
            self.scheduler_g.step()

        total_loss = d_loss.item() + g_loss.item()
        return total_loss, out_specs.detach().cpu(), out_wavs.detach().cpu()

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
