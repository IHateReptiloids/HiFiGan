from argparse_dataclass import ArgumentParser
from matplotlib import pyplot as plt
import torch
import torchaudio
import wandb

from src.configs import DefaultConfig
from src.models import Generator
from src.utils import seed_all
from src.wav2spec import Wav2Spec


def normalize_img(img):
    img -= img.min()
    return img / img.max()


config = ArgumentParser(DefaultConfig).parse_args()
config.device = torch.device('cpu')
wandb.init(config=config)
seed_all(config.random_seed)

wavs = []
for fname in ['audio_1.wav', 'audio_2.wav', 'audio_3.wav']:
    wav, sr = torchaudio.load(fname)
    wavs.append(wav.to(config.device))

model = Generator(config).to(config.device)
assert config.wandb_file_name is not None and \
       config.wandb_run_path is not None
f = wandb.restore(config.wandb_file_name, config.wandb_run_path,
                  root=config.checkpoints_path)
sd = torch.load(f.name, map_location=config.device)
model.load_state_dict(sd['generator'])
model.eval()

table = wandb.Table(columns=['gt_spec', 'gt_wav', 'out_spec', 'out_wav'])
wav2spec = Wav2Spec(config)
viridis = plt.get_cmap('viridis')
for wav in wavs:
    spec = wav2spec(wav)
    with torch.no_grad():
        out_wav = model(spec).squeeze(1)
    out_spec = wav2spec(out_wav)

    gt_spec = wandb.Image(viridis(normalize_img(spec.squeeze())))
    out_spec = wandb.Image(viridis(normalize_img(out_spec.squeeze())))

    gt_wav = wandb.Audio(wav.squeeze(), sample_rate=config.sample_rate)
    out_wav = wandb.Audio(out_wav.squeeze(), sample_rate=config.sample_rate)
    table.add_data(gt_spec, gt_wav, out_spec, out_wav)
wandb.log({'results': table})
