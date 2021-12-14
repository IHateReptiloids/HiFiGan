import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stub = nn.Linear(5, 5)
