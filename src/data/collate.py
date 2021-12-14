from torch.nn.utils.rnn import pad_sequence


def collate(objects):
    return pad_sequence(objects, batch_first=True)
