"""
    based on https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/d87597d0062580c9ec699193e951e3f4/speech_command_recognition_with_torchaudio.ipynb#scrollTo=RnJv481u1725
"""

import os
import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torch import nn
from models import CommandRecognitionNet
import numpy as np
import matplotlib as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

datafolder = os.path.join('.', 'data')
checkpoints_folder = os.path.join('.', 'checkpoints')


class SC_dataset(SPEECHCOMMANDS):
    def __init__(self, root, train=True):
        super().__init__(root, download=True)
        if train:
            excludes = self.load_list("testing_list.txt") + self.load_list("validation_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        else:
            self._walker = self.load_list("testing_list.txt")

    def load_list(self, filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as fileobj:
            return [os.path.join(self._path, line.strip()) for line in fileobj]


ds_train = SC_dataset(root=datafolder, train=True)
ds_test = SC_dataset(root=datafolder, train=False)

labels = sorted(list(set(datapoint[2] for datapoint in ds_train)))


def label_to_index(word):
    return torch.tensor(labels.index(word))


def index_to_label(index):
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, _, _ in batch:
        tensors += [waveform]
        target = torch.zeros(len(labels))
        target[label_to_index(label)] = 1.
        targets += [target]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


pin_mem = True if device == 'cuda' else False
dl_train = DataLoader(ds_train, 32, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)
dl_test = DataLoader(ds_test, 32, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)

# train process:
net = CommandRecognitionNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
epochs = 3
for epoch_num in range(epochs):
    epoch_losses = []
    for batch in dl_train:
        xs, ys = batch
        xs.requires_grad_()
        ys_pred = net(xs)
        loss = loss_fn(ys_pred, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    print(f'epoch {epoch_num} loss: {np.mean(epoch_losses)}')

# save checkpoint:
checkpoint_path = os.path.join(checkpoints_folder, 'net_checkpoint.pt')
torch.save(net.state_dict(), checkpoint_path)

# evaluate:
epoch_losses = []
for batch in dl_test:
    xs, ys = batch
    ys_pred = net(xs)
    loss = loss_fn(ys, ys_pred)
    epoch_losses.append(loss)

print(np.mean(epoch_losses))
