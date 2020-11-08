import argparse
import os
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-N', help='number of dropout samples', default=1, type=int)
args = parser.parse_args()

curr_dir = '.'
data_dir = os.path.join(curr_dir, 'data')
checkpoints_dir = os.path.join(curr_dir, 'checkpoints')
figs_dir = os.path.join(curr_dir, 'figs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True,
                                        transform=torchvision.transforms.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False,
                                       transform=torchvision.transforms.ToTensor())

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

dl_train = DataLoader(ds_train, 32, shuffle=True)
dl_test = DataLoader(ds_test, 32, shuffle=True)


class WidthNet1(nn.Module):
    # I Hope it will tend to overfit
    def __init__(self, dropout_class, p=0.5, N=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32 * 32 * 3, 20000),
            nn.ReLU(),
            dropout_class(p),
            nn.Linear(20000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 10)
        )
        self.N = N

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)


N = args.N  # recommended 1 to 20
net = WidthNet1(nn.Dropout).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
epochs = 3
for epoch_num in range(epochs):
    epoch_losses = []
    epochs_corrects = 0
    for batch in dl_train:
        losses = torch.empty(N, requires_grad=False)
        xs, ys = batch
        xs, ys = xs.to(device), ys.to(device)
        xs.requires_grad_()
        for j in range(N):
            ys_pred = net(xs)
            loss = loss_fn(ys_pred, ys)
            losses[j] = loss

        if N > 1:
            losses = torch.softmax(losses, dim=0)
            loss = torch.sum(losses, dim=0)
        else:
            loss = losses[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        correctness = torch.argmax(ys_pred, dim=1) == ys  # ys index random
        epochs_corrects += torch.sum(correctness).detach().item()

    print(f"epoch {epoch_num} acc: {epochs_corrects / len(dl_train)}")
    print(f'epoch {epoch_num} loss: {np.mean(epoch_losses)}')

    plt.plot(epoch_losses)
    plt.savefig(os.path.join(figs_dir, f'epoch_{epoch_num}_batches_improvement'))
    plt.clf()

    # evaluate:
    epoch_losses = []
    corrects = 0
    for batch in dl_test:
        xs, ys = batch
        xs, ys = xs.to(device), ys.to(device)
        ys_pred = net(xs)
        loss = loss_fn(ys_pred, ys)
        correctness = torch.argmax(ys_pred, dim=1) == ys
        corrects += torch.sum(correctness).detach().item()
        epoch_losses.append(loss.detach().item())

    print(f"evaluate loss: {np.mean(epoch_losses)}")
    print(f"evaluate acc: {corrects / len(dl_test)}")
