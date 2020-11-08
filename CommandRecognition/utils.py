import torch


def divide_signals_to_patches(signals, patch_size):
    if len(signals.shape) == 1:
        signals = torch.unsqueeze(signals, dim=1)
    assert signals.shape[1] % patch_size == 0
    width = signals.shape[1] // patch_size
    height = patch_size
    out = torch.empty(signals.shape[0], width, height)  # saves grads
    c = 0
    for i in range(height):
        out[:, :, i] = signals[:, c:c + width]
        c += width

    return out
