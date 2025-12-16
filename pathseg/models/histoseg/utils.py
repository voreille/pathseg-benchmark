import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, ps: int):
    B, C, H, W = x.shape
    Hp = (H + ps - 1) // ps * ps
    Wp = (W + ps - 1) // ps * ps
    pad = (0, Wp - W, 0, Hp - H)  # (left,right,top,bottom)
    x_pad = F.pad(x, pad, mode='reflect')
    return x_pad, pad


def unpad_logits(y: torch.Tensor, pad):
    _, r, _, b = 0, pad[1], 0, pad[3]
    if r or b:
        return y[..., :y.shape[-2] - b, :y.shape[-1] - r]
    return y
