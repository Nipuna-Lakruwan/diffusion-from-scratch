"""A small U-Net that predicts the noise in a noisy image.

Input:  x_t (B, 1, 32, 32), timestep t (B,), optional class label y (B,)
Output: predicted noise eps_theta(x_t, t, y), same shape as x_t.

Label 10 is the "no label" token, used for classifier-free guidance.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the integer timestep, as in the Transformer paper."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]  # tell every pixel "which step we're at"
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.norm(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        out, _ = self.attn(s, s, s, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, ch=32, mults=(1, 2, 2), num_classes=10, attn_res=(8,)):
        super().__init__()
        emb_dim = ch * 4
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes + 1, emb_dim) if num_classes else None

        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans = [ch]
        c, res = ch, 32
        for i, m in enumerate(mults):
            c_out = ch * m
            for _ in range(2):
                self.down.append(nn.ModuleList([ResBlock(c, c_out, emb_dim),
                                                SelfAttention(c_out) if res in attn_res else nn.Identity()]))
                c = c_out
                chans.append(c)
            if i < len(mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                chans.append(c)
                res //= 2

        self.mid1 = ResBlock(c, c, emb_dim)
        self.mid_attn = SelfAttention(c)
        self.mid2 = ResBlock(c, c, emb_dim)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            c_out = ch * m
            for _ in range(3):
                self.up.append(nn.ModuleList([ResBlock(c + chans.pop(), c_out, emb_dim),
                                              SelfAttention(c_out) if res in attn_res else nn.Identity()]))
                c = c_out
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(c, c, 3, padding=1)))
                res *= 2

        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, y=None):
        emb = self.time_mlp(timestep_embedding(t, self.ch))
        if self.label_emb is not None:
            if y is None:
                y = torch.full_like(t, self.num_classes)
            emb = emb + self.label_emb(y)

        h = self.inc(x)
        hs = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](h, emb))
            else:
                h = layer(h)
            hs.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = torch.cat([h, hs.pop()], dim=1)  # skip connection from the matching down level
                h = layer[1](layer[0](h, emb))
            else:
                h = layer(h)
        return self.out(h)
