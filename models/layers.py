"""PatchTST window encoder and attention block used by DTMHC."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

def reverse_sinusoid(N: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(N - 1, -1, -1, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(N, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe

class Base(nn.Module):
    causal = 'token'
    d_out = 64

    def token_ends(self, W):
        return np.arange(W)

    def precompute(self, X: np.ndarray, pad: np.ndarray):
        return {'x': torch.as_tensor(X), 'pad': torch.as_tensor(pad)}

    def tok_pad(self, pad):
        ends = torch.as_tensor(self.token_ends(pad.shape[1]), device=pad.device)
        return pad[:, ends]

class Block(nn.Module):

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, key_padding_mask=None):
        h = self.ln1(x)
        (a, _) = self.attn(h, h, h, attn_mask=mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x)))

class PatchTSTEncoder(Base):
    """PatchTST-style: channel-independent patches (len 8, stride 4, right-aligned so the
    last patch ends at the window end), patch time = reverse-relative sinusoid on patch
    index, bidirectional encoder per channel (window-causal), channels fused per patch by
    concatenation -> Linear. No RevIN (sensor level carries degradation information).
    Patches entirely inside the left padding are masked; partially padded patches are
    padded with the first real value (protocol padding) and kept.
    The inactive mask_token and recon tensors are retained for checkpoint and
    initialization compatibility; supervised forward does not execute them."""
    causal = 'window'

    def __init__(self, in_dim, d=64, heads=4, layers=2, ff=128, dropout=0.05, patch=8, stride=4, d_ch=16, **_):
        super().__init__()
        (self.patch, self.stride, self.d, self.heads, self.C) = (patch, stride, d, heads, in_dim)
        self.embed = nn.Linear(patch, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList((Block(d, heads, ff, dropout) for _ in range(layers)))
        self.ln = nn.LayerNorm(d)
        self.ch_down = nn.Linear(d, d_ch)
        self.fuse = nn.Linear(in_dim * d_ch, d)
        self.recon = nn.Linear(d, patch)

    def token_ends(self, W):
        return np.arange(W - 1, self.patch - 2, -self.stride)[::-1].copy()

    def _patches(self, x, pad):
        W = x.shape[1]
        first = torch.gather(x, 1, pad.sum(1).clamp(max=W - 1)[:, None, None].expand(-1, 1, x.shape[-1]))
        x = torch.where(pad[..., None], first, x)
        ends = torch.as_tensor(self.token_ends(W), device=x.device)
        idx = ends[:, None] - torch.arange(self.patch - 1, -1, -1, device=x.device)[None]
        xp = x[:, idx]
        return (xp.permute(0, 3, 1, 2), pad[:, ends])

    def encode_channels(self, xp, tok_pad, mask=None):
        (B, C, N, P) = xp.shape
        e = self.embed(xp.reshape(B * C, N, P))
        if mask is not None:
            e = torch.where(mask.reshape(B * C, N, 1), self.mask_token.expand_as(e), e)
        e = e + reverse_sinusoid(N, self.d, xp.device)[None]
        kpm = tok_pad.repeat_interleave(C, 0)
        for blk in self.blocks:
            e = blk(e, key_padding_mask=kpm)
        return self.ln(e).reshape(B, C, N, self.d)

    def forward(self, inp):
        (xp, tp) = self._patches(inp['x'], inp['pad'])
        e = self.encode_channels(xp, tp)
        (B, C, N, _) = e.shape
        h = self.fuse(self.ch_down(e).permute(0, 2, 1, 3).reshape(B, N, -1))
        return (h, tp)
