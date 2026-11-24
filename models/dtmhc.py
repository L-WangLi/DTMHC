"""DTMHC: historical segment statistics, PatchTST, temporal fusion and RUL readout."""

from __future__ import annotations

import math

import torch
from torch import nn

from .layers import Block, PatchTSTEncoder


def sinusoid(pos: torch.Tensor, d: int) -> torch.Tensor:
    """Encode observation offsets [...], returning [..., d]."""
    i = torch.arange(0, d, 2, device=pos.device, dtype=torch.float32)
    ang = pos[..., None].float() / torch.pow(10000.0, i / d)
    return torch.stack([torch.sin(ang), torch.cos(ang)], -1).flatten(-2)


class DTMHC(nn.Module):
    """Final supervised model. All inputs belong to one observed record per sample.

    x: [B,W,D], pad: [B,W]; hx: [B,H,2D+1], hpad/hoff: [B,H].
    True mask entries denote padding. Forward returns an unclipped normalized
    estimate under ``y``; reported RUL is 125 * clamp(y, 0, 1).
    """

    def __init__(self, in_dim, d=64, heads=4, ff=128, layers=2,
                 patch=8, stride=4, d_ch=16, dropout=0.05):
        super().__init__()
        self.d = d
        self.backbone = PatchTSTEncoder(in_dim, d=d, heads=heads, layers=layers,
                                       ff=ff, patch=patch, stride=stride,
                                       d_ch=d_ch, dropout=dropout)
        self.bridge = nn.Identity()
        self.fusion = Block(d, heads, ff, dropout)
        self.heads = heads
        # The third row is inactive in the final model. Keeping it preserves
        # parameter names, initialization and compatibility with saved runs.
        self.type_emb = nn.Embedding(3, d)
        self.fuse_ln = nn.LayerNorm(d)
        self.hist_enc = nn.Sequential(nn.Linear(2 * in_dim + 1, d), nn.LayerNorm(d))
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.hidden = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(d, 1)

    def token_ends(self, window):
        return self.backbone.token_ends(window)

    def _fuse(self, h, tok_pad, inp):
        b, nw, _ = h.shape
        window = inp["pad"].shape[1]
        woff = torch.as_tensor(window - 1 - self.token_ends(window),
                               device=h.device, dtype=torch.float32)
        offs = woff[None].expand(b, -1)
        types = torch.ones(b, nw, dtype=torch.long, device=h.device)
        is_hist = torch.zeros(b, nw, dtype=torch.bool, device=h.device)
        hpad, hoff = inp["hpad"], inp["hoff"]
        hh = self.hist_enc(inp["hx"])
        nh = hh.shape[1]
        h = torch.cat([hh, h], 1)
        tok_pad = torch.cat([hpad, tok_pad], 1)
        offs = torch.cat([hoff, offs], 1)
        types = torch.cat([torch.zeros(b, nh, dtype=torch.long, device=h.device), types], 1)
        is_hist = torch.cat([torch.ones(b, nh, dtype=torch.bool, device=h.device), is_hist], 1)
        h = h + sinusoid(offs, self.d) + self.type_emb(types)
        n = h.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=h.device).triu(1)
        # Preserve the self-connection of padded query rows to avoid empty
        # softmax rows. Padded keys remain invisible to every valid query.
        blocked = (causal[None] | tok_pad[:, None, :]) & ~torch.eye(
            n, dtype=torch.bool, device=h.device)[None]
        h = self.fusion(h, blocked.repeat_interleave(self.heads, 0))
        return self.fuse_ln(h), tok_pad, is_hist

    def forward(self, inp):
        h, tok_pad = self.backbone(inp)
        h = self.bridge(h)
        h, tok_pad, is_hist = self._fuse(h, tok_pad, inp)
        real = (~tok_pad).float()
        logit_a = (h @ self.q) / math.sqrt(self.d)
        r = (h * real[..., None]).sum(1) / real.sum(1, keepdim=True).clamp(min=1.0)
        a = logit_a.masked_fill(tok_pad, float("-inf")).softmax(-1)
        ev = torch.einsum("bn,bnd->bd", a, h)
        off = torch.einsum("bn,bnd->bd", a, h - r[:, None])
        z = self.hidden(torch.cat([ev, off, r], -1))
        return {"y": self.out(z).squeeze(-1), "attn": a,
                "tok_pad": tok_pad, "is_hist": is_hist}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
