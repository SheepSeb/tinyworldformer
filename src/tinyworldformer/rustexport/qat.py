"""Quantization-aware training pieces whose int8 behaviour `intref` reproduces exactly.

Build a model from these and it can be exported: symmetric int8 weights (per output channel),
int8 activations with one EMA abs-max scale per loop index, an exp lookup table for softmax and
uint8 attention probabilities. Before `ActQuant.enabled` is switched on everything runs in float
(the warm-up); after it, the forward pass rounds exactly where the integer spec does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LUT_RES = 64  # exp table entries per unit of (max - logit)
LUT_N = 1024


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.trunc(x + 0.5 * torch.sign(x))


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: the value of `xq`, the gradient of `x`."""
    return x + (xq - x).detach()


def fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = torch.clamp(x / scale, -127, 127)
    return ste(q, round_half_away(q)) * scale


class ActQuant(nn.Module):
    """Symmetric per-tensor int8 activation quantizer with one EMA abs-max scale per loop."""

    enabled = False  # flipped on for every model at once after the float warm-up

    def __init__(self, loops: int, momentum: float = 0.99):
        super().__init__()
        self.register_buffer("absmax", torch.zeros(loops))
        self.m = momentum

    def scale(self, i: int) -> torch.Tensor:
        return self.absmax[i].clamp_min(1e-6) / 127

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        if self.training:
            cur = x.detach().abs().amax()
            old = self.absmax[i]
            self.absmax[i] = torch.where(old == 0, cur, self.m * old + (1 - self.m) * cur)
        return fake_quant(x, self.scale(i)) if ActQuant.enabled else x


def weight_scale(w: torch.Tensor) -> torch.Tensor:
    return w.detach().abs().amax(1, keepdim=True).clamp_min(1e-8) / 127


class QLinear(nn.Module):
    """Linear layer with an int8 input quantizer (per loop) and int8 per-channel weights."""

    def __init__(self, i: int, o: int, loops: int):
        super().__init__()
        self.lin = nn.Linear(i, o)
        self.aq = ActQuant(loops)

    def forward(self, x: torch.Tensor, loop: int) -> torch.Tensor:
        x = self.aq(x, loop)
        w = self.lin.weight
        if ActQuant.enabled:
            w = fake_quant(w, weight_scale(w))
        return F.linear(x, w, self.lin.bias)


def softmax_u8(logits: torch.Tensor) -> torch.Tensor:
    """Softmax as `intref.attention` computes it: exp table on the distance to the row max, then
    uint8 probabilities (straight-through, so it still trains)."""
    if not ActQuant.enabled:
        return logits.softmax(-1)
    r = (logits.amax(-1, keepdim=True) - logits) * LUT_RES
    r = ste(r, torch.clamp(torch.trunc(r + 0.5), max=LUT_N - 1))
    e = torch.exp(-r / LUT_RES)
    e = ste(e, torch.round(e * 65535) / 65535)
    p = e / e.sum(-1, keepdim=True)
    return ste(p, torch.floor(p * 255 + 0.5) / 255)


class LoopedBlock(nn.Module):
    """One pre-norm transformer block (RMSNorm, multi-head attention, ReLU MLP) meant to be applied
    `loops` times with shared weights; every quantizer keeps one scale per loop index."""

    def __init__(self, d: int, heads: int, ff: int, loops: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.RMSNorm(d, eps=1e-5), nn.RMSNorm(d, eps=1e-5)
        self.qkv = QLinear(d, 3 * d, loops)
        self.o = QLinear(d, d, loops)
        self.up = QLinear(d, ff, loops)
        self.down = QLinear(ff, d, loops)
        self.qa, self.ka, self.va = (ActQuant(loops) for _ in range(3))

    def attend(self, x: torch.Tensor, i: int) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x), i).chunk(3, -1)
        q, k, v = (
            a(t, i).view(B, N, self.h, self.dh).transpose(1, 2)
            for a, t in ((self.qa, q), (self.ka, k), (self.va, v))
        )
        p = softmax_u8((q @ k.transpose(-1, -2)) / self.dh**0.5)
        y = (p @ v).transpose(1, 2).reshape(B, N, D)
        return self.o(y, i)

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        x = x + self.attend(x, i)
        return x + self.down(F.relu(self.up(self.n2(x), i)), i)
