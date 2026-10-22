"""Integer inference spec for the world model, in numpy.

This is the contract the Rust engine implements bit-exactly. Every float op is float32 in a
fixed order (sequential sums, no fused multiply-add) and every matmul is int8 x int8 -> int32:

    quant(x)    = clamp(trunc(x * inv_scale +- 0.5), -127, 127)            round half away
    linear(xq)  = float32(xq . W_q) * dequant[j] + bias[j]
    rmsnorm(x)  = (x * (1 / sqrt(sum_seq(x*x) / d + eps))) * g
    softmax     = exp table on round((rowmax - s) * rs), then p_u8 = (e*255 + sum/2) // sum
"""

from dataclasses import dataclass

import numpy as np

from tinyworldformer.rustexport.qat import LUT_N, LUT_RES

f32 = np.float32
EPS = f32(1e-5)


@dataclass
class QLin:
    w: np.ndarray  # (o, k) int8
    deq: np.ndarray  # (loops, o) float32 = act_scale[loop] * w_scale[o]
    bias: np.ndarray  # (o,) float32
    inv: np.ndarray  # (loops,) float32, 1 / act_scale


@dataclass
class IntModel:
    d: int
    heads: int
    loops: int
    tok: np.ndarray  # (SUB, V, d) f32
    pos: np.ndarray  # (T, d) f32
    act: np.ndarray  # (A, d) f32
    g1: np.ndarray
    g2: np.ndarray
    gf: np.ndarray
    qkv: QLin
    o: QLin
    up: QLin
    down: QLin
    cell: QLin
    reward: QLin
    inv_q: np.ndarray  # (loops,)
    inv_k: np.ndarray
    inv_v: np.ndarray
    rs: np.ndarray  # (loops,) score -> table index: s_q * s_k / sqrt(dh) * LUT_RES
    os: np.ndarray  # (loops,) attention output dequant: s_v / 255
    lut: np.ndarray  # (LUT_N,) uint16


def quant(x: np.ndarray, inv: np.float32) -> np.ndarray:
    v = x * inv
    r = np.trunc(v + np.where(v >= 0, f32(0.5), f32(-0.5)).astype(f32))
    return np.clip(r, -127, 127).astype(np.int8)


def linear(xq: np.ndarray, q: QLin, i: int) -> np.ndarray:
    acc = xq.astype(np.int32) @ q.w.astype(np.int32).T
    return acc.astype(f32) * q.deq[i] + q.bias


def rmsnorm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    ss = np.zeros(x.shape[:-1], f32)
    for c in range(x.shape[-1]):
        ss = ss + x[..., c] * x[..., c]
    r = f32(1) / np.sqrt(ss / f32(x.shape[-1]) + EPS)
    return (x * r[..., None]) * g


def attention(m: IntModel, n1: np.ndarray, i: int) -> np.ndarray:
    D, dh = m.d, m.d // m.heads
    qkv = linear(quant(n1, m.qkv.inv[i]), m.qkv, i)
    q8 = quant(qkv[..., :D], m.inv_q[i]).astype(np.int32)
    k8 = quant(qkv[..., D : 2 * D], m.inv_k[i]).astype(np.int32)
    v8 = quant(qkv[..., 2 * D :], m.inv_v[i]).astype(np.int32)
    out = np.zeros(n1.shape, f32)
    for h in range(m.heads):
        sl = slice(h * dh, (h + 1) * dh)
        s = q8[..., sl] @ np.swapaxes(k8[..., sl], -1, -2)  # (B, T, T) int32
        diff = (s.max(-1, keepdims=True) - s).astype(f32) * m.rs[i]
        idx = np.minimum(np.trunc(diff + f32(0.5)), LUT_N - 1).astype(np.int64)
        e = m.lut[idx].astype(np.uint32)
        tot = e.sum(-1, keepdims=True)
        p = (e * 255 + tot // 2) // tot
        out[..., sl] = (p.astype(np.int32) @ v8[..., sl]).astype(f32) * m.os[i]
    return linear(quant(out, m.o.inv[i]), m.o, i)


def block(m: IntModel, h: np.ndarray, i: int) -> np.ndarray:
    x = h + attention(m, rmsnorm(h, m.g1), i)
    u = np.maximum(linear(quant(rmsnorm(x, m.g2), m.up.inv[i]), m.up, i), f32(0))
    return x + linear(quant(u, m.down.inv[i]), m.down, i)


def embed(m: IntModel, patches: np.ndarray, action: np.ndarray) -> np.ndarray:
    """patches (B, T, SUB) tokens, action (B,)."""
    x = m.tok[0][patches[..., 0]]
    for s in range(1, patches.shape[-1]):
        x = x + m.tok[s][patches[..., s]]
    return (x + m.pos) + m.act[action][:, None]


def forward(m: IntModel, patches: np.ndarray, action: np.ndarray, loops: int):
    """Returns cell logits (B, T, SUB*V) and reward logits (B, R) after `loops` loops."""
    inp = embed(m, patches, action)
    x = np.zeros_like(inp)
    for i in range(loops):
        x = block(m, x + inp, i)
    hn = rmsnorm(x, m.gf)
    cells = linear(quant(hn, m.cell.inv[loops - 1]), m.cell, loops - 1)
    pooled = np.zeros((hn.shape[0], hn.shape[2]), f32)
    for t in range(hn.shape[1]):
        pooled = pooled + hn[:, t]
    pooled = pooled / f32(hn.shape[1])
    reward = linear(quant(pooled, m.reward.inv[loops - 1]), m.reward, loops - 1)
    return cells, reward
