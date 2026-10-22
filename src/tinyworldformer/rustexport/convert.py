"""Turn trained QAT modules into the integer spec (`intref`) and into Rust arrays."""

import numpy as np
import torch
import torch.nn as nn

from tinyworldformer.rustexport import intref
from tinyworldformer.rustexport.codegen import rust_array
from tinyworldformer.rustexport.qat import LUT_N, LUT_RES, ActQuant, LoopedBlock, QLinear, round_half_away, weight_scale

f32 = np.float32


def act_scale(a: ActQuant) -> np.ndarray:
    """Per-loop activation scales, computed in float64 like the QAT forward pass rounds them."""
    return (a.absmax.double().clamp_min(1e-6) / 127).numpy()


def qlin(q: QLinear) -> intref.QLin:
    w = q.lin.weight.detach()
    sw = weight_scale(w)  # float32, exactly as during QAT
    wq = round_half_away(torch.clamp(w / sw, -127, 127))
    sa = q.aq.absmax.double().clamp_min(1e-6) / 127
    return intref.QLin(
        w=wq.numpy().astype(np.int8),
        deq=(sa[:, None] * sw[:, 0].double()[None]).numpy().astype(f32),
        bias=q.lin.bias.detach().numpy().astype(f32),
        inv=(1 / sa).numpy().astype(f32),
    )


def exp_lut() -> np.ndarray:
    """The softmax exp table: round(65535 * exp(-i / LUT_RES)) for i < LUT_N."""
    return np.round(65535 * np.exp(-np.arange(LUT_N) / LUT_RES)).astype(np.uint16)


def looped_model(block: LoopedBlock, *, tok: torch.Tensor, pos: torch.Tensor, act: torch.Tensor,
                 norm: nn.RMSNorm, cell: QLinear, reward: QLinear, loops: int) -> intref.IntModel:
    """A looped transformer with summed embedding tables (`tok` (SUB, V, d) per sub-cell, `pos`
    (T, d), `act` (A, d)), a final RMSNorm and two heads: per-token `cell` logits and `reward`
    logits from the mean token."""
    sq, sk, sv = act_scale(block.qa), act_scale(block.ka), act_scale(block.va)
    npf = lambda t: t.detach().numpy().astype(f32)  # noqa: E731
    return intref.IntModel(
        d=block.h * block.dh, heads=block.h, loops=loops,
        tok=npf(tok), pos=npf(pos), act=npf(act),
        g1=npf(block.n1.weight), g2=npf(block.n2.weight), gf=npf(norm.weight),
        qkv=qlin(block.qkv), o=qlin(block.o), up=qlin(block.up), down=qlin(block.down),
        cell=qlin(cell), reward=qlin(reward),
        inv_q=(1 / sq).astype(f32), inv_k=(1 / sk).astype(f32), inv_v=(1 / sv).astype(f32),
        rs=(sq * sk / np.sqrt(block.dh) * LUT_RES).astype(f32), os=(sv / 255).astype(f32),
        lut=exp_lut(),
    )


def int_model_arrays(m: intref.IntModel) -> list[str]:
    """Every tensor of an IntModel as Rust statics (names as the Rust engine expects them)."""
    out = [
        rust_array("TOK", m.tok, "f32"), rust_array("POS", m.pos, "f32"), rust_array("ACT", m.act, "f32"),
        rust_array("G1", m.g1, "f32"), rust_array("G2", m.g2, "f32"), rust_array("GF", m.gf, "f32"),
        rust_array("INV_Q", m.inv_q, "f32"), rust_array("INV_K", m.inv_k, "f32"),
        rust_array("INV_V", m.inv_v, "f32"), rust_array("RS", m.rs, "f32"),
        rust_array("OS", m.os, "f32"), rust_array("LUT", m.lut, "u16"),
    ]
    for name in ("qkv", "o", "up", "down", "cell", "reward"):
        q: intref.QLin = getattr(m, name)
        n = name.upper()
        out += [rust_array(f"{n}_W", q.w, "i8"), rust_array(f"{n}_DEQ", q.deq, "f32"),
                rust_array(f"{n}_BIAS", q.bias, "f32"), rust_array(f"{n}_INV", q.inv, "f32")]
    return out


def mlp_arrays(layers: list[tuple[str, nn.Linear]]) -> list[str]:
    """Float32 dense layers as `{NAME}_W` (out, in) and `{NAME}_B` statics."""
    out = []
    for name, layer in layers:
        out += [rust_array(f"{name}_W", layer.weight.detach().numpy(), "f32"),
                rust_array(f"{name}_B", layer.bias.detach().numpy(), "f32")]
    return out
