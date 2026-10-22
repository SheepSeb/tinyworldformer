//! Forward pass, op for op the same as `tinyworldformer/intref.py` (bit-exact).

use crate::weights::*;
use crate::{dot_i8, dot_u8_i8};

const DH: usize = D / HEADS;
const QKV: usize = 3 * D;
const GW: usize = W / P;
const EPS: f32 = 1e-5;

struct Lin {
    w: &'static [i8],
    deq: &'static [f32], // (LOOPS, n)
    bias: &'static [f32],
    inv: &'static [f32], // (LOOPS,)
    n: usize,
    k: usize,
}

static QKV_L: Lin = Lin { w: &QKV_W, deq: &QKV_DEQ, bias: &QKV_BIAS, inv: &QKV_INV, n: QKV, k: D };
static O_L: Lin = Lin { w: &O_W, deq: &O_DEQ, bias: &O_BIAS, inv: &O_INV, n: D, k: D };
static UP_L: Lin = Lin { w: &UP_W, deq: &UP_DEQ, bias: &UP_BIAS, inv: &UP_INV, n: FF, k: D };
static DOWN_L: Lin = Lin { w: &DOWN_W, deq: &DOWN_DEQ, bias: &DOWN_BIAS, inv: &DOWN_INV, n: D, k: FF };
static CELL_L: Lin = Lin { w: &CELL_W, deq: &CELL_DEQ, bias: &CELL_BIAS, inv: &CELL_INV, n: SUB * V, k: D };
static REWARD_L: Lin = Lin { w: &REWARD_W, deq: &REWARD_DEQ, bias: &REWARD_BIAS, inv: &REWARD_INV, n: REWARD, k: D };

/// Round half away from zero, saturate to [-127, 127].
#[inline(always)]
fn quant(x: &[f32], inv: f32, out: &mut [i8]) {
    for (o, &x) in out.iter_mut().zip(x) {
        let v = x * inv;
        let r = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
        *o = (r as i32).clamp(-127, 127) as i8;
    }
}

/// out (rows x n) = dequant(xq (rows x k) . W^T) + bias, for loop `i`.
fn linear(xq: &[i8], l: &Lin, i: usize, out: &mut [f32]) {
    let deq = &l.deq[i * l.n..(i + 1) * l.n];
    for (row, out_row) in xq.chunks_exact(l.k).zip(out.chunks_exact_mut(l.n)) {
        let cols = l.w.chunks_exact(l.k).zip(deq.iter().zip(l.bias));
        for (o, (w, (&dq, &b))) in out_row.iter_mut().zip(cols) {
            *o = dot_i8(row, w, 0) as f32 * dq + b;
        }
    }
}

fn rmsnorm(x: &[f32], g: &[f32], out: &mut [f32]) {
    for (row, out_row) in x.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
        let mut ss = 0.0f32;
        for &v in row {
            ss += v * v;
        }
        let r = 1.0 / libm::sqrtf(ss / D as f32 + EPS);
        for ((o, &v), &g) in out_row.iter_mut().zip(row).zip(g) {
            *o = (v * r) * g;
        }
    }
}

/// Scratch buffers for one forward pass (~130 KB). Put it in a `static` on the device.
pub struct Engine {
    inp: [f32; T * D],
    x: [f32; T * D],
    h: [f32; T * D],
    n: [f32; T * D],
    att: [f32; T * D],
    qkv: [f32; T * QKV],
    u: [f32; T * FF],
    xq: [i8; T * FF],
    q8: [i8; T * D],
    k8: [i8; T * D],
    vt: [i8; D * T], // V transposed: vt[c * T + u]
    s: [i32; T],
    p: [u8; T],
    pooled: [f32; D],
    pooled_q: [i8; D],
    pub cells: [f32; T * SUB * V],
    pub reward: [f32; REWARD],
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub const fn new() -> Self {
        Self {
            inp: [0.0; T * D], x: [0.0; T * D], h: [0.0; T * D], n: [0.0; T * D],
            att: [0.0; T * D], qkv: [0.0; T * QKV], u: [0.0; T * FF], xq: [0; T * FF],
            q8: [0; T * D], k8: [0; T * D], vt: [0; D * T], s: [0; T], p: [0; T],
            pooled: [0.0; D], pooled_q: [0; D], cells: [0.0; T * SUB * V], reward: [0.0; REWARD],
        }
    }

    fn embed(&mut self, grid: &[u8; H * W], action: usize) {
        for t in 0..T {
            let (py, px) = (t / GW, t % GW);
            let x = &mut self.inp[t * D..(t + 1) * D];
            for s in 0..SUB {
                let tok = grid[(py * P + s / P) * W + px * P + s % P] as usize;
                let e = &TOK[(s * V + tok) * D..][..D];
                for (xv, &ev) in x.iter_mut().zip(e) {
                    *xv = if s == 0 { ev } else { *xv + ev };
                }
            }
            for ((xv, &p), &a) in x.iter_mut().zip(&POS[t * D..]).zip(&ACT[action * D..]) {
                *xv = (*xv + p) + a;
            }
        }
    }

    fn attention(&mut self, i: usize) {
        quant(&self.n, QKV_L.inv[i], &mut self.xq[..T * D]);
        linear(&self.xq[..T * D], &QKV_L, i, &mut self.qkv);
        for t in 0..T {
            let row = &self.qkv[t * QKV..(t + 1) * QKV];
            quant(&row[..D], INV_Q[i], &mut self.q8[t * D..(t + 1) * D]);
            quant(&row[D..2 * D], INV_K[i], &mut self.k8[t * D..(t + 1) * D]);
            let mut v = [0i8; D];
            quant(&row[2 * D..], INV_V[i], &mut v);
            for (c, &vc) in v.iter().enumerate() {
                self.vt[c * T + t] = vc;
            }
        }
        for h in 0..HEADS {
            for t in 0..T {
                let q = &self.q8[t * D + h * DH..][..DH];
                let mut max = i32::MIN;
                for (u, s) in self.s.iter_mut().enumerate() {
                    *s = dot_i8(q, &self.k8[u * D + h * DH..][..DH], 0);
                    max = max.max(*s);
                }
                let mut e = [0u32; T];
                let mut tot = 0u32;
                for (ev, &s) in e.iter_mut().zip(&self.s) {
                    let idx = (((max - s) as f32 * RS[i] + 0.5) as u32).min(LUT_N as u32 - 1);
                    *ev = LUT[idx as usize] as u32;
                    tot += *ev;
                }
                for (p, &ev) in self.p.iter_mut().zip(&e) {
                    *p = ((ev * 255 + tot / 2) / tot) as u8;
                }
                for c in 0..DH {
                    let col = h * DH + c;
                    let acc = dot_u8_i8(&self.p, &self.vt[col * T..(col + 1) * T], 0);
                    self.att[t * D + col] = acc as f32 * OS[i];
                }
            }
        }
        quant(&self.att, O_L.inv[i], &mut self.xq[..T * D]);
        linear(&self.xq[..T * D], &O_L, i, &mut self.n); // n is free again: reuse for the projection
        for ((x, &h), &a) in self.x.iter_mut().zip(&self.h).zip(&self.n) {
            *x = h + a;
        }
    }

    fn block(&mut self, i: usize) {
        rmsnorm(&self.h, &G1, &mut self.n);
        self.attention(i); // x = h + attn
        rmsnorm(&self.x, &G2, &mut self.n);
        quant(&self.n, UP_L.inv[i], &mut self.xq[..T * D]);
        linear(&self.xq[..T * D], &UP_L, i, &mut self.u);
        for v in self.u.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        quant(&self.u, DOWN_L.inv[i], &mut self.xq);
        linear(&self.xq, &DOWN_L, i, &mut self.n);
        for (x, &d) in self.x.iter_mut().zip(&self.n) {
            *x += d;
        }
    }

    /// Run `loops` (1..=LOOPS) iterations; fills `self.cells` (T x SUB*V) and `self.reward` logits.
    pub fn forward(&mut self, grid: &[u8; H * W], action: usize, loops: usize) {
        assert!((1..=LOOPS).contains(&loops) && action < ACTIONS);
        self.embed(grid, action);
        self.x = [0.0; T * D];
        for i in 0..loops {
            for ((h, &x), &e) in self.h.iter_mut().zip(&self.x).zip(&self.inp) {
                *h = x + e;
            }
            self.block(i);
        }
        let last = loops - 1;
        rmsnorm(&self.x, &GF, &mut self.n);
        quant(&self.n, CELL_L.inv[last], &mut self.xq[..T * D]);
        linear(&self.xq[..T * D], &CELL_L, last, &mut self.cells);
        self.pooled = [0.0; D];
        for row in self.n.chunks_exact(D) {
            for (p, &v) in self.pooled.iter_mut().zip(row) {
                *p += v;
            }
        }
        for p in self.pooled.iter_mut() {
            *p /= T as f32;
        }
        quant(&self.pooled, REWARD_L.inv[last], &mut self.pooled_q);
        linear(&self.pooled_q, &REWARD_L, last, &mut self.reward);
    }

    /// Logits of cell (row, col) after `forward`.
    pub fn cell_logits(&self, row: usize, col: usize) -> &[f32] {
        let t = (row / P) * GW + col / P;
        let s = (row % P) * P + col % P;
        &self.cells[(t * SUB + s) * V..][..V]
    }
}

/// Index of the first maximum (numpy `argmax` semantics).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}
