//! Stream AC(lambda) (Elsayed, Vasan & Mahmood, 2024), learning one transition at a time on the
//! device: no replay buffer, no batches, no allocator. Mirrors `streamx.StreamAC`:
//!
//! - policy and value nets: Linear -> LayerNorm (no affine) -> LeakyReLU, twice, then Linear
//! - ObGD: eligibility traces z = gamma*lambda*z + grad, step = lr / max(1, lr*kappa*max(|delta|,1)*|z|_1),
//!   w += step * delta * z
//! - sparse init (90% of each row zeroed), online observation normalization, reward scaling
//!
//! The backward pass is written out by hand. `tests/learner.rs` checks it against PyTorch.

use crate::agent_weights::OBS;
use crate::dream::XorShift;
use crate::weights::ACTIONS;

/// Hidden width for learning on the device (the exported, pretrained agent may be wider).
pub const HIDDEN: usize = 32;
pub const GAMMA: f32 = 0.99;
pub const LAMBDA: f32 = 0.8;
pub const ENTROPY: f32 = 0.01;
const LR: f32 = 1.0;
const LN_EPS: f32 = 1e-5;
const SLOPE: f32 = 0.01;

/// Two hidden layers of `HIDDEN`, `O` outputs; every weight has an eligibility trace.
pub struct Net<const O: usize> {
    pub w1: [[f32; OBS]; HIDDEN],
    pub b1: [f32; HIDDEN],
    pub w2: [[f32; HIDDEN]; HIDDEN],
    pub b2: [f32; HIDDEN],
    pub w3: [[f32; HIDDEN]; O],
    pub b3: [f32; O],
    z1: [[f32; OBS]; HIDDEN],
    zb1: [f32; HIDDEN],
    z2: [[f32; HIDDEN]; HIDDEN],
    zb2: [f32; HIDDEN],
    z3: [[f32; HIDDEN]; O],
    zb3: [f32; O],
    kappa: f32,
}

/// Activations kept from the forward pass for the backward pass.
struct Cache {
    n1: [f32; HIDDEN],
    a1: [f32; HIDDEN],
    rstd1: f32,
    n2: [f32; HIDDEN],
    a2: [f32; HIDDEN],
    rstd2: f32,
}

fn layer_norm(h: &[f32; HIDDEN]) -> ([f32; HIDDEN], f32) {
    let mean = h.iter().sum::<f32>() / HIDDEN as f32;
    let var = h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / HIDDEN as f32;
    let rstd = 1.0 / libm::sqrtf(var + LN_EPS);
    (core::array::from_fn(|i| (h[i] - mean) * rstd), rstd)
}

fn leaky(n: &[f32; HIDDEN]) -> [f32; HIDDEN] {
    core::array::from_fn(|i| if n[i] > 0.0 { n[i] } else { SLOPE * n[i] })
}

/// d(loss)/d(h) of y = layer_norm(h) given d(loss)/d(y), with n = layer_norm(h).
fn layer_norm_back(dn: &[f32; HIDDEN], n: &[f32; HIDDEN], rstd: f32) -> [f32; HIDDEN] {
    let mean_dn = dn.iter().sum::<f32>() / HIDDEN as f32;
    let mean_dn_n = dn.iter().zip(n).map(|(d, n)| d * n).sum::<f32>() / HIDDEN as f32;
    core::array::from_fn(|i| rstd * (dn[i] - mean_dn - n[i] * mean_dn_n))
}

impl<const O: usize> Net<O> {
    /// All zeros; call `sparse_init` (or load weights) before use.
    pub const fn zeroed(kappa: f32) -> Self {
        Self {
            w1: [[0.0; OBS]; HIDDEN], b1: [0.0; HIDDEN], w2: [[0.0; HIDDEN]; HIDDEN], b2: [0.0; HIDDEN],
            w3: [[0.0; HIDDEN]; O], b3: [0.0; O],
            z1: [[0.0; OBS]; HIDDEN], zb1: [0.0; HIDDEN], z2: [[0.0; HIDDEN]; HIDDEN], zb2: [0.0; HIDDEN],
            z3: [[0.0; HIDDEN]; O], zb3: [0.0; O], kappa,
        }
    }

    fn clear_traces(&mut self) {
        self.z1 = [[0.0; OBS]; HIDDEN];
        self.zb1 = [0.0; HIDDEN];
        self.z2 = [[0.0; HIDDEN]; HIDDEN];
        self.zb2 = [0.0; HIDDEN];
        self.z3 = [[0.0; HIDDEN]; O];
        self.zb3 = [0.0; O];
    }

    /// Uniform(+-1/sqrt(fan_in)) weights with 90% of each row zeroed, zero biases.
    pub fn sparse_init(&mut self, rng: &mut XorShift) {
        fn row(r: &mut [f32], rng: &mut XorShift) {
            let bound = 1.0 / libm::sqrtf(r.len() as f32);
            for w in r.iter_mut() {
                *w = (rng.unit() * 2.0 - 1.0) * bound;
            }
            let zeros = (r.len() * 9).div_ceil(10);
            for k in 0..zeros {
                // zero `zeros` distinct entries: a partial Fisher-Yates over the indices
                let j = k + rng.next_u32() as usize % (r.len() - k);
                r.swap(k, j);
                r[k] = 0.0;
            }
        }
        self.w1.iter_mut().for_each(|r| row(r, rng));
        self.w2.iter_mut().for_each(|r| row(r, rng));
        self.w3.iter_mut().for_each(|r| row(r, rng));
        self.b1 = [0.0; HIDDEN];
        self.b2 = [0.0; HIDDEN];
        self.b3 = [0.0; O];
    }

    fn forward(&self, x: &[f32; OBS]) -> ([f32; O], Cache) {
        let h1: [f32; HIDDEN] = core::array::from_fn(|i| self.w1[i].iter().zip(x).fold(self.b1[i], |a, (w, x)| a + w * x));
        let (n1, rstd1) = layer_norm(&h1);
        let a1 = leaky(&n1);
        let h2: [f32; HIDDEN] = core::array::from_fn(|i| self.w2[i].iter().zip(&a1).fold(self.b2[i], |a, (w, x)| a + w * x));
        let (n2, rstd2) = layer_norm(&h2);
        let a2 = leaky(&n2);
        let y = core::array::from_fn(|o| self.w3[o].iter().zip(&a2).fold(self.b3[o], |a, (w, x)| a + w * x));
        (y, Cache { n1, a1, rstd1, n2, a2, rstd2 })
    }

    pub fn output(&self, x: &[f32; OBS]) -> [f32; O] {
        self.forward(x).0
    }

    /// Decay every trace by gamma*lambda, then add the gradient of `dy . y` (y the outputs).
    #[allow(clippy::needless_range_loop)] // indexed form reads like the maths
    fn accumulate(&mut self, x: &[f32; OBS], c: &Cache, dy: &[f32; O]) {
        let decay = GAMMA * LAMBDA;
        let mut da2 = [0.0f32; HIDDEN];
        for o in 0..O {
            self.zb3[o] = decay * self.zb3[o] + dy[o];
            for i in 0..HIDDEN {
                self.z3[o][i] = decay * self.z3[o][i] + dy[o] * c.a2[i];
                da2[i] += self.w3[o][i] * dy[o];
            }
        }
        let dn2: [f32; HIDDEN] = core::array::from_fn(|i| if c.n2[i] > 0.0 { da2[i] } else { SLOPE * da2[i] });
        let dh2 = layer_norm_back(&dn2, &c.n2, c.rstd2);
        let mut da1 = [0.0f32; HIDDEN];
        for i in 0..HIDDEN {
            self.zb2[i] = decay * self.zb2[i] + dh2[i];
            for j in 0..HIDDEN {
                self.z2[i][j] = decay * self.z2[i][j] + dh2[i] * c.a1[j];
                da1[j] += self.w2[i][j] * dh2[i];
            }
        }
        let dn1: [f32; HIDDEN] = core::array::from_fn(|i| if c.n1[i] > 0.0 { da1[i] } else { SLOPE * da1[i] });
        let dh1 = layer_norm_back(&dn1, &c.n1, c.rstd1);
        for i in 0..HIDDEN {
            self.zb1[i] = decay * self.zb1[i] + dh1[i];
            for (z, &xv) in self.z1[i].iter_mut().zip(x) {
                *z = decay * *z + dh1[i] * xv;
            }
        }
    }

    /// ObGD: one bounded step along the traces, scaled by the TD error.
    fn obgd(&mut self, delta: f32, reset: bool) {
        let mut z_sum = 0.0f32;
        let mut sum = |z: &[f32]| z_sum += z.iter().map(|v| libm::fabsf(*v)).sum::<f32>();
        self.z1.iter().for_each(|r| sum(r));
        sum(&self.zb1);
        self.z2.iter().for_each(|r| sum(r));
        sum(&self.zb2);
        self.z3.iter().for_each(|r| sum(r));
        sum(&self.zb3);
        let bound = LR * self.kappa * libm::fabsf(delta).max(1.0) * z_sum;
        let alpha = (if bound > 1.0 { LR / bound } else { LR }) * delta;
        fn apply(w: &mut [f32], z: &mut [f32], alpha: f32, reset: bool) {
            for (w, z) in w.iter_mut().zip(z.iter_mut()) {
                *w += alpha * *z;
                if reset {
                    *z = 0.0;
                }
            }
        }
        for (w, z) in self.w1.iter_mut().zip(self.z1.iter_mut()) {
            apply(w, z, alpha, reset);
        }
        apply(&mut self.b1, &mut self.zb1, alpha, reset);
        for (w, z) in self.w2.iter_mut().zip(self.z2.iter_mut()) {
            apply(w, z, alpha, reset);
        }
        apply(&mut self.b2, &mut self.zb2, alpha, reset);
        for (w, z) in self.w3.iter_mut().zip(self.z3.iter_mut()) {
            apply(w, z, alpha, reset);
        }
        apply(&mut self.b3, &mut self.zb3, alpha, reset);
    }
}

/// Welford running mean / variance (variance 1 until there are two samples).
struct Running<const N: usize> {
    n: f32,
    mean: [f32; N],
    m2: [f32; N],
}

impl<const N: usize> Running<N> {
    const fn new() -> Self {
        Self { n: 0.0, mean: [0.0; N], m2: [0.0; N] }
    }

    fn update(&mut self, x: &[f32; N]) {
        self.n += 1.0;
        for ((&x, mean), m2) in x.iter().zip(self.mean.iter_mut()).zip(self.m2.iter_mut()) {
            let d = x - *mean;
            *mean += d / self.n;
            *m2 += d * (x - *mean);
        }
    }

    fn var(&self, i: usize) -> f32 {
        if self.n > 1.0 { self.m2[i] / self.n } else { 1.0 }
    }
}

pub struct Learner {
    pub pi: Net<ACTIONS>,
    pub v: Net<1>,
    obs: Running<OBS>,
    ret_norm: Running<1>,
    ret: f32,
    pub rng: XorShift,
}

impl Learner {
    /// A fresh, untrained agent. `const` so it can live in a `static` (it is ~85 KB).
    pub const fn zeroed(seed: u32) -> Self {
        Self {
            pi: Net::zeroed(3.0), v: Net::zeroed(2.0), obs: Running::new(), ret_norm: Running::new(), ret: 0.0,
            rng: XorShift(if seed == 0 { 1 } else { seed }),
        }
    }

    pub fn init(&mut self) {
        self.pi.sparse_init(&mut self.rng);
        self.v.sparse_init(&mut self.rng);
    }

    /// Forget everything and start from scratch, in place (no ~85 KB temporary on the stack).
    pub fn reset(&mut self, seed: u32) {
        self.rng = XorShift(seed.max(1));
        self.pi.clear_traces();
        self.v.clear_traces();
        self.obs = Running::new();
        self.ret_norm = Running::new();
        self.ret = 0.0;
        self.init();
    }

    /// Normalize an observation with the running statistics (updating them first, if asked).
    pub fn normalize(&mut self, obs: &[f32; OBS], update: bool) -> [f32; OBS] {
        if update {
            self.obs.update(obs);
        }
        core::array::from_fn(|i| (obs[i] - self.obs.mean[i]) / libm::sqrtf(self.obs.var(i) + 1e-8))
    }

    /// Scale a reward by the running spread of the discounted return.
    pub fn scale_reward(&mut self, r: f32, done: bool) -> f32 {
        self.ret = GAMMA * self.ret * if done { 0.0 } else { 1.0 } + r;
        self.ret_norm.update(&[self.ret]);
        r / libm::sqrtf(self.ret_norm.var(0) + 1e-8)
    }

    /// Sample from the policy (or take its argmax when `greedy`).
    pub fn act(&mut self, x: &[f32; OBS], greedy: bool) -> usize {
        let logits = self.pi.output(x);
        if greedy {
            return crate::model::argmax(&logits);
        }
        let p = softmax(&logits);
        let mut u = self.rng.unit();
        for (a, &pa) in p.iter().enumerate() {
            if u < pa {
                return a;
            }
            u -= pa;
        }
        ACTIONS - 1
    }

    /// One streaming TD update from (x, a, r, x2). `reset` clears the traces (end of episode).
    pub fn update(&mut self, x: &[f32; OBS], a: usize, r: f32, x2: &[f32; OBS], terminal: bool, reset: bool) {
        let target = r + if terminal { 0.0 } else { GAMMA * self.v.output(x2)[0] };
        let (v, vc) = self.v.forward(x);
        let delta = target - v[0];
        // value: trace of grad V
        self.v.accumulate(x, &vc, &[1.0]);
        // policy: trace of grad (log pi(a|x) + entropy * sign(delta) * H(pi(.|x)))
        let (logits, pc) = self.pi.forward(x);
        let p = softmax(&logits);
        let logp: [f32; ACTIONS] = core::array::from_fn(|i| libm::logf(p[i].max(1e-30)));
        let h = -p.iter().zip(&logp).map(|(p, l)| p * l).sum::<f32>();
        let sign = if delta < 0.0 { -1.0 } else { 1.0 };
        let dy: [f32; ACTIONS] = core::array::from_fn(|j| {
            let dlogp = if j == a { 1.0 } else { 0.0 } - p[j];
            let dh = -p[j] * (logp[j] + h);
            dlogp + ENTROPY * sign * dh
        });
        self.pi.accumulate(x, &pc, &dy);
        self.v.obgd(delta, reset);
        self.pi.obgd(delta, reset);
    }
}

fn softmax<const N: usize>(l: &[f32; N]) -> [f32; N] {
    let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: [f32; N] = core::array::from_fn(|i| libm::expf(l[i] - m));
    let s: f32 = e.iter().sum();
    core::array::from_fn(|i| e[i] / s)
}
