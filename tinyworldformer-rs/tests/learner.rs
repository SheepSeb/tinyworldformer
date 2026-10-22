//! The Rust learner must follow PyTorch's StreamAC: same start, same transitions, same weights after.

use tinyworldformer_rs::agent_weights::OBS;
use tinyworldformer_rs::learner::{Learner, Net, HIDDEN};

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn f32s<const N: usize>(&mut self) -> [f32; N] {
        let out = core::array::from_fn(|i| f32::from_le_bytes(self.0[4 * i..4 * i + 4].try_into().unwrap()));
        self.0 = &self.0[4 * N..];
        out
    }
    fn u8(&mut self) -> u8 {
        let v = self.0[0];
        self.0 = &self.0[1..];
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.0[..4].try_into().unwrap());
        self.0 = &self.0[4..];
        v
    }
    fn net<const O: usize>(&mut self, net: &mut Net<O>) {
        for row in net.w1.iter_mut() {
            *row = self.f32s();
        }
        net.b1 = self.f32s();
        for row in net.w2.iter_mut() {
            *row = self.f32s();
        }
        net.b2 = self.f32s();
        for row in net.w3.iter_mut() {
            *row = self.f32s();
        }
        net.b3 = self.f32s();
    }
}

fn max_rel_diff<const O: usize>(a: &Net<O>, b: &Net<O>) -> f32 {
    let mut worst = 0.0f32;
    let mut cmp = |x: &[f32], y: &[f32]| {
        for (x, y) in x.iter().zip(y) {
            worst = worst.max((x - y).abs() / (1e-3 + y.abs()));
        }
    };
    a.w1.iter().zip(&b.w1).for_each(|(x, y)| cmp(x, y));
    cmp(&a.b1, &b.b1);
    a.w2.iter().zip(&b.w2).for_each(|(x, y)| cmp(x, y));
    cmp(&a.b2, &b.b2);
    a.w3.iter().zip(&b.w3).for_each(|(x, y)| cmp(x, y));
    cmp(&a.b3, &b.b3);
    worst
}

#[test]
fn matches_pytorch_stream_ac() {
    let data = std::fs::read(concat!(env!("TWF_AGENT_DIR"), "/golden_learner.bin")).unwrap();
    let mut r = Reader(&data);
    let steps = r.u32();
    let mut learner = Box::new(Learner::zeroed(1));
    r.net(&mut learner.pi);
    r.net(&mut learner.v);
    let mut x = learner.normalize(&r.f32s::<OBS>(), true);
    for _ in 0..steps {
        let a = r.u8() as usize;
        let reward = r.f32s::<1>()[0];
        let done = r.u8() != 0;
        let x2 = learner.normalize(&r.f32s::<OBS>(), true);
        let scaled = learner.scale_reward(reward, done);
        learner.update(&x, a, scaled, &x2, done, done);
        x = x2;
        if done {
            x = learner.normalize(&r.f32s::<OBS>(), true);
        }
    }
    let mut want = Box::new(Learner::zeroed(1));
    r.net(&mut want.pi);
    r.net(&mut want.v);
    assert!(r.0.is_empty());
    let (dp, dv) = (max_rel_diff(&learner.pi, &want.pi), max_rel_diff(&learner.v, &want.v));
    println!("after {steps} updates: max relative weight difference, policy {dp:.2e}, value {dv:.2e} (hidden {HIDDEN})");
    assert!(dp < 1e-4 && dv < 1e-4, "policy {dp}, value {dv}");
}
