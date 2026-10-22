//! The Rust engine must reproduce the Python integer spec bit-for-bit.

use tinyworldformer_rs::model::Engine;
use tinyworldformer_rs::weights::{ACTIONS, H, REWARD, SUB, T, V, W};

#[test]
fn matches_python_integer_spec() {
    let data = std::fs::read(concat!(env!("TWF_ENV_DIR"), "/golden.bin")).unwrap();
    let n = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let mut engine = Box::new(Engine::new());
    let mut off = 4;
    for sample in 0..n {
        let grid: [u8; H * W] = data[off..off + H * W].try_into().unwrap();
        off += H * W;
        let (action, loops) = (data[off] as usize, data[off + 1] as usize);
        off += 2;
        let cells = f32s(&data[off..off + T * SUB * V * 4]);
        off += T * SUB * V * 4;
        let reward = f32s(&data[off..off + REWARD * 4]);
        off += REWARD * 4;

        engine.forward(&grid, action, loops);
        for (i, (&got, &want)) in engine.cells.iter().zip(&cells).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "sample {sample} (loops {loops}) cell logit {i}: {got} vs {want}");
        }
        for (&got, &want) in engine.reward.iter().zip(&reward) {
            assert_eq!(got.to_bits(), want.to_bits(), "sample {sample} reward logit: {got} vs {want}");
        }
    }
    assert_eq!(off, data.len());
}

#[test]
fn agent_matches_python() {
    use tinyworldformer_rs::agent::{logits, observe};
    use tinyworldformer_rs::agent_weights::OBS;
    use tinyworldformer_rs::model::argmax;

    let data = std::fs::read(concat!(env!("TWF_AGENT_DIR"), "/golden_agent.bin")).unwrap();
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    const CELLS: usize = H * W;
    let rec = CELLS + (OBS + ACTIONS) * 4;
    assert_eq!(data.len() % rec, 0);
    for chunk in data.chunks_exact(rec) {
        let grid: [u8; CELLS] = chunk[..CELLS].try_into().unwrap();
        let obs: [f32; OBS] = f32s(&chunk[CELLS..CELLS + OBS * 4]).try_into().unwrap();
        assert_eq!(observe(&grid), obs, "features differ from Python");
        let want = f32s(&chunk[CELLS + OBS * 4..]);
        let got = logits(&obs);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4 * (1.0 + w.abs()), "{got:?} vs {want:?}");
        }
        assert_eq!(argmax(&got), argmax(&want));
    }
}
