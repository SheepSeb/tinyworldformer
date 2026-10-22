//! The game as the network dreams it: frame + action -> next frame, reward.
//!
//! Decoding mirrors `model.decode`: every cell takes its argmax among the background tokens, then
//! each singleton group (Breakout's ball, Doom-lite's minimap enemy and player) is placed exactly
//! once, sampled among the (cell, token) pairs whose logit is within `WINDOW` of the best. That
//! sample is how the random events (the serve, the respawn) get dreamed. Last, `STAMPS` draws the
//! rest of each placed sprite (Pac-Man's body, the ghosts' eyes) around it. Lives, score and the
//! episode end are plain counters on top of the predicted reward. Everything game-specific comes
//! from the exported `weights.rs` of the env this build picked (`build.rs`).

use crate::model::{argmax, Engine};
use crate::weights::{BG, CELL_TOKENS, COLORS, DONE_WHEN_ABSENT, H, INIT_GRIDS, LIVES, N_INIT, REWARD_VALUES, SINGLETONS, SINGLETON_MASK, STAMPS, V, W};

/// Candidates whose logit is within this of the best are sampled (exp(-2.2) ~ 0.1).
pub const WINDOW: f32 = 2.2;

pub struct XorShift(pub u32);

impl XorShift {
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

pub struct Dream {
    pub grid: [u8; H * W],
    pub lives: u8,
    pub steps: u32,
    pub rng: XorShift,
}

impl Dream {
    pub fn new(seed: u32) -> Self {
        let mut d = Self { grid: [0; H * W], lives: LIVES, steps: 0, rng: XorShift(seed.max(1)) };
        d.reset();
        d
    }

    /// One of the real game's opening frames, picked at random.
    pub fn reset(&mut self) {
        let k = self.rng.next_u32() as usize % N_INIT;
        self.grid.copy_from_slice(&INIT_GRIDS[k * H * W..(k + 1) * H * W]);
        self.lives = LIVES;
        self.steps = 0;
    }

    /// One dreamed tick. Returns (reward, done); call `reset` after done.
    pub fn step(&mut self, engine: &mut Engine, action: usize, loops: usize, max_steps: u32) -> (i32, bool) {
        engine.forward(&self.grid, action, loops);
        let stamped = |t: usize| STAMPS.iter().any(|s| s.3 as usize == t);
        for r in 0..H {
            for c in 0..W {
                let logits = &engine.cell_logits(r, c)[..BG];
                let allowed = CELL_TOKENS[r * W + c]; // the background tokens this cell can hold
                let best = (0..BG).filter(|&t| !stamped(t) && allowed >> t & 1 == 1).fold(None, |b: Option<usize>, t| match b {
                    Some(b) if logits[b] >= logits[t] => Some(b), // first maximum, like argmax
                    _ => Some(t),
                });
                self.grid[r * W + c] = best.unwrap_or(0) as u8;
            }
        }
        let mut placed = [(usize::MAX, 0u8); SINGLETONS.len()];
        for (k, &(lo, hi, _)) in SINGLETONS.iter().enumerate() {
            let (i, tok) = self.sample(engine, k, lo as usize, hi as usize, &placed[..k]);
            self.grid[i] = tok as u8;
            placed[k] = (i, tok as u8);
        }
        for &(anchor, dr, dc, tok) in STAMPS.iter() {
            for &(i, _) in placed.iter().filter(|p| p.1 == anchor) {
                let (r, c) = ((i / W) as i32 + dr as i32, (i % W) as i32 + dc as i32);
                let j = r as usize * W + c as usize;
                if (0..H as i32).contains(&r) && (0..W as i32).contains(&c) && !placed.iter().any(|p| p.0 == j) {
                    self.grid[j] = tok; // never over another singleton
                }
            }
        }
        let reward = REWARD_VALUES[argmax(&engine.reward)] as i32;
        if reward < 0 {
            self.lives = self.lives.saturating_sub(1);
        }
        self.steps += 1;
        let gone = DONE_WHEN_ABSENT >= 0 && !self.grid.contains(&(DONE_WHEN_ABSENT as u8));
        (reward, self.lives == 0 || gone || self.steps >= max_steps)
    }

    /// (cell, token) for singleton group `group`: a sample among the near-best logits of tokens
    /// lo..hi, over the cells where the real game can have that group (`SINGLETON_MASK`).
    /// Cells already holding an earlier group (`taken`) are skipped: one object per cell.
    fn sample(&mut self, engine: &Engine, group: usize, lo: usize, hi: usize, taken: &[(usize, u8)]) -> (usize, usize) {
        const NONE: &[f32] = &[f32::NEG_INFINITY; 32];
        let mask = &SINGLETON_MASK[group * H * W..(group + 1) * H * W];
        let free = |i: usize| mask[i] != 0 && !taken.iter().any(|p| p.0 == i);
        let logits = |i: usize| if free(i) { &engine.cell_logits(i / W, i % W)[lo..hi] } else { &NONE[..hi - lo] };
        let best = (0..H * W).flat_map(|i| logits(i).iter().copied()).fold(f32::NEG_INFINITY, f32::max);
        let weight = |l: f32| if l >= best - WINDOW { libm::expf(l - best) } else { 0.0 };
        let total: f32 = (0..H * W).map(|i| logits(i).iter().map(|&l| weight(l)).sum::<f32>()).sum();
        let mut u = self.rng.unit() * total;
        let mut chosen = (0, lo);
        for i in 0..H * W {
            for (k, &l) in logits(i).iter().enumerate() {
                let w = weight(l);
                if w > 0.0 {
                    chosen = (i, lo + k); // the last candidate also covers float round-off in `u`
                    if u < w {
                        return chosen;
                    }
                    u -= w;
                }
            }
        }
        chosen
    }
}

/// Colour of a token on a given row, same as the Python game's `colorize`.
pub fn color(row: usize, tok: u8) -> (u8, u8, u8) {
    let c = &COLORS[(row * V + tok as usize) * 3..][..3];
    (c[0], c[1], c[2])
}

/// ANSI truecolor frame (cursor home first); only emits a colour code when the colour changes.
pub fn render_ansi(grid: &[u8; H * W], out: &mut impl core::fmt::Write) -> core::fmt::Result {
    out.write_str("\x1b[H")?;
    for r in 0..H {
        let mut prev = None;
        for c in 0..W {
            let col = color(r, grid[r * W + c]);
            if prev != Some(col) {
                write!(out, "\x1b[48;2;{};{};{}m", col.0, col.1, col.2)?;
                prev = Some(col);
            }
            out.write_str("  ")?;
        }
        out.write_str("\x1b[0m\r\n")?;
    }
    Ok(())
}
