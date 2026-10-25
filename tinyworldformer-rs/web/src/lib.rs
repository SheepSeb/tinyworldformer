//! WebAssembly build: the dreamed game in a browser, with exactly the int8 engine that runs on
//! the Pico. Same loop as the firmware's core 1 (`../firmware/src/main.rs`); the page
//! (`www/index.html`) calls `tick` once per frame, sends keys with `key`, and draws `rgba`.
//!
//! No wasm-bindgen: plain `extern "C"` exports, so the module needs no imports at all. The env
//! and agent are baked in at build time like on the Pico (`TWF_ENV` / `TWF_AGENT`, see
//! `../build.rs`); `build.sh` builds one module per env.
#![no_std]

use core::ptr::addr_of_mut;

use tinyworldformer_rs::agent;
use tinyworldformer_rs::dream::{color, Dream};
use tinyworldformer_rs::learner::Learner;
use tinyworldformer_rs::model::Engine;
use tinyworldformer_rs::weights::{GAME, H, LOOPS, MAX_STEPS, W};
use tinyworldformer_rs::{AGENT_NAME, ENV_NAME};

const DOOM: bool = matches!(GAME.as_bytes(), b"doomlite");
const FLAPPY: bool = matches!(GAME.as_bytes(), b"flappy");
const PACMAN: bool = matches!(GAME.as_bytes(), b"pacman");
const PONG: bool = matches!(GAME.as_bytes(), b"pong");
/// Doom-lite, Flappy Bird and Pac-Man take one action per press (Pac-Man then keeps going);
/// Breakout and Pong hold the last direction.
const PER_PRESS: bool = DOOM || FLAPPY || PACMAN;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// Everything the loop keeps between frames (the locals of the firmware's `dream_loop`).
struct State {
    dream: Dream,
    x: [f32; tinyworldformer_rs::agent_weights::OBS],
    score: i32,
    wins: i32,
    episode: u32,
    learn_steps: u32,
    recent: [i32; 10],
    n_recent: usize,
    human: bool,
    key_action: u8,
    learning: bool,
    loops: u8,
}

// Single-threaded: only the exports below touch these, one call at a time.
static mut ENGINE: Engine = Engine::new();
static mut LEARNER: Learner = Learner::zeroed(1);
static mut STATE: Option<State> = None;
static mut RGBA: [u8; H * W * 4] = [0; H * W * 4];

fn state() -> &'static mut State {
    // SAFETY: wasm is single-threaded and no reference outlives an export call.
    unsafe { (*addr_of_mut!(STATE)).as_mut().expect("init first") }
}

fn learner() -> &'static mut Learner {
    // SAFETY: as above.
    unsafe { &mut *addr_of_mut!(LEARNER) }
}

fn draw(grid: &[u8; H * W]) {
    // SAFETY: as above.
    let rgba = unsafe { &mut *addr_of_mut!(RGBA) };
    for (i, (px, &tok)) in rgba.chunks_exact_mut(4).zip(grid).enumerate() {
        let (r, g, b) = color(i / W, tok);
        px.copy_from_slice(&[r, g, b, 255]);
    }
}

/// Start (or restart) with a seed from the page.
#[unsafe(no_mangle)]
pub extern "C" fn init(seed: u32) {
    let seed = seed | 1;
    let dream = Dream::new(seed);
    let learner = learner();
    learner.reset(seed ^ 0x5eed);
    let x = learner.normalize(&agent::observe(&dream.grid), true);
    draw(&dream.grid);
    // SAFETY: as above.
    unsafe {
        *addr_of_mut!(STATE) = Some(State {
            dream, x, score: 0, wins: 0, episode: 0, learn_steps: 0, recent: [0; 10], n_recent: 0,
            human: false, key_action: 0, learning: false, loops: if LOOPS < 2 { LOOPS as u8 } else { 2 },
        });
    }
}

/// One frame: the agent (or the player) acts, the network dreams the next frame, the learner
/// learns from it when on. Same order as the firmware.
#[unsafe(no_mangle)]
pub extern "C" fn tick() {
    let s = state();
    let learner = learner();
    // SAFETY: as above.
    let engine = unsafe { &mut *addr_of_mut!(ENGINE) };
    let learning = s.learning && !s.human;
    let action = match (s.human, PER_PRESS) {
        (true, true) => core::mem::take(&mut s.key_action) as usize, // consumed: one step per press
        (true, false) => s.key_action as usize,
        (false, _) if learning => learner.act(&s.x, false),
        (false, _) => agent::act(&s.dream.grid),
    };
    let (reward, done) = s.dream.step(engine, action, s.loops as usize, MAX_STEPS);
    let x2 = learner.normalize(&agent::observe(&s.dream.grid), learning);
    if learning {
        let terminal = done && s.dream.steps < MAX_STEPS; // out of lives, not out of time
        let r = learner.scale_reward(reward as f32, terminal);
        learner.update(&s.x, action, r, &x2, terminal, done);
        s.learn_steps += 1;
    }
    s.x = x2;
    s.score += reward;
    s.wins += reward.max(0);
    draw(&s.dream.grid);
    if done {
        s.recent[s.n_recent % 10] = s.wins;
        s.n_recent += 1;
        s.dream.reset();
        s.x = learner.normalize(&agent::observe(&s.dream.grid), learning);
        (s.score, s.wins) = (0, 0);
        s.episode += 1;
    }
}

/// A key, as the firmware reads them over USB serial (the page maps arrows to w/a/s/d).
#[unsafe(no_mangle)]
pub extern "C" fn key(key: u8) {
    let s = state();
    let mut set_action = |a: u8| {
        s.key_action = a;
        s.human = true;
    };
    match key {
        b'w' if PONG => set_action(1),
        b's' if PONG => set_action(2),
        b'w' if PACMAN => set_action(1),
        b'd' if PACMAN => set_action(2),
        b's' if PACMAN => set_action(3),
        b'a' if PACMAN => set_action(4),
        b' ' | b'w' if FLAPPY => set_action(1),
        b's' => set_action(0),
        b' ' => set_action(if DOOM { 4 } else { 0 }),
        b'w' if DOOM => set_action(1),
        b'a' if !FLAPPY => set_action(if DOOM { 2 } else { 1 }),
        b'd' if !FLAPPY => set_action(if DOOM { 3 } else { 2 }),
        b'g' => s.human = false,
        b'l' => s.learning = !s.learning,
        b'r' => {
            learner().reset(s.dream.rng.next_u32() | 1);
            (s.learn_steps, s.n_recent) = (0, 0);
            s.x = learner().normalize(&agent::observe(&s.dream.grid), true);
        }
        b'1'..=b'9' if ((key - b'0') as usize) <= LOOPS => s.loops = key - b'0',
        _ => {}
    }
}

/// The current frame, `height() * width()` RGBA pixels.
#[unsafe(no_mangle)]
pub extern "C" fn rgba() -> *const u8 {
    addr_of_mut!(RGBA).cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn width() -> u32 { W as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn height() -> u32 { H as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn max_loops() -> u32 { LOOPS as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn human_frame_ms() -> u32 { tinyworldformer_rs::weights::HUMAN_FRAME_MS as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn score() -> i32 { state().score }
#[unsafe(no_mangle)]
pub extern "C" fn wins() -> i32 { state().wins }
#[unsafe(no_mangle)]
pub extern "C" fn lives() -> u32 { state().dream.lives as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn episode() -> u32 { state().episode }
#[unsafe(no_mangle)]
pub extern "C" fn loops() -> u32 { state().loops as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn human() -> u32 { state().human as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn learning() -> u32 { state().learning as u32 }
#[unsafe(no_mangle)]
pub extern "C" fn learn_steps() -> u32 { state().learn_steps }

/// Mean score of the last 10 finished episodes (0 before the first).
#[unsafe(no_mangle)]
pub extern "C" fn avg10() -> i32 {
    let s = state();
    let n = s.n_recent.min(10);
    if n == 0 { 0 } else { s.recent[..n].iter().sum::<i32>() / n as i32 }
}

/// Names, as (pointer, length) pairs: `name_ptr(i)` / `name_len(i)` with 0 game, 1 env, 2 agent.
fn name(i: u32) -> &'static str {
    match i {
        0 => GAME,
        1 => ENV_NAME,
        _ => AGENT_NAME,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn name_ptr(i: u32) -> *const u8 { name(i).as_ptr() }
#[unsafe(no_mangle)]
pub extern "C" fn name_len(i: u32) -> u32 { name(i).len() as u32 }
