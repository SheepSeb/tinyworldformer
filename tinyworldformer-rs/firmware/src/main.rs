//! Pico 2 W firmware: a game dreamed by an int8 looped transformer.
//!
//! Core 1 runs the world model and the stream-x policy in a tight loop. Core 0 runs USB and
//! streams coloured frames plus per-frame timings over USB serial (open the port in any
//! terminal, e.g. `picocom /dev/ttyACM0`). Keys (they take over from the agent): Breakout a/d/s
//! move; Doom-lite w forward, a/d turn, space fire, s wait; Flappy Bird space / w flaps;
//! Pac-Man w/a/s/d turn, space keeps going; Pong w/s move, space stops. g gives control back to the agent,
//! 1-4 sets the number of loops.
//!
//! The world model ("env") and the agent are chosen at build time: `../flash.sh --env <env>
//! --agent <agent>` (or `TWF_ENV` / `TWF_AGENT`, see `../build.rs`).
//!
//! With `--features display`, core 0 also draws on a Pimoroni Pico Display Pack 2.0/2.8 and
//! reads its buttons (see `display.rs`).
#![no_std]
#![no_main]

#[cfg(feature = "display")]
mod display;

use core::fmt::Write as _;
use core::ptr::addr_of_mut;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_rp::bind_interrupts;
use embassy_rp::clocks::{ClockConfig, CoreVoltage};
use embassy_rp::config::Config;
use embassy_rp::executor::Executor;
use embassy_rp::multicore::{spawn_core1, Stack};
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, InterruptHandler};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_time::{block_for, Duration, Instant};
use embassy_usb::class::cdc_acm::{CdcAcmClass, Receiver, Sender, State};
use embassy_usb::UsbDevice;
use panic_halt as _;
use static_cell::StaticCell;
use tinyworldformer_rs::agent;
use tinyworldformer_rs::dream::{render_ansi, Dream};
use tinyworldformer_rs::learner::Learner;
use tinyworldformer_rs::model::Engine;
use tinyworldformer_rs::{AGENT_NAME, ENV_NAME};
use tinyworldformer_rs::weights::{GAME, H, HUMAN_FRAME_MS, LOOPS, MAX_STEPS, W};

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [embassy_rp::binary_info::EntryAddr; 2] = [
    embassy_rp::binary_info::rp_program_name!(c"tinyworldformer"),
    embassy_rp::binary_info::rp_program_description!(c"A game dreamed by an int8 looped transformer"),
];

const DOOM: bool = matches!(GAME.as_bytes(), b"doomlite");
const FLAPPY: bool = matches!(GAME.as_bytes(), b"flappy");
const PACMAN: bool = matches!(GAME.as_bytes(), b"pacman");
const PONG: bool = matches!(GAME.as_bytes(), b"pong");
/// Doom-lite, Flappy Bird and Pac-Man take one action per press (Pac-Man then keeps going);
/// Breakout and Pong hold the last direction.
const PER_PRESS: bool = DOOM || FLAPPY || PACMAN;
/// Opt-in overclock, e.g. `Some(200_000_000)`. Not officially supported on the RP2350: the core
/// voltage below is the guess from embassy's `overclock` example. `None` keeps the stock 150 MHz.
const OVERCLOCK_HZ: Option<u32> = None;
const HUMAN_FRAME: Duration = Duration::from_millis(HUMAN_FRAME_MS); // slow enough to play

#[derive(Clone, Copy)]
struct Frame {
    grid: [u8; H * W],
    score: i32,
    lives: u8,
    episode: u32,
    model_us: u32,
    agent_us: u32,
    loops: u8,
    human: bool,
    learning: bool,
    learn_steps: u32,
    avg10: i32, // mean score of the last 10 finished episodes
}

static FRAMES: Channel<CriticalSectionRawMutex, Frame, 2> = Channel::new();
#[cfg(feature = "display")]
static DISPLAY_FRAMES: Channel<CriticalSectionRawMutex, Frame, 2> = Channel::new();
static HUMAN: AtomicBool = AtomicBool::new(false);
static KEY_ACTION: AtomicU8 = AtomicU8::new(0);
/// On-device learning: a fresh stream-x agent learns inside the dream from boot. Off: the
/// pretrained agent from `agent_weights.rs` plays. USB `l` toggles, `r` starts learning over.
static LEARNING: AtomicBool = AtomicBool::new(true);
static RESET_LEARNER: AtomicBool = AtomicBool::new(false);
static LOOPS_SET: AtomicU8 = AtomicU8::new(if LOOPS < 2 { LOOPS as u8 } else { 2 }); // 2 loops: best speed/quality trade-off

static mut CORE1_STACK: Stack<16384> = Stack::new();
// 114-368 KB of activations depending on the game; a const-initialised static so it never touches a stack.
static mut ENGINE: Engine = Engine::new();
// ~85 KB of weights and eligibility traces for the on-device learner.
static mut LEARNER: Learner = Learner::zeroed(1);
static EXECUTOR0: StaticCell<Executor> = StaticCell::new();

#[cortex_m_rt::entry]
fn main() -> ! {
    let config = match OVERCLOCK_HZ.and_then(|hz| ClockConfig::system_freq(hz).ok()) {
        Some(clocks) => {
            let mut config = Config::new(clocks);
            config.clocks.core_voltage = CoreVoltage::V1_15;
            config
        }
        None => Config::default(),
    };
    let p = embassy_rp::init(config);
    #[cfg(feature = "display")]
    let (lcd, buttons) = display::init(display::Pins {
        spi: p.SPI0, sck: p.PIN_18, mosi: p.PIN_19, cs: p.PIN_17, dc: p.PIN_16, backlight: p.PIN_20,
        a: p.PIN_12, b: p.PIN_13, x: p.PIN_14, y: p.PIN_15, led_r: p.PIN_6, led_g: p.PIN_7, led_b: p.PIN_8,
    });
    spawn_core1(p.CORE1, unsafe { &mut *addr_of_mut!(CORE1_STACK) }, || dream_loop());
    let executor0 = EXECUTOR0.init(Executor::new());
    executor0.run(|spawner| {
        spawner.spawn(usb_main(spawner, p.USB).unwrap());
        #[cfg(feature = "display")]
        {
            spawner.spawn(display::draw(lcd).unwrap());
            spawner.spawn(display::buttons(buttons).unwrap());
        }
    })
}

/// Core 1: dream a frame, let the agent (or the player) act, learn from it, publish, repeat.
fn dream_loop() -> ! {
    // SAFETY: only core 1 ever touches ENGINE and LEARNER.
    let engine = unsafe { &mut *addr_of_mut!(ENGINE) };
    let learner = unsafe { &mut *addr_of_mut!(LEARNER) };
    let seed = Instant::now().as_ticks() as u32 | 1;
    let mut dream = Dream::new(seed);
    learner.reset(seed ^ 0x5eed);
    let mut x = learner.normalize(&agent::observe(&dream.grid), true);
    let (mut score, mut wins, mut episode, mut learn_steps) = (0i32, 0i32, 0u32, 0u32);
    let (mut recent, mut n_recent) = ([0i32; 10], 0usize);
    loop {
        if RESET_LEARNER.swap(false, Ordering::Relaxed) {
            learner.reset(Instant::now().as_ticks() as u32 | 1);
            (learn_steps, n_recent) = (0, 0);
            x = learner.normalize(&agent::observe(&dream.grid), true);
        }
        let t0 = Instant::now();
        let human = HUMAN.load(Ordering::Relaxed);
        let learning = LEARNING.load(Ordering::Relaxed) && !human;
        let action = match (human, PER_PRESS) {
            (true, true) => KEY_ACTION.swap(0, Ordering::Relaxed) as usize, // consumed: one step per press
            (true, false) => KEY_ACTION.load(Ordering::Relaxed) as usize,
            (false, _) if learning => learner.act(&x, false),
            (false, _) => agent::act(&dream.grid),
        };
        let t1 = Instant::now();
        let loops = LOOPS_SET.load(Ordering::Relaxed) as usize;
        let (reward, done) = dream.step(engine, action, loops, MAX_STEPS);
        let t2 = Instant::now();
        // learn from this transition (normalization keeps tracking frames while it is off, so
        // switching it back on starts from a current observation)
        let x2 = learner.normalize(&agent::observe(&dream.grid), learning);
        if learning {
            let terminal = done && dream.steps < MAX_STEPS; // out of lives, not out of time
            let r = learner.scale_reward(reward as f32, terminal);
            learner.update(&x, action, r, &x2, terminal, done);
            learn_steps += 1;
        }
        x = x2;
        let t3 = Instant::now();
        score += reward;
        wins += reward.max(0);
        let avg10 = if n_recent == 0 { 0 } else { recent[..n_recent.min(10)].iter().sum::<i32>() / n_recent.min(10) as i32 };
        let frame = Frame {
            grid: dream.grid,
            score,
            lives: dream.lives,
            episode,
            model_us: (t2 - t1).as_micros() as u32,
            agent_us: ((t1 - t0) + (t3 - t2)).as_micros() as u32,
            loops: loops as u8,
            human,
            learning,
            learn_steps,
            avg10,
        };
        let _ = FRAMES.try_send(frame);
        #[cfg(feature = "display")]
        let _ = DISPLAY_FRAMES.try_send(frame);
        if done {
            recent[n_recent % 10] = wins;
            n_recent += 1;
            dream.reset();
            x = learner.normalize(&agent::observe(&dream.grid), learning);
            (score, wins) = (0, 0);
            episode += 1;
        }
        if human {
            let spent = Instant::now() - t0;
            if spent < HUMAN_FRAME {
                block_for(HUMAN_FRAME - spent);
            }
        }
    }
}

type UsbDriver = Driver<'static, USB>;

#[embassy_executor::task]
async fn usb_run(mut usb: UsbDevice<'static, UsbDriver>) -> ! {
    usb.run().await
}

#[embassy_executor::task]
async fn usb_main(spawner: Spawner, usb: embassy_rp::Peri<'static, USB>) {
    let driver = Driver::new(usb, Irqs);
    let mut config = embassy_usb::Config::new(0xc0de, 0xcafe); // test VID/PID, fine for a hobby device
    config.manufacturer = Some("tinyworldformer");
    config.product = Some("dreamed games");
    config.serial_number = Some("0001");
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static STATE: StaticCell<State> = StaticCell::new();
    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        CONFIG_DESC.init([0; 256]),
        BOS_DESC.init([0; 256]),
        &mut [],
        CONTROL_BUF.init([0; 64]),
    );
    let class = CdcAcmClass::new(&mut builder, STATE.init(State::new()), 64);
    spawner.spawn(usb_run(builder.build()).unwrap());

    let (tx, rx) = class.split();
    join(write_frames(tx), read_keys(rx)).await;
}

async fn read_keys(mut rx: Receiver<'static, UsbDriver>) {
    let mut buf = [0u8; 64];
    loop {
        rx.wait_connection().await;
        while let Ok(n) = rx.read_packet(&mut buf).await {
            for &key in &buf[..n] {
                let set_action = |a: u8| {
                    KEY_ACTION.store(a, Ordering::Relaxed);
                    HUMAN.store(true, Ordering::Relaxed);
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
                    b'g' => HUMAN.store(false, Ordering::Relaxed),
                    b'l' => {
                        LEARNING.fetch_xor(true, Ordering::Relaxed);
                    }
                    b'r' => RESET_LEARNER.store(true, Ordering::Relaxed),
                    b'1'..=b'9' if ((key - b'0') as usize) <= LOOPS => LOOPS_SET.store(key - b'0', Ordering::Relaxed),
                    _ => {}
                }
            }
        }
    }
}

/// Fixed-size text buffer for one rendered frame.
struct Buf {
    data: [u8; 8192],
    len: usize,
}

impl core::fmt::Write for Buf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let end = self.len + s.len();
        if end > self.data.len() {
            return Err(core::fmt::Error);
        }
        self.data[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

fn render(f: &Frame, out: &mut Buf) -> core::fmt::Result {
    render_ansi(&f.grid, out)?;
    let frame_us = (f.model_us + f.agent_us).max(1);
    write!(
        out,
        "score {:3}  lives {}  episode {}  | {} loops: model {:.1} ms, agent {} us, {:.1} fps | {}\x1b[K\r\n\
         learned for {} steps, mean score of the last 10 episodes {}\x1b[K\r\n\
         keys: {}, g agent, l learning on/off, r learn from scratch, 1-{} loops | env {}, agent {}\x1b[K\r\n",
        f.score, f.lives, f.episode, f.loops, f.model_us as f32 / 1000.0, f.agent_us,
        1e6 / frame_us as f32,
        match (f.human, f.learning) {
            (true, _) => "you",
            (false, true) => "stream-x LEARNING",
            (false, false) => "pretrained stream-x agent",
        },
        f.learn_steps,
        f.avg10,
        match (DOOM, FLAPPY, PACMAN) {
            (true, _, _) => "w forward, a/d turn, space fire, s wait",
            (_, true, _) => "space / w flaps",
            (_, _, true) => "w/a/s/d turn, space keeps going",
            _ if PONG => "w/s move, space stops",
            _ => "a/d/s move",
        },
        LOOPS,
        ENV_NAME,
        AGENT_NAME,
    )
}

async fn write_frames(mut tx: Sender<'static, UsbDriver>) {
    let mut buf = Buf { data: [0; 8192], len: 0 };
    loop {
        tx.wait_connection().await;
        let _ = tx.write_packet(b"\x1b[2J").await;
        loop {
            let frame = FRAMES.receive().await;
            buf.len = 0;
            if render(&frame, &mut buf).is_err() {
                continue;
            }
            let mut sent = Ok(());
            for chunk in buf.data[..buf.len].chunks(64) {
                sent = tx.write_packet(chunk).await;
                if sent.is_err() {
                    break;
                }
            }
            if sent.is_err() {
                break; // disconnected: wait for the next connection
            }
        }
    }
}
