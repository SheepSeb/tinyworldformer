//! Pimoroni Pico Display Pack 2.0 / 2.8: ST7789 320x240 LCD on SPI0 plus four buttons.
//!
//! Landscape: the game frame in the middle (Breakout 16x20 at 12 px cells, Doom-lite 32x32 at 7
//! px), score and lives on the left, loops / timing / who is playing on the right. Only cells
//! that changed are redrawn, so the LCD costs little next to the model.
//!
//! Every game: hold B for 1 s to switch between the agent learning on the device and the
//! pretrained one (the agent then plays). B's game action below fires when a short press ends.
//! Buttons, Breakout: A = left, X = right (held); Pong the same with A = up, X = down; B = hand control to you / back to the agent,
//! Y = cycle the number of loops.
//! Buttons, Doom-lite: A / X = turn left / right, B = forward, Y = fire (held = repeat), A+X
//! together = cycle the loops. Any press takes over; the agent comes back after 8 s idle.
//! Buttons, Flappy Bird: any single button flaps; A+X together cycle the loops; the agent comes
//! back after 8 s idle.
//! Buttons, Pac-Man lite: B = up, X = right, Y = down, A = left (Pac-Man keeps going between
//! presses); A+X together cycle the loops; the agent comes back after 8 s idle.
//!
//! Pinout from Pimoroni's Display Pack docs. If the colours come out inverted or the picture is
//! upside down, flip `INVERT` or `ROTATION` below.

use core::fmt::Write as _;
use core::sync::atomic::Ordering;

use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::peripherals::{PIN_12, PIN_13, PIN_14, PIN_15, PIN_16, PIN_17, PIN_18, PIN_19, PIN_20, PIN_6, PIN_7, PIN_8, SPI0};
use embassy_rp::spi::{self, Blocking, Spi};
use embassy_rp::Peri;
use embassy_time::{Delay, Duration, Instant, Timer};
use embedded_graphics::mono_font::ascii::{FONT_6X10, FONT_9X15_BOLD};
use embedded_graphics::mono_font::MonoTextStyleBuilder;
use embedded_graphics::pixelcolor::Rgb565;
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::{PrimitiveStyle, Rectangle};
use embedded_graphics::text::Text;
use embedded_hal_bus::spi::ExclusiveDevice;
use mipidsi::interface::SpiInterface;
use mipidsi::models::ST7789;
use mipidsi::options::{ColorInversion, Orientation, Rotation};
use mipidsi::Builder;
use static_cell::StaticCell;
use tinyworldformer_rs::dream::color;
use tinyworldformer_rs::weights::{H, LOOPS, W};

use crate::{Frame, DISPLAY_FRAMES, DOOM, FLAPPY, HUMAN, KEY_ACTION, LEARNING, LOOPS_SET, PACMAN, PER_PRESS, PONG};

const INVERT: ColorInversion = ColorInversion::Inverted; // the Display Pack's IPS panel needs it
const ROTATION: Rotation = Rotation::Deg90;
const SPI_HZ: u32 = 62_500_000; // ST7789 limit; the RP2350 rounds down to what clk_peri allows
const CELL: i32 = 240 / H as i32; // Breakout 12 px, Doom-lite 7 px, Flappy Bird 10 px, Pac-Man 5 px
const BOARD_X: i32 = (320 - W as i32 * CELL) / 2; // side panels: Breakout 64 px, Doom-lite 48 px, Flappy and Pac-Man 40 px
const BOARD_Y: i32 = (240 - H as i32 * CELL) / 2;
const AGENT_RESUMES: Duration = Duration::from_secs(8); // idle time before the agent plays again
const HOLD_TO_SWITCH: Duration = Duration::from_millis(1000); // holding B this long: learning <-> pretrained
const BG: Rgb565 = Rgb565::BLACK;

type Lcd = mipidsi::Display<
    SpiInterface<'static, ExclusiveDevice<Spi<'static, SPI0, Blocking>, Output<'static>, Delay>, Output<'static>>,
    ST7789,
    mipidsi::NoResetPin,
>;

pub struct Pins {
    pub spi: Peri<'static, SPI0>,
    pub sck: Peri<'static, PIN_18>,
    pub mosi: Peri<'static, PIN_19>,
    pub cs: Peri<'static, PIN_17>,
    pub dc: Peri<'static, PIN_16>,
    pub backlight: Peri<'static, PIN_20>,
    pub a: Peri<'static, PIN_12>,
    pub b: Peri<'static, PIN_13>,
    pub x: Peri<'static, PIN_14>,
    pub y: Peri<'static, PIN_15>,
    pub led_r: Peri<'static, PIN_6>,
    pub led_g: Peri<'static, PIN_7>,
    pub led_b: Peri<'static, PIN_8>,
}

pub struct Buttons {
    a: Input<'static>,
    b: Input<'static>,
    x: Input<'static>,
    y: Input<'static>,
    /// RGB LED (active low): green while the agent plays, blue while you do.
    led: [Output<'static>; 3],
}

pub fn init(p: Pins) -> (Lcd, Buttons) {
    let mut config = spi::Config::default();
    config.frequency = SPI_HZ;
    let spi = Spi::new_blocking_txonly(p.spi, p.sck, p.mosi, config);
    let cs = Output::new(p.cs, Level::High);
    let device = ExclusiveDevice::new(spi, cs, Delay).unwrap();
    static BUF: StaticCell<[u8; 512]> = StaticCell::new();
    let di = SpiInterface::new(device, Output::new(p.dc, Level::Low), BUF.init([0; 512]));
    let mut lcd = Builder::new(ST7789, di)
        .display_size(240, 320)
        .orientation(Orientation::new().rotate(ROTATION))
        .invert_colors(INVERT)
        .init(&mut Delay)
        .unwrap();
    lcd.clear(BG).unwrap();
    core::mem::forget(Output::new(p.backlight, Level::High)); // keep the backlight on forever

    // buttons pull to ground
    let buttons = Buttons {
        a: Input::new(p.a, Pull::Up),
        b: Input::new(p.b, Pull::Up),
        x: Input::new(p.x, Pull::Up),
        y: Input::new(p.y, Pull::Up),
        led: [Output::new(p.led_r, Level::High), Output::new(p.led_g, Level::Low), Output::new(p.led_b, Level::High)],
    };
    (lcd, buttons)
}

fn rgb565((r, g, b): (u8, u8, u8)) -> Rgb565 {
    Rgb565::new(r >> 3, g >> 2, b >> 3)
}

#[embassy_executor::task]
pub async fn draw(mut lcd: Lcd) {
    let mut shown = [u8::MAX; H * W]; // forces a full board draw on the first frame
    let mut last_panel = (i32::MIN, u8::MAX, u32::MAX, u8::MAX, (false, false), u32::MAX, (i32::MIN, u32::MAX));
    let font = if BOARD_X >= 60 { &FONT_9X15_BOLD } else { &FONT_6X10 }; // 6 characters must fit a panel
    let text = MonoTextStyleBuilder::new().font(font).text_color(Rgb565::WHITE).background_color(BG).build();
    loop {
        let f: Frame = DISPLAY_FRAMES.receive().await;
        for (i, (&tok, old)) in f.grid.iter().zip(shown.iter_mut()).enumerate() {
            if tok != *old {
                let (r, c) = (i / W, i % W);
                let at = Point::new(BOARD_X + c as i32 * CELL, BOARD_Y + r as i32 * CELL);
                let _ = Rectangle::new(at, Size::new(CELL as u32, CELL as u32))
                    .into_styled(PrimitiveStyle::with_fill(rgb565(color(r, tok))))
                    .draw(&mut lcd);
                *old = tok;
            }
        }
        // side panels: only when something shown there changed (fps is bucketed to avoid flicker)
        let fps10 = 10_000_000 / (f.model_us + f.agent_us).max(1);
        let panel = (f.score, f.lives, f.episode, f.loops, (f.human, f.learning), fps10 / 5, (f.avg10, f.learn_steps / 1000));
        if panel != last_panel {
            last_panel = panel;
            let mut s: heapless_str::Str<16> = Default::default();
            let mut put = |x: i32, y: i32, args: core::fmt::Arguments| {
                s.clear();
                let _ = s.write_fmt(args);
                let _ = Text::new(s.as_str(), Point::new(x, y), text).draw(&mut lcd);
            };
            put(4, 20, format_args!("SCORE"));
            put(4, 38, format_args!("{:<6}", f.score));
            put(4, 70, format_args!("{}", if DOOM { "HEALTH" } else if FLAPPY { "ALIVE" } else if PONG { "TO LOSE" } else { "LIVES" }));
            put(4, 88, format_args!("{:<6}", f.lives));
            put(4, 120, format_args!("EP {:<3}", f.episode % 1000));
            put(4, 152, format_args!("AVG10"));
            put(4, 170, format_args!("{:<6}", f.avg10));
            put(4, 228, format_args!("{}", if DOOM { "A< B^" } else if FLAPPY { "FLAP" } else if PACMAN { "A< B^" } else if PONG { "A ^" } else { "A <" }));
            let rx = BOARD_X + W as i32 * CELL + 4;
            put(rx, 20, format_args!("LOOPS"));
            put(rx, 38, format_args!("{:<6}", f.loops));
            put(rx, 70, format_args!("FPS"));
            put(rx, 88, format_args!("{:>3}.{}", fps10 / 10, fps10 % 10)); // fixed width: no stale digits
            put(rx, 120, format_args!("MS"));
            put(rx, 138, format_args!("{:<6}", (f.model_us + 500) / 1000));
            put(rx, 170, format_args!("{:<6}", if f.human { "YOU" } else if f.learning { "LEARN" } else { "AGENT" }));
            put(rx, 188, format_args!("{:>5}k", f.learn_steps / 1000)); // fixed width: no stale digits
            put(rx, 228, format_args!("{}", if DOOM { "X> Y*" } else if FLAPPY { "FLAP" } else if PACMAN { "X> Yv" } else if PONG { "X v" } else { "X >" }));
        }
    }
}

#[embassy_executor::task]
pub async fn buttons(mut btn: Buttons) {
    let (mut b_was, mut y_was, mut ax_was, mut flap_was) = (false, false, false, false);
    let mut last_press = Instant::now();
    let (mut b_since, mut b_held_long) = (Instant::now(), false);
    loop {
        let (a, b, x, y) = (btn.a.is_low(), btn.b.is_low(), btn.x.is_low(), btn.y.is_low());
        // B: a 1 s hold switches learning <-> pretrained; a shorter press does B's game action on release
        if b && !b_was {
            (b_since, b_held_long) = (Instant::now(), false);
        }
        if b && !b_held_long && Instant::now() - b_since >= HOLD_TO_SWITCH {
            LEARNING.fetch_xor(true, Ordering::Relaxed);
            HUMAN.store(false, Ordering::Relaxed); // show the agent that was just switched in
            b_held_long = true;
        }
        let b_tap = !b && b_was && !b_held_long;
        if PER_PRESS {
            let flap = a || x || y;
            if a && x {
                if !ax_was {
                    cycle_loops();
                }
            } else if FLAPPY {
                if (flap && !flap_was) || b_tap {
                    HUMAN.store(true, Ordering::Relaxed);
                    KEY_ACTION.store(1, Ordering::Relaxed); // one flap per press; core 1 consumes it
                }
            } else if PACMAN {
                if let Some(action) = [(b_tap, 1), (x, 2), (y, 3), (a, 4)].iter().find(|(held, _)| *held).map(|&(_, act)| act) {
                    HUMAN.store(true, Ordering::Relaxed);
                    KEY_ACTION.store(action, Ordering::Relaxed); // a turn; core 1 consumes it, then Pac-Man keeps going
                }
            } else if let Some(action) = [(b_tap, 1), (a, 2), (x, 3), (y, 4)].iter().find(|(held, _)| *held).map(|&(_, act)| act) {
                HUMAN.store(true, Ordering::Relaxed);
                KEY_ACTION.store(action, Ordering::Relaxed); // core 1 consumes it, we re-store it while held
            }
            if a || x || y || b_tap {
                last_press = Instant::now();
            } else if !b && Instant::now() - last_press > AGENT_RESUMES {
                HUMAN.store(false, Ordering::Relaxed);
            }
            ax_was = a && x;
            flap_was = flap;
        } else {
            if a || x {
                HUMAN.store(true, Ordering::Relaxed);
                KEY_ACTION.store(if a { 1 } else { 2 }, Ordering::Relaxed);
            } else {
                KEY_ACTION.store(0, Ordering::Relaxed); // released: paddle stops
            }
            if b_tap {
                HUMAN.store(!HUMAN.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            if y && !y_was {
                cycle_loops();
            }
        }
        (b_was, y_was) = (b, y);
        let human = HUMAN.load(Ordering::Relaxed);
        btn.led[1].set_level(if human { Level::High } else { Level::Low });
        btn.led[2].set_level(if human { Level::Low } else { Level::High });
        Timer::after_millis(20).await; // poll: also debounces
    }
}

fn cycle_loops() {
    let next = LOOPS_SET.load(Ordering::Relaxed) as usize % LOOPS + 1;
    LOOPS_SET.store(next as u8, Ordering::Relaxed);
}

/// Tiny fixed-capacity string for formatting panel text without an allocator.
mod heapless_str {
    pub struct Str<const N: usize> {
        buf: [u8; N],
        len: usize,
    }

    impl<const N: usize> Default for Str<N> {
        fn default() -> Self {
            Self { buf: [0; N], len: 0 }
        }
    }

    impl<const N: usize> Str<N> {
        pub fn clear(&mut self) {
            self.len = 0;
        }

        pub fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
        }
    }

    impl<const N: usize> core::fmt::Write for Str<N> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let end = (self.len + s.len()).min(N);
            self.buf[self.len..end].copy_from_slice(&s.as_bytes()[..end - self.len]);
            self.len = end;
            Ok(())
        }
    }
}
