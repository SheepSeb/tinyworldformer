//! Host demo: the dreamed game in your terminal, using exactly the int8 engine that runs on the
//! Pico. The stream-x agent plays until you press a key.
//!
//!     cargo run --release              # watch / play (keys on screen; g agent, 1-4 loops, q quit)
//!     cargo run --release -- --bench   # time the engine, print episode scores
//!     cargo run --release -- --learn   # a fresh stream-x agent learns inside the dream, from scratch
//!
//! The world model ("env") and the agent are chosen at build time:
//!
//!     TWF_ENV=doomlite TWF_AGENT=doom_agent_roam_p0.1 cargo run --release
//!     ./flash.sh --list                # what has been exported
//!
//! (see `build.rs`; without them, `models/default` decides).
use std::io::Write as _;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::{cursor, execute, terminal};
use tinyworldformer_rs::agent;
use tinyworldformer_rs::learner::Learner;
use tinyworldformer_rs::dream::{render_ansi, Dream};
use tinyworldformer_rs::model::Engine;
use tinyworldformer_rs::{AGENT_NAME, ENV_NAME};
use tinyworldformer_rs::weights::{GAME, HUMAN_FRAME_MS, LOOPS, MAX_STEPS};

const FRAME: Duration = Duration::from_millis(HUMAN_FRAME_MS);
const DOOM: bool = matches!(GAME.as_bytes(), b"doomlite");
const FLAPPY: bool = matches!(GAME.as_bytes(), b"flappy");
const PACMAN: bool = matches!(GAME.as_bytes(), b"pacman");
const PONG: bool = matches!(GAME.as_bytes(), b"pong");
/// Doom-lite, Flappy Bird and Pac-Man take one action per key press (Pac-Man then keeps going);
/// Breakout and Pong hold the last direction.
const PER_PRESS: bool = DOOM || FLAPPY || PACMAN;

/// Key -> action. Breakout: a/d move, s stop. Doom-lite: w forward, a/d turn, space fire, s wait.
/// Flappy Bird: space / w / up flap. Pac-Man: w/a/s/d or arrows turn, space keeps going.
/// Pong: w/s or up/down move, space stops.
fn key_action(code: KeyCode) -> Option<usize> {
    if PONG {
        return Some(match code {
            KeyCode::Char('w') | KeyCode::Up => 1,
            KeyCode::Char('s') | KeyCode::Down => 2,
            KeyCode::Char(' ') => 0,
            _ => return None,
        });
    }
    if PACMAN {
        return Some(match code {
            KeyCode::Char('w') | KeyCode::Up => 1,
            KeyCode::Char('d') | KeyCode::Right => 2,
            KeyCode::Char('s') | KeyCode::Down => 3,
            KeyCode::Char('a') | KeyCode::Left => 4,
            KeyCode::Char(' ') => 0,
            _ => return None,
        });
    }
    if FLAPPY {
        return matches!(code, KeyCode::Char(' ' | 'w') | KeyCode::Up).then_some(1);
    }
    Some(match (DOOM, code) {
        (true, KeyCode::Char('w') | KeyCode::Up) => 1,
        (true, KeyCode::Char('a') | KeyCode::Left) => 2,
        (true, KeyCode::Char('d') | KeyCode::Right) => 3,
        (true, KeyCode::Char(' ')) => 4,
        (false, KeyCode::Char('a') | KeyCode::Left) => 1,
        (false, KeyCode::Char('d') | KeyCode::Right) => 2,
        (_, KeyCode::Char('s') | KeyCode::Down) | (false, KeyCode::Char(' ')) => 0,
        _ => return None,
    })
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loops = args.iter().filter_map(|a| a.parse().ok()).next().unwrap_or(2).clamp(1, LOOPS);
    let mut engine = Box::new(Engine::new());
    if args.iter().any(|a| a == "--bench") {
        bench(&mut engine, loops);
        return Ok(());
    }
    if args.iter().any(|a| a == "--learn") {
        learn(&mut engine, loops, 300_000);
        return Ok(());
    }
    let mut out = std::io::stdout();
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide, terminal::Clear(terminal::ClearType::All))?;
    let result = play(&mut engine, loops, &mut out);
    execute!(out, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    result
}

fn play(engine: &mut Engine, mut loops: usize, out: &mut std::io::Stdout) -> std::io::Result<()> {
    let mut dream = Dream::new(Instant::now().elapsed().subsec_nanos() | 0x9e37_79b9);
    let (mut score, mut wins, mut episode, mut last) = (0, 0, 0, String::new());
    let (won, help) = match (DOOM, FLAPPY, PACMAN) {
        (true, _, _) => ("kills", "w forward, a/d turn, space fire, s wait"),
        (_, true, _) => ("pipes", "space / w / up flaps"),
        (_, _, true) => ("pellets", "w/a/s/d or arrows turn, Pac-Man keeps going"),
        _ if PONG => ("points", "w/s or up/down move, space stops"),
        _ => ("bricks", "a/d or arrows move, s stop"),
    };
    let mut human: Option<usize> = None; // Some(action) once you take over
    let mut frame = String::new();
    loop {
        let start = Instant::now();
        let action = human.unwrap_or_else(|| agent::act(&dream.grid));
        let t = Instant::now();
        let (reward, done) = dream.step(engine, action, loops, MAX_STEPS);
        let model_ms = t.elapsed().as_secs_f64() * 1e3;
        score += reward;
        wins += reward.max(0);
        if PER_PRESS && human.is_some() {
            human = Some(0); // one action per key press (holding a key repeats it)
        }
        if done {
            last = format!("last episode: score {score}, {wins} {won}, {} steps", dream.steps);
            dream.reset();
            (score, wins) = (0, 0);
            episode += 1;
        }
        frame.clear();
        render_ansi(&dream.grid, &mut frame).unwrap();
        let who = if human.is_some() { "you" } else { "stream-x agent" };
        write!(
            out,
            "{frame}score {score:3}  {won} {wins}  lives {}  episode {episode}  | {who} playing | {loops} loops, {model_ms:.2} ms/frame\x1b[K\r\n\
             {last}\x1b[K\r\n{help}, g agent plays, 1-{LOOPS} loops, q quit | env {ENV_NAME}, agent {AGENT_NAME}\x1b[K\r\n",
            dream.lives
        )?;
        out.flush()?;
        // read keys until it is time for the next frame
        while let Some(left) = FRAME.checked_sub(start.elapsed()) {
            if !event::poll(left)? {
                break;
            }
            let Event::Key(k) = event::read()? else { continue };
            if k.kind == KeyEventKind::Release {
                continue;
            }
            match k.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => return Ok(()),
                KeyCode::Char('g') => human = None,
                KeyCode::Char(c @ '1'..='9') if (c as usize - '0' as usize) <= LOOPS => loops = c as usize - '0' as usize,
                code => {
                    if let Some(a) = key_action(code) {
                        human = Some(a);
                    }
                }
            }
        }
    }
}

fn bench(engine: &mut Engine, loops: usize) {
    let mut dream = Dream::new(0x9e37_79b9);
    let (mut score, mut wins, mut episode, mut model_time) = (0, 0, 0, Duration::ZERO);
    for _ in 0..20_000 {
        let action = agent::act(&dream.grid);
        let t = Instant::now();
        let (reward, done) = dream.step(engine, action, loops, MAX_STEPS);
        model_time += t.elapsed();
        score += reward;
        wins += reward.max(0);
        if done {
            println!("episode {episode}: score {score}, {wins} positive rewards, {} steps, {} lives left", dream.steps, dream.lives);
            dream.reset();
            (score, wins) = (0, 0);
            episode += 1;
        }
    }
    println!("{loops} loops: {:.1} us per dreamed frame on this host", model_time.as_micros() as f64 / 20_000.0);
}

/// Stream-x learning from scratch inside the dream (what the Pico does with LEARN on).
fn learn(engine: &mut Engine, loops: usize, steps: u32) {
    let mut learner = Box::new(Learner::zeroed(0x5eed));
    learner.init();
    let mut dream = Dream::new(0x9e37_79b9);
    let mut x = learner.normalize(&agent::observe(&dream.grid), true);
    let (mut wins, mut recent, mut episode) = (0i32, Vec::new(), 0u32);
    let start = Instant::now();
    for step in 1..=steps {
        let a = learner.act(&x, false);
        let (reward, done) = dream.step(engine, a, loops, MAX_STEPS);
        wins += reward.max(0);
        let terminal = done && dream.steps < MAX_STEPS; // ran out of lives (or bricks), not time
        let x2 = learner.normalize(&agent::observe(&dream.grid), true);
        let r = learner.scale_reward(reward as f32, terminal);
        learner.update(&x, a, r, &x2, terminal, done);
        x = x2;
        if done {
            recent.push(wins);
            (wins, episode) = (0, episode + 1);
            dream.reset();
            x = learner.normalize(&agent::observe(&dream.grid), true);
        }
        if step % 25_000 == 0 {
            let mean = recent.iter().sum::<i32>() as f32 / recent.len().max(1) as f32;
            println!("step {step:7}: {episode:5} episodes, mean score of the last {:4} {mean:6.1} | {:.0} us/step (dream + learning)",
                     recent.len(), start.elapsed().as_micros() as f64 / step as f64);
            recent.clear();
        }
    }
}
