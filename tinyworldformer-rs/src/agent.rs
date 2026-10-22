//! The stream-x policy network (inference only), same features as `streamx.observe`.

use crate::agent_weights::*;
use crate::model::argmax;
use crate::weights::{ACTIONS, GAME, H, W};

/// Frame -> features for the game this build was exported for (mirrors `streamx.features`).
pub fn observe(grid: &[u8; H * W]) -> [f32; OBS] {
    let mut obs = [0.0f32; OBS];
    match GAME {
        "doomlite" => observe_doom(grid, &mut obs),
        "flappy" => observe_flappy(grid, &mut obs),
        "pacman" => observe_pacman(grid, &mut obs),
        "pong" => observe_pong(grid, &mut obs),
        _ => observe_breakout(grid, &mut obs),
    }
    obs
}

/// Breakout: ball row/col/heading one-hots, paddle column one-hot,
/// ball-minus-paddle column one-hot, brick mask.
fn observe_breakout(grid: &[u8], obs: &mut [f32]) {
    const BRICK: u8 = 2;
    const PADDLE: u8 = 3;
    const BALL: u8 = 4;
    let paddle_row = H - 2;
    let row = &grid[paddle_row * W..(paddle_row + 1) * W];
    let (sum, n) = row.iter().enumerate().filter(|&(_, &c)| c == PADDLE).fold((0, 0), |(s, n), (i, _)| (s + i, n + 1));
    // round half to even, like Python's round()
    let paddle = (n > 0).then(|| libm::rintf(sum as f32 / n as f32) as usize);
    if let Some(i) = grid.iter().position(|&c| c >= BALL) {
        let (r, c) = (i / W, i % W);
        obs[r] = 1.0;
        obs[H + c] = 1.0;
        obs[H + W + (grid[i] - BALL) as usize] = 1.0;
        if let Some(pc) = paddle {
            obs[H + 2 * W + 4 + c + W - 1 - pc] = 1.0;
        }
    }
    if let Some(pc) = paddle {
        obs[H + W + 4 + pc] = 1.0;
    }
    let mut k = H + 2 * W + 4 + 2 * W - 1;
    for r in 2..6 {
        for c in 1..W - 1 {
            obs[k] = (grid[r * W + c] == BRICK) as u8 as f32;
            k += 1;
        }
    }
}

/// Doom-lite: player cell and facing and enemy cell from the minimap, the enemy's offset
/// ahead/right of the player, health from the HUD bar, enemy pixels in the crosshair.
fn observe_doom(grid: &[u8], obs: &mut [f32]) {
    const N: usize = 8;
    const VIEW_H: usize = 24;
    const ENEMY: u8 = 6;
    const ENEMY_EYE: u8 = 7;
    const HEALTH_BAR: u8 = 13;
    const MAP_ENEMY: u8 = 16;
    const MAP_PLAYER: u8 = 17;
    const DIRS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    let mini = |y: usize, x: usize| grid[(VIEW_H + y) * W + W - N + x];
    let find = |want: &dyn Fn(u8) -> bool| (0..N * N).find(|&i| want(mini(i / N, i % N))).map(|i| (i % N, i / N));
    let player = find(&|t| t >= MAP_PLAYER);
    let enemy = find(&|t| t == MAP_ENEMY);
    if let Some((px, py)) = player {
        obs[py * N + px] = 1.0;
        obs[N * N + (mini(py, px) - MAP_PLAYER) as usize] = 1.0;
    }
    let k = N * N + 4;
    if let Some((ex, ey)) = enemy {
        obs[k + ey * N + ex] = 1.0;
        if let Some((px, py)) = player {
            let (dx, dy) = DIRS[(mini(py, px) - MAP_PLAYER) as usize];
            let (rx, ry) = (ex as i32 - px as i32, ey as i32 - py as i32);
            let ahead = rx * dx + ry * dy;
            let right = rx * -dy + ry * dx;
            obs[k + N * N + (ahead + N as i32 - 1) as usize] = 1.0;
            obs[k + N * N + 2 * N - 1 + (right + N as i32 - 1) as usize] = 1.0;
        }
    }
    let k = k + N * N + 2 * (2 * N - 1);
    let health = grid[27 * W..28 * W].iter().filter(|&&t| t == HEALTH_BAR).count() / 3;
    obs[k + health.min(5)] = 1.0;
    let crosshair = (0..VIEW_H).any(|r| (15..17).any(|c| matches!(grid[r * W + c], ENEMY | ENEMY_EYE)));
    obs[OBS - 1] = crosshair as u8 as f32;
}

/// Flappy Bird: bird row and speed, distance to the next pipe, that pipe's gap centre and the
/// next-gap marker, both relative to the bird.
fn observe_flappy(grid: &[u8], obs: &mut [f32]) {
    const ROWS: usize = 22; // rows the bird can be in (the ground starts at row 22)
    const PLAY_W: usize = W - 1;
    const BIRD_X: usize = 6;
    const GAP: usize = 7;
    const PIPE: u8 = 5;
    const PIPE_CAP: u8 = 7;
    const MARKER: u8 = 10;
    const BIRD: u8 = 11;
    const DIST: usize = PLAY_W - BIRD_X + 1;
    const REL: usize = 2 * ROWS - 1;
    let pipe = |t: u8| (PIPE..=PIPE_CAP).contains(&t);
    let Some(i) = (0..H * W).find(|&i| i % W < PLAY_W && grid[i] >= BIRD) else { return };
    let y = i / W;
    obs[y] = 1.0;
    obs[ROWS + (grid[i] - BIRD) as usize] = 1.0;
    let k = ROWS + 5;
    for c in BIRD_X - 1..PLAY_W {
        if (0..ROWS).any(|r| pipe(grid[r * W + c])) {
            obs[k + c - (BIRD_X - 1)] = 1.0;
            let top = (0..ROWS).find(|&r| !pipe(grid[r * W + c])).unwrap_or(0);
            obs[k + DIST + top + GAP / 2 + ROWS - 1 - y] = 1.0;
            break;
        }
    }
    if let Some(m) = (0..H).find(|&r| grid[r * W + PLAY_W] == MARKER) {
        obs[k + DIST + REL + m + ROWS - 1 - y] = 1.0;
    }
}

/// Pac-Man lite: for each direction, is it open, how far the nearest pellet is from the next cell
/// (best way, distance) and how close each ghost is to it; then heading, whose turn it is, how
/// close each ghost is to Pac-Man and the share of pellets left. The frame draws each maze cell as
/// 2x2 tokens; the sprites are read back to maze cells (as `pacman.locate`), shortest paths by BFS.
fn observe_pacman(grid: &[u8], obs: &mut [f32]) {
    const MAP_H: usize = 21;
    const MAP_W: usize = 23;
    const CELLS: usize = MAP_H * MAP_W;
    const WALL: u8 = 1;
    const WALL_FILL: u8 = 2;
    const PELLET: u8 = 3;
    const LIGHT_A: u8 = 10;
    const GHOST_A: u8 = 12;
    const GHOST_B: u8 = 14;
    const PAC: u8 = 16;
    const HUD_ROW: usize = 2 * MAP_H + 1;
    const LIGHT_COL: usize = 2 * MAP_W - 3;
    const PER_DIR: usize = 11;
    const DIRS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    const MOUTH: [(usize, usize); 4] = [(1, 0), (1, 1), (0, 1), (0, 0)];
    const FAR: u8 = 255;

    let tok = |r: usize, c: usize| grid[r * W + c];
    let wall = |i: usize| matches!(tok(2 * (i / MAP_W), 2 * (i % MAP_W)), WALL | WALL_FILL);
    // the one maze token in `lo..hi`, as (row, col); None when missing or doubled
    let unique = |lo: u8, hi: u8| {
        let mut found = None;
        for r in 0..2 * MAP_H {
            for c in 0..2 * MAP_W {
                if (lo..hi).contains(&tok(r, c)) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some((r, c));
                }
            }
        }
        found
    };
    let Some((r, c)) = unique(PAC, PAC + 4) else { return };
    let heading = (tok(r, c) - PAC) as usize;
    let (dx, dy) = MOUTH[heading];
    if r < dy || c < dx || (r - dy) % 2 != 0 || (c - dx) % 2 != 0 {
        return;
    }
    let pac = (r - dy) / 2 * MAP_W + (c - dx) / 2;
    if wall(pac) {
        return;
    }
    let mut pellet = [false; CELLS];
    for (i, p) in pellet.iter_mut().enumerate() {
        *p = tok(2 * (i / MAP_W), 2 * (i % MAP_W)) == PELLET;
    }
    let mut ghosts = [None; 2];
    for (g, t) in ghosts.iter_mut().zip([GHOST_A, GHOST_B]) {
        if let Some((r, c)) = unique(t, t + 2) {
            if r % 2 == 1 && c % 2 == 0 {
                let i = r / 2 * MAP_W + c / 2;
                pellet[i] = tok(r, c) == t + 1;
                *g = (!wall(i)).then_some(i);
            }
        }
    }

    // distance of every maze cell to the nearest source (BFS through the cells that are not walls)
    let bfs = |sources: &dyn Fn(usize) -> bool| -> [u8; CELLS] {
        let mut dist = [FAR; CELLS];
        let mut queue = [0u16; CELLS];
        let (mut head, mut tail) = (0, 0);
        for i in (0..CELLS).filter(|&i| sources(i)) {
            dist[i] = 0;
            queue[tail] = i as u16;
            tail += 1;
        }
        while head < tail {
            let i = queue[head] as usize;
            head += 1;
            let (x, y) = ((i % MAP_W) as i32, (i / MAP_W) as i32);
            for (dx, dy) in DIRS {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= MAP_W as i32 || ny >= MAP_H as i32 {
                    continue;
                }
                let n = ny as usize * MAP_W + nx as usize;
                if !wall(n) && dist[n] == FAR {
                    dist[n] = dist[i] + 1;
                    queue[tail] = n as u16;
                    tail += 1;
                }
            }
        }
        dist
    };
    let left = (0..CELLS).filter(|&i| pellet[i] && !wall(i)).count();
    let pellet_dist = bfs(&|i| pellet[i] && !wall(i));
    let ghost_dist = ghosts.map(|g| g.map(|g| bfs(&|i| i == g)));

    let (px, py) = ((pac % MAP_W) as i32, (pac / MAP_W) as i32);
    let next = |d: usize| {
        let n = (py + DIRS[d].1) as usize * MAP_W + (px + DIRS[d].0) as usize;
        (!wall(n)).then_some(n)
    };
    let dp = |n: usize| if left > 0 { pellet_dist[n] } else { 0 };
    let best = (0..4).filter_map(next).map(dp).min().unwrap_or(0);
    for d in 0..4 {
        let Some(n) = next(d) else { continue };
        let k = d * PER_DIR;
        obs[k] = 1.0;
        obs[k + 1] = (dp(n) == best) as u8 as f32;
        obs[k + 2] = dp(n).min(20) as f32 / 20.0;
        for (i, g) in ghost_dist.iter().enumerate() {
            if let Some(g) = g {
                for (j, limit) in [1u8, 2, 4, 8].into_iter().enumerate() {
                    obs[k + 3 + 4 * i + j] = (g[n] <= limit) as u8 as f32;
                }
            }
        }
    }
    let k = 4 * PER_DIR;
    obs[k + heading] = 1.0;
    obs[k + 4] = (tok(HUD_ROW, LIGHT_COL) == LIGHT_A) as u8 as f32;
    for (i, g) in ghost_dist.iter().enumerate() {
        if let Some(g) = g {
            for (j, limit) in [2u8, 4, 8, 14].into_iter().enumerate() {
                obs[k + 5 + 4 * i + j] = (g[pac] <= limit) as u8 as f32;
            }
        }
    }
    obs[OBS - 1] = left as f32 / 250.0;
}

/// Pong: ball row/col/heading one-hots, your paddle's top row, the ball's row relative to it, and
/// the CPU paddle's top row.
fn observe_pong(grid: &[u8], obs: &mut [f32]) {
    const PLAYER_X: usize = 1;
    const CPU_X: usize = W - 2;
    const PADDLE: u8 = 3;
    const CPU: u8 = 4;
    const BALL: u8 = 10;
    let k = H + W + 6;
    let paddle = (0..H).find(|&r| grid[r * W + PLAYER_X] == PADDLE);
    let cpu = (0..H).find(|&r| grid[r * W + CPU_X] == CPU);
    if let Some(i) = grid.iter().position(|&t| t >= BALL) {
        let (r, c) = (i / W, i % W);
        obs[r] = 1.0;
        obs[H + c] = 1.0;
        obs[H + W + (grid[i] - BALL) as usize] = 1.0;
        if let Some(p) = paddle {
            obs[k + H + r + H - 1 - p] = 1.0;
        }
    }
    if let Some(p) = paddle {
        obs[k + p] = 1.0;
    }
    if let Some(p) = cpu {
        obs[k + H + 2 * H - 1 + p] = 1.0;
    }
}

fn dense<const I: usize, const O: usize>(x: &[f32; I], w: &[f32], b: &[f32]) -> [f32; O] {
    let mut y = [0.0f32; O];
    for (o, (row, &bias)) in y.iter_mut().zip(w.chunks_exact(I).zip(b)) {
        *o = row.iter().zip(x).fold(bias, |acc, (&w, &x)| acc + w * x);
    }
    y
}

/// layer_norm (no affine, eps 1e-5) then leaky_relu(0.01), as in `streamx.Net`.
fn norm_act<const N: usize>(mut x: [f32; N]) -> [f32; N] {
    let mean = x.iter().sum::<f32>() / N as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / N as f32;
    let inv = 1.0 / libm::sqrtf(var + 1e-5);
    for v in x.iter_mut() {
        let y = (*v - mean) * inv;
        *v = if y >= 0.0 { y } else { 0.01 * y };
    }
    x
}

pub fn logits(obs: &[f32; OBS]) -> [f32; ACTIONS] {
    let mut x = [0.0f32; OBS];
    for ((x, &o), (&m, &s)) in x.iter_mut().zip(obs).zip(OBS_MEAN.iter().zip(&OBS_INV_STD)) {
        *x = (o - m) * s;
    }
    let h1 = norm_act(dense::<OBS, HIDDEN>(&x, &L1_W, &L1_B));
    let h2 = norm_act(dense::<HIDDEN, HIDDEN>(&h1, &L2_W, &L2_B));
    dense::<HIDDEN, ACTIONS>(&h2, &OUT_W, &OUT_B)
}

/// Greedy action for a frame.
pub fn act(grid: &[u8; H * W]) -> usize {
    argmax(&logits(&observe(grid)))
}
