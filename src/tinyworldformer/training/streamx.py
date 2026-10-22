"""Stream AC(lambda) (Elsayed, Vasan & Mahmood, 2024): one transition at a time, no replay
buffer, no batches. Trains on the real game or inside the world model's dream.

    python -m tinyworldformer.training.streamx --env real --steps 1000000 --out models/agent_real.pt
    python -m tinyworldformer.training.streamx --env dream --wm models/wm.pt --out models/agent_dream.pt
    python -m tinyworldformer.training.streamx --out models/agent_dream.pt --export tinyworldformer-rs [--name dream]
    python -m tinyworldformer.training.streamx --game doomlite --env real --out models/doom_agent.pt
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyworldformer import games
from tinyworldformer.games import breakout as b, doomlite as dl, flappy as fb, pacman as pm, pong as pg

BALL_ROWS = range(b.H)
REL = 2 * b.W - 1  # ball column minus paddle column, -15..15
BRICKS_AT = b.H + 2 * b.W + 4 + REL
OBS_DIM = BRICKS_AT + len(b.BRICK_ROWS) * (b.W - 2)  # 143


def observe(grid: np.ndarray) -> np.ndarray:
    """Breakout frame -> features: ball row/col/heading one-hots, paddle col one-hot,
    ball-minus-paddle column one-hot, brick mask."""
    obs = np.zeros(OBS_DIM, np.float32)
    balls = np.argwhere(grid >= b.BALL_UL)
    paddle = np.flatnonzero(grid[b.PADDLE_ROW] == b.PADDLE)
    pc = int(round(paddle.mean())) if len(paddle) else None
    if len(balls):
        r, c = balls[0]
        obs[r] = obs[b.H + c] = obs[b.H + b.W + grid[r, c] - b.BALL_UL] = 1
        if pc is not None:
            obs[b.H + 2 * b.W + 4 + c - pc + b.W - 1] = 1
    if pc is not None:
        obs[b.H + b.W + 4 + pc] = 1
    bricks = grid[b.BRICK_ROWS.start : b.BRICK_ROWS.stop, 1 : b.W - 1] == b.BRICK
    obs[BRICKS_AT:] = bricks.reshape(-1)
    return obs


N = dl.MAP_N
DOOM_OBS = N * N + 4 + N * N + 2 * (2 * N - 1) + (dl.HEALTH + 1) + 1  # 169


def observe_doom(grid: np.ndarray) -> np.ndarray:
    """Doom-lite frame -> features, all read off the frame: player cell and facing and enemy cell
    from the minimap, the enemy's offset ahead/right of the player, health from the HUD bar, and
    whether enemy pixels sit in the crosshair (the centre columns of the view)."""
    obs = np.zeros(DOOM_OBS, np.float32)
    mini = grid[dl.VIEW_H :, dl.W - N :]
    player = np.argwhere(mini >= dl.MAP_PLAYER)
    enemy = np.argwhere(mini == dl.MAP_ENEMY)
    k = 0
    if len(player):
        py, px = player[0]
        facing = mini[py, px] - dl.MAP_PLAYER
        obs[py * N + px] = obs[N * N + facing] = 1
    k = N * N + 4
    if len(enemy):
        ey, ex = enemy[0]
        obs[k + ey * N + ex] = 1
        if len(player):
            dx, dy = dl.DIRS[facing]
            ahead = (ex - px) * dx + (ey - py) * dy
            right = (ex - px) * -dy + (ey - py) * dx
            obs[k + N * N + ahead + N - 1] = 1
            obs[k + N * N + 2 * N - 1 + right + N - 1] = 1
    k += N * N + 2 * (2 * N - 1)
    health = int((grid[27] == dl.HEALTH_BAR).sum()) // 3
    obs[k + min(health, dl.HEALTH)] = 1
    obs[-1] = float(np.isin(grid[: dl.VIEW_H, 15:17], (dl.ENEMY, dl.ENEMY_EYE)).any())
    return obs


FB_ROWS = fb.GROUND  # rows the bird can be in
FB_REL = 2 * FB_ROWS - 1  # a target row minus the bird row
FB_DIST = fb.PLAY_W - fb.BIRD_X + 1  # next pipe column minus the bird column, -1..16
FLAPPY_OBS = FB_ROWS + 5 + FB_DIST + 2 * FB_REL  # 131
_PIPE_TOKENS = (fb.PIPE, fb.PIPE_EDGE, fb.PIPE_CAP)


def observe_flappy(grid: np.ndarray) -> np.ndarray:
    """Flappy Bird frame -> features: bird row and speed, distance to the next pipe, that pipe's
    gap centre and the next-gap marker, both relative to the bird."""
    obs = np.zeros(FLAPPY_OBS, np.float32)
    bird = np.argwhere(grid[:, : fb.PLAY_W] >= fb.BIRD)
    if not len(bird):
        return obs
    y, x = bird[0]
    obs[y] = obs[FB_ROWS + grid[y, x] - fb.BIRD] = 1
    k = FB_ROWS + 5
    for c in range(fb.BIRD_X - 1, fb.PLAY_W):
        col = grid[:FB_ROWS, c]
        if np.isin(col, _PIPE_TOKENS).any():
            obs[k + c - (fb.BIRD_X - 1)] = 1
            top = int(np.argmax(~np.isin(col, _PIPE_TOKENS)))  # first open row of the gap
            obs[k + FB_DIST + top + fb.GAP // 2 - y + FB_ROWS - 1] = 1
            break
    marker = np.flatnonzero(grid[:, fb.PLAY_W] == fb.MARKER)
    if len(marker):
        obs[k + FB_DIST + FB_REL + marker[0] - y + FB_ROWS - 1] = 1
    return obs


PM_PER_DIR = 11
PACMAN_OBS = 4 * PM_PER_DIR + 4 + 1 + 8 + 1  # 58
_PM_XS, _PM_YS = np.array([c[0] for c in pm.FREE]), np.array([c[1] for c in pm.FREE])


def observe_pacman(grid: np.ndarray) -> np.ndarray:
    """Pac-Man lite frame -> features, egocentric and read off the frame: for each direction, is it
    open, how far is the nearest pellet from the next cell (is it the best way, and the distance),
    and how close each ghost is to that cell (within 1, 2, 4, 8 steps); then the heading, whether
    ghost A moves next, how close each ghost is to Pac-Man (within 2, 4, 8, 14), and the share of
    pellets left. Distances are shortest paths through the maze."""
    obs = np.zeros(PACMAN_OBS, np.float32)
    pac, ga, gb, pellet = pm.locate(grid)
    if pac is None or pac[0] not in pm.INDEX:
        return obs
    (px, py), heading = pac
    pellets = pellet[_PM_YS, _PM_XS]
    left = int(pellets.sum())
    ghosts = [pm.INDEX.get(g) if g is not None else None for g in (ga, gb)]
    dp = [None] * 4
    for d, (dx, dy) in enumerate(pm.DIRS):
        n = (px + dx, py + dy)
        if n in pm.INDEX:
            dp[d] = int(pm.DIST[pm.INDEX[n]][pellets].min()) if left else 0
    best = min(v for v in dp if v is not None)
    for d, (dx, dy) in enumerate(pm.DIRS):
        if dp[d] is None:
            continue
        k = d * PM_PER_DIR
        obs[k] = 1
        obs[k + 1] = dp[d] == best
        obs[k + 2] = np.float32(min(dp[d], 20)) / np.float32(20)
        n = pm.INDEX[(px + dx, py + dy)]
        for i, g in enumerate(ghosts):
            if g is not None:
                gd = int(pm.DIST[g][n])
                obs[k + 3 + 4 * i : k + 7 + 4 * i] = [gd <= 1, gd <= 2, gd <= 4, gd <= 8]
    k = 4 * PM_PER_DIR
    obs[k + heading] = 1
    obs[k + 4] = grid[pm.HUD_ROW, pm.LIGHT_COL] == pm.LIGHT_A
    for i, g in enumerate(ghosts):
        if g is not None:
            gd = int(pm.DIST[g][pm.INDEX[(px, py)]])
            obs[k + 5 + 4 * i : k + 9 + 4 * i] = [gd <= 2, gd <= 4, gd <= 8, gd <= 14]
    obs[-1] = np.float32(left) / np.float32(250)
    return obs


PONG_REL = 2 * pg.H - 1  # ball row minus your paddle's top row
PONG_OBS = pg.H + pg.W + 6 + pg.H + PONG_REL + pg.H  # 137


def observe_pong(grid: np.ndarray) -> np.ndarray:
    """Pong frame -> features: ball row/col/heading one-hots, your paddle's top row, the ball's row
    relative to it, and the CPU paddle's top row."""
    obs = np.zeros(PONG_OBS, np.float32)
    k = pg.H + pg.W + 6
    paddle = np.flatnonzero(grid[:, pg.PLAYER_X] == pg.PADDLE)
    cpu = np.flatnonzero(grid[:, pg.CPU_X] == pg.CPU)
    balls = np.argwhere(grid >= pg.BALL)
    if len(balls):
        r, c = balls[0]
        obs[r] = obs[pg.H + c] = obs[pg.H + pg.W + grid[r, c] - pg.BALL] = 1
        if len(paddle):
            obs[k + pg.H + r - paddle[0] + pg.H - 1] = 1
    if len(paddle):
        obs[k + paddle[0]] = 1
    if len(cpu):
        obs[k + pg.H + PONG_REL + cpu[0]] = 1
    return obs


def features(game: str):
    """(observe function, feature count) for a game."""
    return {"breakout": (observe, OBS_DIM), "doomlite": (observe_doom, DOOM_OBS),
            "flappy": (observe_flappy, FLAPPY_OBS), "pacman": (observe_pacman, PACMAN_OBS),
            "pong": (observe_pong, PONG_OBS)}[game]


# --- environments -----------------------------------------------------------------------------


class RealEnv:
    def __init__(self, seed: int, game: str = "breakout"):
        self.rng, self.g = np.random.default_rng(seed), games.get(game)

    def reset(self) -> np.ndarray:
        self.state = self.g.module.reset(self.rng)
        return self.state.grid

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.state, reward, done = self.g.module.step(self.state, action, self.rng)
        return self.state.grid, reward, done


class DreamEnv:
    """The world model as the environment; mirrors tinyworldformer-rs/src/dream.rs."""

    def __init__(self, wm_path: Path, seed: int, loops: int | None = None, device: str = "cpu"):
        from tinyworldformer.training.export import load
        from tinyworldformer.training.model import ActQuant, decode

        self.model = load(wm_path).to(device)
        self.g = self.model.g
        ActQuant.enabled = True  # dream with the quantized model, like the Pico does
        self.decode, self.device = decode, device
        self.loops = loops or self.model.c.loops
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device).manual_seed(seed)

    def reset(self) -> np.ndarray:
        self.grid = self.g.module.reset(self.rng).grid
        self.lives, self.steps = self.g.lives, 0
        return self.grid

    @torch.no_grad()
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        g = torch.from_numpy(self.grid)[None].to(self.device)
        a = torch.tensor([action], device=self.device)
        cells, reward = self.model(g, a, self.loops)[-1]
        grid = self.decode(cells, self.g, self.gen)[0].cpu().numpy()
        r = self.g.rewards[int(reward.argmax(-1).item())]
        self.lives -= r < 0
        self.steps += 1
        self.grid = grid
        gone = self.g.done_when_absent >= 0 and not (grid == self.g.done_when_absent).any()
        return grid, r, bool(self.lives <= 0 or gone or self.steps >= self.g.max_steps)


# --- stream-x pieces --------------------------------------------------------------------------


def sparse_init(layer: nn.Linear, sparsity: float = 0.9) -> None:
    fan_in = layer.weight.shape[1]
    with torch.no_grad():
        layer.weight.uniform_(-math.sqrt(1 / fan_in), math.sqrt(1 / fan_in))
        for row in layer.weight:
            row[torch.randperm(fan_in)[: math.ceil(sparsity * fan_in)]] = 0
        layer.bias.zero_()


class Net(nn.Module):
    def __init__(self, inp: int, out: int, hidden: int = 64):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(inp, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, out)
        for layer in (self.l1, self.l2, self.out):
            sparse_init(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(F.layer_norm(self.l1(x), (self.l1.out_features,)))
        x = F.leaky_relu(F.layer_norm(self.l2(x), (self.l2.out_features,)))
        return self.out(x)


class ObGD:
    """Overshooting-bounded gradient descent with eligibility traces.

    The trace holds gradients of the quantity being *increased* (V, log pi); the update is
    w += step * delta * z, with the step shrunk so one update cannot overshoot the TD target.
    """

    def __init__(self, params, lr: float = 1.0, gamma: float = 0.99, lam: float = 0.8, kappa: float = 2.0):
        self.params = list(params)
        self.lr, self.decay, self.kappa = lr, gamma * lam, kappa
        self.z = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, delta: float, reset: bool) -> None:
        z_sum = 0.0
        for p, z in zip(self.params, self.z):
            z.mul_(self.decay).add_(p.grad)
            z_sum += z.abs().sum().item()
        bound = self.lr * self.kappa * max(abs(delta), 1.0) * z_sum
        step = self.lr / bound if bound > 1 else self.lr
        for p, z in zip(self.params, self.z):
            p.add_(z, alpha=step * delta)
            if reset:
                z.zero_()


class RunningNorm:
    """Welford running mean/variance."""

    def __init__(self, shape=()):
        self.n, self.mean, self.m2 = 0, np.zeros(shape), np.zeros(shape)

    def update(self, x) -> None:
        self.n += 1
        d = x - self.mean
        self.mean = self.mean + d / self.n
        self.m2 = self.m2 + d * (x - self.mean)

    @property
    def var(self):
        return self.m2 / self.n if self.n > 1 else np.ones_like(self.mean)


class StreamAC:
    def __init__(self, obs_dim: int, n_actions: int, gamma=0.99, lam=0.8, entropy=0.01, hidden=64):
        self.pi, self.v = Net(obs_dim, n_actions, hidden), Net(obs_dim, 1, hidden)
        self.opt_pi = ObGD(self.pi.parameters(), gamma=gamma, lam=lam, kappa=3.0)
        self.opt_v = ObGD(self.v.parameters(), gamma=gamma, lam=lam, kappa=2.0)
        self.gamma, self.entropy = gamma, entropy
        self.obs_norm, self.ret_norm, self.ret = RunningNorm(obs_dim), RunningNorm(), 0.0

    def normalize(self, obs: np.ndarray, update: bool = True) -> torch.Tensor:
        if update:
            self.obs_norm.update(obs)
        x = (obs - self.obs_norm.mean) / np.sqrt(self.obs_norm.var + 1e-8)
        return torch.as_tensor(x, dtype=torch.float32)

    def scale_reward(self, r: float, done: bool) -> float:
        self.ret = self.gamma * self.ret * (1 - done) + r
        self.ret_norm.update(self.ret)
        return r / math.sqrt(float(self.ret_norm.var) + 1e-8)

    @torch.no_grad()
    def act(self, x: torch.Tensor, greedy: bool = False) -> int:
        logits = self.pi(x)
        return int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())

    def update(self, x, a: int, r: float, x2, terminal: bool, reset: bool) -> None:
        with torch.no_grad():
            target = r + (0.0 if terminal else self.gamma * self.v(x2).item())
        v = self.v(x)
        delta = target - v.item()
        dist = torch.distributions.Categorical(logits=self.pi(x))
        objective = dist.log_prob(torch.tensor(a)) + self.entropy * math.copysign(1, delta) * dist.entropy()
        self.pi.zero_grad(), self.v.zero_grad()
        v.sum().backward()
        objective.backward()
        self.opt_v.step(delta, reset)
        self.opt_pi.step(delta, reset)


# --- training ---------------------------------------------------------------------------------


def player_cell(grid: np.ndarray) -> tuple[int, int] | None:
    """Doom-lite: the player's cell, read off the minimap."""
    p = np.argwhere(grid[dl.VIEW_H :, dl.W - dl.MAP_N :] >= dl.MAP_PLAYER)
    return (int(p[0][1]), int(p[0][0])) if len(p) else None


class IdlePenalty:
    """Reward shaping against camping: a penalty on every step where, within the last `window`
    steps, the player spent more than `max_same` of them in its current cell. Stepping back and
    forth between two cells does not escape it; patrolling a few cells does."""

    def __init__(self, penalty: float, window: int = 20, max_same: int = 6):
        self.penalty, self.max_same = penalty, max_same
        self.cells: deque = deque(maxlen=window)

    def reset(self) -> None:
        self.cells.clear()

    def __call__(self, grid: np.ndarray) -> float:
        cell = player_cell(grid)
        self.cells.append(cell)
        return self.penalty if self.cells.count(cell) > self.max_same else 0.0


def evaluate(agent: StreamAC, episodes: int, seed: int, max_steps: int, game: str = "breakout") -> tuple[float, float, float]:
    """Greedy play in the *real* game: (mean return, mean positive reward per episode: bricks
    broken in Breakout, kills in Doom-lite, mean distinct cells visited per episode in Doom-lite)."""
    observe_fn, _ = features(game)
    env, rets, wins, roam = RealEnv(seed, game), [], [], []
    for _ in range(episodes):
        grid, ret, win = env.reset(), 0.0, 0.0
        cells = set()
        for _ in range(max_steps):
            grid, r, done = env.step(agent.act(agent.normalize(observe_fn(grid), update=False), greedy=True))
            ret, win = ret + r, win + max(r, 0)
            if game == "doomlite":
                cells.add(player_cell(grid))
            if done:
                break
        rets.append(ret)
        wins.append(win)
        roam.append(len(cells))
    return float(np.mean(rets)), float(np.mean(wins)), float(np.mean(roam))


def save(agent: StreamAC, args, curve: list, path: Path | None = None) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pi": agent.pi.state_dict(), "v": agent.v.state_dict(), "hidden": args.hidden,
                "obs_mean": agent.obs_norm.mean, "obs_var": agent.obs_norm.var,
                "curve": curve, "env": args.env, "game": args.game}, path or args.out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("real", "dream"), default="real")
    p.add_argument("--game", choices=list(games.GAMES), default="breakout", help="for --env real (a dream knows its game)")
    p.add_argument("--wm", type=Path, default=Path("models/wm.pt"))
    p.add_argument("--loops", type=int, default=None, help="world-model loops in the dream")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--max-ep-steps", type=int, default=3000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--miss-penalty", type=float, default=1.0, help="negative rewards are scaled by this (shaping)")
    p.add_argument("--idle-penalty", type=float, default=0.0,
                   help="Doom-lite shaping: penalty per step spent camping (see IdlePenalty)")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("models/agent.pt"))
    p.add_argument("--snapshots", type=lambda s: {int(float(x)) for x in s.split(",")}, default=set(),
                   help="also keep the policy at these steps, e.g. 1e4,5e4: <out>_<k>k.pt")
    p.add_argument("--export", type=Path, default=None, help="export --out checkpoint to this Rust crate")
    p.add_argument("--name", default=None, help="agent name in the crate (default: the checkpoint's file name)")
    args = p.parse_args()
    if args.export:
        game = games.get(torch.load(args.out, weights_only=False).get("game", "breakout")).name
        out = args.export / "models" / "agents" / game / (args.name or args.out.stem)
        export_learner_golden(out, game)
        return export_rust(args.out, out)

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    env = RealEnv(args.seed, args.game) if args.env == "real" else DreamEnv(args.wm, args.seed, args.loops)
    args.game = env.g.name
    observe, obs_dim = features(args.game)
    agent = StreamAC(obs_dim, env.g.A, lam=args.lam, hidden=args.hidden)
    assert args.idle_penalty == 0 or args.game == "doomlite", "--idle-penalty reads the Doom-lite minimap"
    idle = IdlePenalty(args.idle_penalty)
    grid, ep_ret, ep_len, returns, t0 = env.reset(), 0.0, 0, [], time.time()
    x = agent.normalize(observe(grid))
    curve = []
    if 0 in args.snapshots:  # the untrained policy
        save(agent, args, curve, args.out.with_name(f"{args.out.stem}_0k.pt"))
    for step in range(1, args.steps + 1):
        a = agent.act(x)
        grid, r, done = env.step(a)
        ep_ret, ep_len = ep_ret + r, ep_len + 1
        truncated = ep_len >= args.max_ep_steps
        x2 = agent.normalize(observe(grid))
        shaped = (r * args.miss_penalty if r < 0 else r) - (idle(grid) if args.idle_penalty else 0.0)
        agent.update(x, a, agent.scale_reward(shaped, done), x2, terminal=done, reset=done or truncated)
        x = x2
        if done or truncated:
            returns.append(ep_ret)
            grid, ep_ret, ep_len = env.reset(), 0.0, 0
            idle.reset()
            x = agent.normalize(observe(grid))
        if step in args.snapshots:
            save(agent, args, curve, args.out.with_name(f"{args.out.stem}_{step / 1000:g}k.pt"))
        if step % args.eval_every == 0:
            ret, wins, roam = evaluate(agent, 5, 1000 + step, args.max_ep_steps, args.game)
            train_ret = float(np.mean(returns[-20:])) if returns else float("nan")
            curve.append((step, train_ret, ret, wins))
            what = {"breakout": "bricks", "doomlite": "kills", "flappy": "pipes", "pacman": "pellets", "pong": "points"}[args.game]
            moved = f", cells visited {roam:.1f}" if args.game == "doomlite" else ""
            print(f"step {step} {time.time() - t0:.0f}s | train return (last 20 ep) {train_ret:.1f} | "
                  f"real-game greedy return {ret:.1f}, {what} {wins:.1f}{moved}", flush=True)
            save(agent, args, curve)  # keep the latest policy around (for inspection, data collection)

    save(agent, args, curve)
    print(f"saved {args.out}")


# --- export for the Pico ------------------------------------------------------------------------


def export_rust(ckpt: Path, out: Path, n_golden: int = 64) -> None:
    """Write one agent of the crate: `out/agent_weights.rs` and `out/golden_agent.bin` (grid u8[H*W],
    obs f32[OBS], logits f32[A]). `out` is `models/agents/<game>/<name>`; builds pick it with
    `TWF_AGENT=<name>`."""
    from tinyworldformer.rustexport import Golden, rust_array, rust_const, write_module
    from tinyworldformer.rustexport.convert import mlp_arrays

    ck = torch.load(ckpt, weights_only=False)
    game = ck.get("game", "breakout")
    observe, obs_dim = features(game)
    n_actions = games.get(game).A
    pi = Net(obs_dim, n_actions, ck["hidden"])
    pi.load_state_dict(ck["pi"])
    mean = ck["obs_mean"].astype(np.float32)
    inv_std = (1 / np.sqrt(ck["obs_var"] + 1e-8)).astype(np.float32)
    parts = [rust_const("OBS", obs_dim), rust_const("HIDDEN", ck["hidden"]),
             rust_array("OBS_MEAN", mean, "f32"), rust_array("OBS_INV_STD", inv_std, "f32"),
             *mlp_arrays([("L1", pi.l1), ("L2", pi.l2), ("OUT", pi.out)])]
    write_module(out / "agent_weights.rs", parts, "python -m tinyworldformer.training.streamx --export")

    env, golden = RealEnv(0, game), Golden()
    grid = env.reset()
    for i in range(n_golden):
        obs = observe(grid)
        x = torch.from_numpy((obs - mean) * inv_std)
        with torch.no_grad():
            logits = pi(x).numpy().astype(np.float32)
        golden.array(grid, np.uint8).array(obs).array(logits)
        grid, _, done = env.step(int(logits.argmax()) if i % 3 else int(env.rng.integers(n_actions)))
        if done:
            grid = env.reset()
    golden.save(out / "golden_agent.bin")
    print(f"wrote agent {out.name} for {game}: {out}/agent_weights.rs (real-game return curve: {ck['curve'][-1:]})")


def export_learner_golden(out: Path, game: str, steps: int = 200, hidden: int = 32, seed: int = 0) -> None:
    """Reference run for the Rust learner (`out/golden_learner.bin`): initial weights, a fixed
    sequence of real-game transitions (random actions), the final weights. Arrays are f32 in the
    Rust layout: per net w1 (H, OBS), b1, w2 (H, H), b2, w3 (O, H), b3."""
    import struct

    torch.manual_seed(seed)
    observe_fn, obs_dim = features(game)
    n_actions = games.get(game).A
    agent = StreamAC(obs_dim, n_actions, hidden=hidden)
    rng = np.random.default_rng(seed)

    def weights() -> bytes:
        return b"".join(
            t.detach().numpy().astype(np.float32).tobytes()
            for net in (agent.pi, agent.v) for layer in (net.l1, net.l2, net.out) for t in (layer.weight, layer.bias))

    env = RealEnv(seed, game)
    obs = observe_fn(env.reset())
    buf = [struct.pack("<I", steps), weights(), obs.tobytes()]
    x = agent.normalize(obs)
    for _ in range(steps):
        a = int(rng.integers(n_actions))
        grid, r, done = env.step(a)
        obs2 = observe_fn(grid)
        x2 = agent.normalize(obs2)
        agent.update(x, a, agent.scale_reward(r, done), x2, terminal=done, reset=done)
        buf += [struct.pack("<BfB", a, r, done), obs2.tobytes()]
        x = x2
        if done:
            obs = observe_fn(env.reset())
            buf.append(obs.tobytes())
            x = agent.normalize(obs)
    buf.append(weights())
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden_learner.bin").write_bytes(b"".join(buf))
    print(f"wrote {out / 'golden_learner.bin'} ({game}, {steps} steps, hidden {hidden})")


if __name__ == "__main__":
    main()
