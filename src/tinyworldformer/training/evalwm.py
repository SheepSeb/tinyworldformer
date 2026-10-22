"""Where does the world model go wrong, and how long do its dreams stay faithful?

    python -m tinyworldformer.training.evalwm --ckpt models/wm.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tinyworldformer.games import breakout as b
from tinyworldformer.training.dataset import Transitions
from tinyworldformer.training.export import load
from tinyworldformer.training.model import ActQuant, decode


@torch.no_grad()
def predict(model, grid: np.ndarray, action: np.ndarray, loops: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    cells, reward = model(torch.from_numpy(grid).to(device), torch.from_numpy(action).to(device), loops)[-1]
    return decode(cells, model.g).cpu().numpy(), (reward.argmax(-1) - 1).cpu().numpy()


def error_breakdown(model, d: dict, loops: int, device: str, n: int = 20000) -> None:
    grid, action, nxt = d["grid"][:n], d["action"][:n], d["next_grid"][:n]
    pred = np.concatenate([predict(model, grid[i : i + 2048], action[i : i + 2048], loops, device)[0]
                           for i in range(0, len(grid), 2048)])
    wrong = pred != nxt
    bad = wrong.any((1, 2))
    serve = d["reward"][:n] < 0
    ball = lambda g: g >= b.BALL_UL  # noqa: E731
    only_heading = bad & ~((ball(pred) != ball(nxt)) | (wrong & ~ball(nxt) & ~ball(pred))).any((1, 2))
    cats = {
        "serve frames (random heading, unavoidable ~50%)": bad & serve,
        "ball heading only (non-serve)": bad & ~serve & only_heading,
        "ball position": bad & ~serve & (ball(pred) != ball(nxt)).any((1, 2)),
        "brick cells": bad & ~serve & (wrong & ((pred == b.BRICK) | (nxt == b.BRICK))).any((1, 2)),
        "paddle cells": bad & ~serve & (wrong & ((pred == b.PADDLE) | (nxt == b.PADDLE))).any((1, 2)),
    }
    print(f"loops {loops}: wrong frames {bad.mean():.4f} of {n} (serve frames are {serve.mean():.4f} of all)")
    for k, v in cats.items():
        print(f"  {k:48s} {v.mean():.4f}")
    print(f"  mean wrong cells per wrong frame {wrong.sum((1, 2))[bad].mean():.2f}")


def rollouts(model, d: dict, loops: int, device: str, episodes: int = 274) -> None:
    """Feed the model its own frames with the recorded actions, all episodes in one batch, and
    count the steps that stay identical to the real game. Random events cannot be matched: on a
    Breakout serve the dream only has to put the ball in the right cell, on a Doom-lite respawn
    nothing is checked; either way the dream then adopts the real frame."""
    g = model.g
    starts = np.flatnonzero(d["t"] == 0)[:episodes]
    ends = np.array([s + np.flatnonzero(d["episode"][s:] != d["episode"][s])[:1].sum() or len(d["t"]) for s in starts])
    ends = np.where(ends == starts, len(d["t"]), ends)
    grid = d["grid"][starts].copy()
    alive = np.ones(len(starts), bool)
    faithful = np.zeros(len(starts), int)
    for k in range(int((ends - starts).max())):
        idx = starts + k
        alive &= idx < ends
        if not alive.any():
            break
        live = np.flatnonzero(alive)
        pred, _ = predict(model, grid[live], d["action"][idx[live]], loops, device)
        real = d["next_grid"][idx[live]]
        serve = d["random"][idx[live]]
        same = (pred == real).all((1, 2))
        if g.name == "breakout":
            ball_cell = ((pred >= b.BALL_UL) == (real >= b.BALL_UL)) | ((pred < b.BALL_UL) & (real < b.BALL_UL))
            same |= serve & ((pred == real) | ((pred >= b.BALL_UL) & (real >= b.BALL_UL))).all((1, 2)) & ball_cell.all((1, 2))
        else:
            same |= serve
        faithful[live[same]] += 1
        alive[live[~same]] = False
        grid[live] = real  # on survivors pred == real up to the serve heading
    full = (faithful == ends - starts).sum()
    print(f"loops {loops}: steps identical to the real game: median {np.median(faithful):.0f}, "
          f"mean {faithful.mean():.0f}, max {faithful.max()}; whole episode exact in {full}/{len(starts)} "
          f"(mean episode length {np.mean(ends - starts):.0f})")


def glitches(wm: Path, agent: Path, loops: int, steps: int = 5000, seed: int = 0) -> None:
    """Let a trained agent play in the dream and count the ways the dream breaks the rules."""
    from tinyworldformer.training.streamx import OBS_DIM, DreamEnv, Net, observe

    ck = torch.load(agent, weights_only=False)
    pi = Net(OBS_DIM, b.NUM_ACTIONS, ck["hidden"])
    pi.load_state_dict(ck["pi"])
    mean, inv = ck["obs_mean"], 1 / np.sqrt(ck["obs_var"] + 1e-8)
    env = DreamEnv(wm, seed, loops=loops)
    interior = np.zeros((b.H, b.W), bool)
    interior[1:, 1 : b.W - 1] = True
    grid, counts, score = env.reset(), dict(reward_no_break=0, break_no_reward=0, regrow=0, bad_tokens=0), 0.0
    for _ in range(steps):
        x = torch.as_tensor((observe(grid) - mean) * inv, dtype=torch.float32)
        prev = grid
        with torch.no_grad():
            grid, r, done = env.step(int(pi(x).argmax()))
        score += r
        broke = (prev == b.BRICK).sum() - (grid == b.BRICK).sum()
        counts["reward_no_break"] += r > 0 and broke <= 0
        counts["break_no_reward"] += broke > 0 and r <= 0
        counts["regrow"] += broke < 0
        bad = (grid == b.WALL) & interior
        bad |= (grid == b.PADDLE) & (np.arange(b.H) != b.PADDLE_ROW)[:, None]
        counts["bad_tokens"] += bad.any()
        if done:
            grid = env.reset()
    print(f"{wm.name} ({loops} loops), agent {agent.name}, {steps} steps: dream score {score:.0f}, " +
          ", ".join(f"{k} {v}" for k, v in counts.items()))


def doom_check(wm: Path, agent: Path, loops: int, steps: int = 5000, seed: int = 0) -> None:
    """Doom-lite: let an agent play in the dream, read the game state back out of every dreamed
    frame (minimap, health bar, turn light) and compare each dreamed step with the real rules
    applied to that same state. Random respawns are only checked for the reward."""
    from tinyworldformer.games import doomlite as dl
    from tinyworldformer.training.streamx import DreamEnv, Net, features

    observe, obs_dim = features("doomlite")
    ck = torch.load(agent, weights_only=False)
    pi = Net(obs_dim, dl.NUM_ACTIONS, ck["hidden"])
    pi.load_state_dict(ck["pi"])
    mean, inv = ck["obs_mean"], 1 / np.sqrt(ck["obs_var"] + 1e-8)

    counts = dict(steps=0, unreadable=0, in_wall=0, state_wrong=0, reward_wrong=0, pixels_off=0)
    def state(grid: np.ndarray) -> dl.State | None:
        s = dl.read(grid)
        mini = grid[dl.VIEW_H :, dl.W - dl.MAP_N :]
        cells = np.argwhere((mini >= dl.MAP_PLAYER) | (mini == dl.MAP_ENEMY))
        if s is None and len(cells) == 2 and any(dl.WALLS[y, x] for y, x in cells):
            counts["in_wall"] += 1  # the dream put someone inside a wall: no real state matches
        return s

    env = DreamEnv(wm, seed, loops=loops)
    grid = env.reset()
    score = 0.0
    for _ in range(steps):
        x = torch.as_tensor((observe(grid) - mean) * inv, dtype=torch.float32)
        with torch.no_grad():
            a = int(pi(x).argmax())
        before = state(grid)
        grid, r, done = env.step(a)
        score += r
        counts["steps"] += 1
        after = state(grid)
        if before is None or after is None:
            counts["unreadable"] += 1
        else:
            real, rr, _ = dl.step(before, a, np.random.default_rng(0))
            counts["reward_wrong"] += rr != r
            if rr <= 0:  # no respawn: the whole next state is determined
                key = lambda s: (s.px, s.py, s.facing, s.ex, s.ey, s.health, s.steps % 2)  # noqa: E731
                counts["state_wrong"] += key(real) != key(after)
                counts["pixels_off"] += int((real.grid != grid).sum())
        if done:
            grid = env.reset()
    n = counts["steps"]
    print(f"{wm.name} ({loops} loops), agent {agent.name}: dream score {score:.0f} over {n} steps | "
          f"unreadable frames {counts['unreadable']} (inside a wall {counts['in_wall']}), wrong state {counts['state_wrong']}, "
          f"wrong reward {counts['reward_wrong']}, wrong pixels per step {counts['pixels_off'] / n:.2f}")


def flappy_check(wm: Path, agent: Path, loops: int, steps: int = 5000, seed: int = 0) -> None:
    """Flappy Bird: let an agent play in the dream, read the game state back out of every dreamed
    frame and compare each dreamed step with the real rules applied to that state. On steps where
    a pipe spawns the marker's new height is random, so only the rest is compared."""
    from tinyworldformer.games import flappy as fb
    from tinyworldformer.training.streamx import DreamEnv, Net, features

    observe, obs_dim = features("flappy")
    ck = torch.load(agent, weights_only=False)
    pi = Net(obs_dim, fb.NUM_ACTIONS, ck["hidden"])
    pi.load_state_dict(ck["pi"])
    mean, inv = ck["obs_mean"], 1 / np.sqrt(ck["obs_var"] + 1e-8)
    state = fb.read

    env = DreamEnv(wm, seed, loops=loops)
    grid = env.reset()
    counts = dict(steps=0, unreadable=0, state_wrong=0, reward_wrong=0, pixels_off=0)
    score = 0.0
    key = lambda s: (s.y, s.v, s.pipes, s.phase)  # noqa: E731
    for _ in range(steps):
        x = torch.as_tensor((observe(grid) - mean) * inv, dtype=torch.float32)
        with torch.no_grad():
            a = int(pi(x).argmax())
        before = state(grid)
        grid, r, done = env.step(a)
        score += r
        counts["steps"] += 1
        after = state(grid)
        if before is None or after is None:
            counts["unreadable"] += 1
        else:
            real, rr, _ = fb.step(before, a, np.random.default_rng(0))
            counts["reward_wrong"] += rr != r
            if not real.random_event:
                counts["state_wrong"] += key(real) != key(after) or real.next_gap != after.next_gap
                counts["pixels_off"] += int((real.grid != grid).sum())
            else:
                counts["state_wrong"] += key(real) != key(after)
        if done:
            grid = env.reset()
    n = counts["steps"]
    print(f"{wm.name} ({loops} loops), agent {agent.name}: dream score {score:.0f} over {n} steps | "
          f"unreadable frames {counts['unreadable']}, wrong state {counts['state_wrong']}, "
          f"wrong reward {counts['reward_wrong']}, wrong pixels per step {counts['pixels_off'] / n:.2f}")


def pacman_check(wm: Path, agent: Path, loops: int, steps: int = 5000, seed: int = 0) -> None:
    """Pac-Man lite: let an agent play in the dream, read the game state back out of every dreamed
    frame and compare each dreamed step with the real rules applied to that state. After a catch the
    ghosts respawn at random, so only Pac-Man, the pellets and the lives are compared there."""
    from tinyworldformer.games import pacman as pm
    from tinyworldformer.training.streamx import DreamEnv, Net, features

    observe, obs_dim = features("pacman")
    ck = torch.load(agent, weights_only=False)
    pi = Net(obs_dim, pm.NUM_ACTIONS, ck["hidden"])
    pi.load_state_dict(ck["pi"])
    mean, inv = ck["obs_mean"], 1 / np.sqrt(ck["obs_var"] + 1e-8)

    counts = dict(steps=0, unreadable=0, state_wrong=0, reward_wrong=0, walls_wrong=0, pixels_off=0)
    key = lambda s: (s.pellets.tobytes(), s.pac, s.heading, s.lives, s.steps % 2)  # noqa: E731
    env = DreamEnv(wm, seed, loops=loops)
    grid = env.reset()
    score = 0.0
    for _ in range(steps):
        x = torch.as_tensor((observe(grid) - mean) * inv, dtype=torch.float32)
        with torch.no_grad():
            a = int(pi(x).argmax())
        before = pm.read(grid)
        grid, r, done = env.step(a)
        score += r
        counts["steps"] += 1
        after = pm.read(grid)
        maze = grid[: pm.S * pm.MAP_H, : pm.S * pm.MAP_W]
        counts["walls_wrong"] += int((np.isin(maze, (pm.WALL, pm.WALL_FILL)) != np.isin(pm.BACKGROUND[: pm.S * pm.MAP_H, : pm.S * pm.MAP_W], (pm.WALL, pm.WALL_FILL))).any())
        if before is None or after is None:
            counts["unreadable"] += 1
        else:
            real, rr, _ = pm.step(before, a, np.random.default_rng(0))
            counts["reward_wrong"] += rr != r
            if real.random_event:
                counts["state_wrong"] += key(real) != key(after)
            else:
                counts["state_wrong"] += key(real) != key(after) or (real.a, real.b) != (after.a, after.b)
                counts["pixels_off"] += int((real.grid != grid).sum())
        if done:
            grid = env.reset()
    n = counts["steps"]
    print(f"{wm.name} ({loops} loops), agent {agent.name}: dream score {score:.0f} over {n} steps | "
          f"unreadable frames {counts['unreadable']}, wrong maze {counts['walls_wrong']}, wrong state {counts['state_wrong']}, "
          f"wrong reward {counts['reward_wrong']}, wrong pixels per step {counts['pixels_off'] / n:.2f}")


def pong_check(wm: Path, agent: Path, loops: int, steps: int = 5000, seed: int = 0) -> None:
    """Pong: let an agent play in the dream, read the game state back out of every dreamed frame and
    compare each dreamed step with the real rules applied to that state. After a point the serve is
    random, so only the paddles, the score and the ball's direction of travel are compared there."""
    from tinyworldformer.games import pong as pg
    from tinyworldformer.training.streamx import DreamEnv, Net, features

    observe, obs_dim = features("pong")
    ck = torch.load(agent, weights_only=False)
    pi = Net(obs_dim, pg.NUM_ACTIONS, ck["hidden"])
    pi.load_state_dict(ck["pi"])
    mean, inv = ck["obs_mean"], 1 / np.sqrt(ck["obs_var"] + 1e-8)

    counts = dict(steps=0, unreadable=0, state_wrong=0, reward_wrong=0, pixels_off=0)
    key = lambda s: (s.player, s.cpu, s.score, s.lost)  # noqa: E731
    env = DreamEnv(wm, seed, loops=loops)
    grid = env.reset()
    score = 0.0
    for _ in range(steps):
        x = torch.as_tensor((observe(grid) - mean) * inv, dtype=torch.float32)
        with torch.no_grad():
            a = int(pi(x).argmax())
        before = pg.read(grid)
        grid, r, done = env.step(a)
        score += r
        counts["steps"] += 1
        after = pg.read(grid)
        if before is None or after is None:
            counts["unreadable"] += 1
        else:
            real, rr, _ = pg.step(before, a, np.random.default_rng(0))
            counts["reward_wrong"] += rr != r
            if real.random_event:
                counts["state_wrong"] += key(real) != key(after) or (real.bx, real.dx) != (after.bx, after.dx)
            else:
                counts["state_wrong"] += key(real) != key(after) or (real.bx, real.by, real.dx, real.dy) != (after.bx, after.by, after.dx, after.dy)
                counts["pixels_off"] += int((real.grid != grid).sum())
        if done:
            grid = env.reset()
    n = counts["steps"]
    print(f"{wm.name} ({loops} loops), agent {agent.name}: dream score {score:.0f} over {n} steps | "
          f"unreadable frames {counts['unreadable']}, wrong state {counts['state_wrong']}, "
          f"wrong reward {counts['reward_wrong']}, wrong pixels per step {counts['pixels_off'] / n:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("models/wm.pt"))
    p.add_argument("--data", type=Path, default=None, help="held-out transitions (default: data/<game>/val.npz)")
    p.add_argument("--agent", type=Path, default=None, help="only run the glitch check with this agent")
    p.add_argument("--loops", type=int, default=2)
    args = p.parse_args()
    if args.agent:
        game = torch.load(args.ckpt, map_location="cpu")["config"].get("game")
        if game == "doomlite":
            return doom_check(args.ckpt, args.agent, args.loops)
        if game == "flappy":
            return flappy_check(args.ckpt, args.agent, args.loops)
        if game == "pacman":
            return pacman_check(args.ckpt, args.agent, args.loops)
        if game == "pong":
            return pong_check(args.ckpt, args.agent, args.loops)
        return glitches(args.ckpt, args.agent, args.loops)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.ckpt).to(dev)
    ActQuant.enabled = True
    d = Transitions(args.data or Path("data") / model.g.name / "val.npz").d
    if model.g.name == "breakout":
        for loops in range(1, model.c.loops + 1):
            error_breakdown(model, d, loops, dev)
    for loops in range(1, model.c.loops + 1):
        rollouts(model, d, loops, dev)


if __name__ == "__main__":
    main()
