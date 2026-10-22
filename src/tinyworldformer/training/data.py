"""Roll out scripted agents in the real engine and save (frame, action) -> next frame transitions.

    python -m tinyworldformer.training.data --transitions 1000000                 # -> data/breakout/
    python -m tinyworldformer.training.data --game doomlite --varied 0.5          # -> data/doomlite/
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tinyworldformer import games
from tinyworldformer.games import agents, breakout as b
from tinyworldformer.training.streamx import Net, features


def rollout(game: str, agent: str, rng: np.random.Generator, max_steps: int, varied: bool = False) -> dict[str, np.ndarray]:
    """One episode. `varied` starts from a less usual state for coverage (Breakout: bricks with
    holes; Doom-lite: random start cell and facing)."""
    g = games.get(game)
    policy = agents.agents_for(game)[agent][0](rng)
    state = g.module.reset(rng, varied)
    rows = {k: [] for k in ("grid", "action", "next_grid", "reward", "done", "lives", "next_lives", "random")}
    for _ in range(max_steps):
        action = policy(state)
        nxt, reward, done = g.module.step(state, action, rng)
        random = getattr(nxt, "random_event", reward in g.random_rewards)
        for k, v in zip(rows, (state.grid, action, nxt.grid, reward, done, state.lives, nxt.lives, random)):
            rows[k].append(v)
        state = nxt
        if done:
            break
    dtypes = dict(grid=np.uint8, action=np.uint8, next_grid=np.uint8, reward=np.float32,
                  done=np.bool_, lives=np.uint8, next_lives=np.uint8, random=np.bool_)
    return {k: np.asarray(v, dtypes[k]) for k, v in rows.items()}


def generate(transitions: int, max_steps: int, seed: int, varied: float = 0.0, game: str = "breakout") -> dict[str, np.ndarray]:
    """Each agent fills its share of `transitions` (the weights in agents.agents_for(game))."""
    rng = np.random.default_rng(seed)
    chunks = []
    table = agents.agents_for(game)
    for i, name in enumerate(table):
        quota, got = table[name][1] * transitions, 0
        while got < quota:
            chunk = rollout(game, name, rng, max_steps, varied=rng.random() < varied)
            chunk["agent"] = np.full(len(chunk["action"]), i, np.uint8)
            chunks.append(chunk)
            got += len(chunk["action"])
    # shuffle episodes so a split by episode id mixes agents
    for ep, k in enumerate(rng.permutation(len(chunks))):
        n = len(chunks[k]["action"])
        chunks[k]["episode"] = np.full(n, ep, np.int32)
        chunks[k]["t"] = np.arange(n, dtype=np.int32)
    chunks.sort(key=lambda c: c["episode"][0])
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def stats(d: dict[str, np.ndarray], game: str) -> str:
    g = games.get(game)
    n = len(d["action"])
    lines = [
        f"transitions {n}, episodes {len(np.unique(d['episode']))}",
        "actions   " + " ".join(f"{a}:{(d['action'] == a).mean():.2f}" for a in range(g.A)),
        "agents    " + " ".join(f"{nm}:{(d['agent'] == i).mean():.2f}" for i, nm in enumerate(agents.agents_for(game))),
        f"reward>0 {(d['reward'] > 0).mean():.4f}  reward<0 {(d['reward'] < 0).mean():.4f}  "
        f"random event {d['random'].mean():.4f}  done {d['done'].mean():.5f}",
    ]
    if game == "breakout":
        bricks = (d["grid"] == b.BRICK).sum((1, 2))
        lines.append(f"bricks left: mean {bricks.mean():.1f}, min {bricks.min()}, max {bricks.max()}")
    return "\n".join(lines)


def add_learned(ckpt: str, game: str = "breakout") -> None:
    """Register a trained stream-x policy (greedy, and with per-episode epsilon noise) and shift
    most of the data share to it, so the dataset covers the states a good player reaches."""
    observe, obs_dim = features(game)
    n_actions = games.get(game).A
    ck = torch.load(ckpt, weights_only=False)
    net = Net(obs_dim, n_actions, ck["hidden"])
    net.load_state_dict(ck["pi"])
    mean, inv_std = ck["obs_mean"], 1 / np.sqrt(ck["obs_var"] + 1e-8)
    table = agents.agents_for(game)

    def learned(rng: np.random.Generator, eps: float) -> agents.Policy:
        def policy(s) -> int:
            if rng.random() < eps:
                return int(rng.integers(n_actions))
            x = torch.as_tensor((observe(s.grid) - mean) * inv_std, dtype=torch.float32)
            with torch.no_grad():
                return int(net(x).argmax())

        return policy

    for name in list(table):  # scripted agents keep 40% of the data between them
        table[name] = (table[name][0], table[name][1] * 0.4)
    table["learned"] = (lambda rng: learned(rng, 0.02), 0.3)
    table["noisy_learned"] = (lambda rng: learned(rng, rng.uniform(0.0, 0.3)), 0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default: data/<game>")
    p.add_argument("--game", choices=list(games.GAMES), default="breakout")
    p.add_argument("--learned", type=str, default=None, help="stream-x checkpoint to collect with")
    p.add_argument("--varied", "--holes", type=float, default=0.0,
                   help="share of episodes from varied starts (Breakout: bricks with holes; Doom-lite: random start)")
    args = p.parse_args()
    args.out = args.out or Path("data") / args.game
    if args.learned:
        add_learned(args.learned, args.game)

    d = generate(args.transitions, args.max_steps, args.seed, args.varied, args.game)
    val = d["episode"] >= int(d["episode"].max() * (1 - args.val_frac))  # split by episode, never by step
    args.out.mkdir(parents=True, exist_ok=True)
    for name, mask in (("train", ~val), ("val", val)):
        part = {k: v[mask] for k, v in d.items()}
        np.savez_compressed(args.out / f"{name}.npz", game=args.game, **part)
        print(f"--- {name}\n{stats(part, args.game)}")


if __name__ == "__main__":
    main()
